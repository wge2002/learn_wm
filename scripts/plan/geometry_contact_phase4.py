"""Phase 4 geometry contact/event analysis for PushT.

This script checks when geometry replay diverges in physical state space and
whether the divergence aligns with contact events. It does not load LeWM.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_worldmodel as swm


DEFAULT_SHIFTS = (
    ("id", ()),
    (
        "geometry",
        ("block.scale", "agent.scale", "block.shape", "goal.scale"),
    ),
)


@dataclass(frozen=True)
class WindowBatch:
    episodes: np.ndarray
    starts: np.ndarray
    init_states: np.ndarray
    goal_states: np.ndarray
    raw_actions: np.ndarray


def parse_shift(raw: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in raw:
        return raw, ()
    label, values = raw.split(":", 1)
    return label, tuple(v for v in values.split(",") if v)


def reset_options(variations: tuple[str, ...]) -> dict | None:
    if not variations:
        return None
    return {"variation": list(variations)}


def to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sample_windows(
    dataset,
    *,
    num_samples: int,
    max_env_steps: int,
    goal_offset: int,
    seed: int,
) -> WindowBatch:
    required = max(max_env_steps, goal_offset) + 1
    lengths = np.asarray(dataset.lengths)
    valid_eps = np.flatnonzero(lengths > required)
    if len(valid_eps) == 0:
        raise ValueError(f"No episodes are long enough for {required=}")

    rng = np.random.default_rng(seed)
    episodes = rng.choice(valid_eps, size=num_samples, replace=True)
    starts = np.array(
        [rng.integers(0, lengths[ep] - required) for ep in episodes],
        dtype=np.int64,
    )
    chunks = dataset.load_chunk(episodes, starts, starts + required)

    init_states = []
    goal_states = []
    raw_actions = []
    for chunk in chunks:
        state = to_numpy(chunk["state"])
        action = to_numpy(chunk["action"])
        init_states.append(state[0])
        goal_states.append(state[goal_offset])
        raw_actions.append(action[:max_env_steps])

    return WindowBatch(
        episodes=episodes,
        starts=starts,
        init_states=np.asarray(init_states, dtype=np.float32),
        goal_states=np.asarray(goal_states, dtype=np.float32),
        raw_actions=np.asarray(raw_actions, dtype=np.float32),
    )


def set_state_and_goal(env, state: np.ndarray, goal_state: np.ndarray) -> None:
    raw = env.unwrapped
    raw._set_goal_state(goal_state)
    if hasattr(raw, "goal_pose"):
        raw.goal_pose = np.asarray(
            [goal_state[2], goal_state[3], goal_state[4]], dtype=np.float64
        )
    raw._set_state(state)


def get_env_state(env) -> np.ndarray:
    return np.asarray(env.unwrapped._get_obs(), dtype=np.float32)


def replay_with_contacts(
    *,
    env_name: str,
    init_states: np.ndarray,
    goal_states: np.ndarray,
    raw_actions: np.ndarray,
    variations: tuple[str, ...],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, max_env_steps, _ = raw_actions.shape
    states = np.empty((n, max_env_steps + 1, 7), dtype=np.float32)
    contacts = np.zeros((n, max_env_steps + 1), dtype=np.float32)
    terminated = np.zeros((n, max_env_steps + 1), dtype=bool)
    rewards = np.zeros((n, max_env_steps + 1), dtype=np.float32)
    env = gym.make(
        env_name,
        max_episode_steps=max_env_steps + 5,
        render_mode="rgb_array",
    )
    options = reset_options(variations)
    try:
        for i in range(n):
            env.reset(seed=seed + i, options=options)
            set_state_and_goal(env, init_states[i], goal_states[i])
            states[i, 0] = get_env_state(env)
            done_seen = False
            for t in range(max_env_steps):
                _, reward, done, truncated, _ = env.step(raw_actions[i, t])
                done_seen = done_seen or bool(done or truncated)
                states[i, t + 1] = get_env_state(env)
                contacts[i, t + 1] = float(
                    getattr(env.unwrapped, "n_contact_points", 0)
                )
                terminated[i, t + 1] = done_seen
                rewards[i, t + 1] = float(reward)
    finally:
        env.close()
    return states, contacts, terminated, rewards


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def divergence_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    agent_xy_l2 = np.linalg.norm(a[..., :2] - b[..., :2], axis=-1)
    block_xy_l2 = np.linalg.norm(a[..., 2:4] - b[..., 2:4], axis=-1)
    angle_abs = np.abs(wrap_angle(a[..., 4] - b[..., 4]))
    pose_l2 = np.sqrt(agent_xy_l2**2 + block_xy_l2**2 + angle_abs**2)
    return {
        "agent_xy_l2": agent_xy_l2,
        "block_xy_l2": block_xy_l2,
        "angle_abs": angle_abs,
        "pose_l2": pose_l2,
    }


def first_true(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[0], -1, dtype=np.int32)
    any_true = mask.any(axis=1)
    out[any_true] = mask[any_true].argmax(axis=1)
    return out


def summarize_curve(
    rows: list[dict],
    *,
    kind: str,
    metric: str,
    values: np.ndarray,
    action_block: int,
) -> dict:
    mean_curve = np.nanmean(values, axis=0)
    for env_step in range(values.shape[1]):
        col = values[:, env_step]
        rows.append(
            {
                "kind": kind,
                "metric": metric,
                "env_step": env_step,
                "k": env_step / action_block,
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col)),
                "median": float(np.nanmedian(col)),
                "p25": float(np.nanpercentile(col, 25)),
                "p75": float(np.nanpercentile(col, 75)),
                "n": int(np.isfinite(col).sum()),
            }
        )
    return {
        "mean_curve": mean_curve.tolist(),
        "median_curve": np.nanmedian(values, axis=0).tolist(),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "metric",
        "env_step",
        "k",
        "mean",
        "std",
        "median",
        "p25",
        "p75",
        "n",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-env-steps", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=50)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lghl_phase4_contact")
    parser.add_argument("--shift", action="append", default=None)
    args = parser.parse_args()

    shifts = (
        [parse_shift(s) for s in args.shift]
        if args.shift
        else list(DEFAULT_SHIFTS)
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = sample_windows(
        dataset,
        num_samples=args.num_samples,
        max_env_steps=args.max_env_steps,
        goal_offset=args.goal_offset,
        seed=args.seed,
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "env_name": args.env_name,
        "num_samples": args.num_samples,
        "max_env_steps": args.max_env_steps,
        "goal_offset": args.goal_offset,
        "action_block": args.action_block,
        "seed": args.seed,
        "shifts": {label: list(vars_) for label, vars_ in shifts},
        "episodes": batch.episodes.tolist(),
        "starts": batch.starts.tolist(),
    }
    (output_dir / "phase4_contact_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    replay = {}
    for label, variations in shifts:
        print(f"[contact] replay shift={label} variations={variations}", flush=True)
        states, contacts, terminated, rewards = replay_with_contacts(
            env_name=args.env_name,
            init_states=batch.init_states,
            goal_states=batch.goal_states,
            raw_actions=batch.raw_actions,
            variations=variations,
            seed=args.seed,
        )
        replay[label] = {
            "states": states,
            "contacts": contacts,
            "terminated": terminated,
            "rewards": rewards,
        }
        np.savez_compressed(
            output_dir / f"phase4_contact_{label}.npz",
            states=states,
            contacts=contacts,
            terminated=terminated,
            rewards=rewards,
        )

    if "id" not in replay or "geometry" not in replay:
        raise ValueError("contact analysis requires id and geometry shifts")

    id_data = replay["id"]
    geo_data = replay["geometry"]
    metrics = divergence_metrics(geo_data["states"], id_data["states"])
    id_first_contact = first_true(id_data["contacts"] > 0)
    geo_first_contact = first_true(geo_data["contacts"] > 0)
    first_div_5 = first_true(metrics["block_xy_l2"] >= 5.0)
    first_div_20 = first_true(metrics["block_xy_l2"] >= 20.0)

    rows: list[dict] = []
    summary = {
        "metadata": {
            k: v for k, v in metadata.items() if k not in {"episodes", "starts"}
        },
        "curves": {},
        "events": {
            "id_first_contact_step": id_first_contact.tolist(),
            "geometry_first_contact_step": geo_first_contact.tolist(),
            "first_block_divergence_ge_5_step": first_div_5.tolist(),
            "first_block_divergence_ge_20_step": first_div_20.tolist(),
        },
    }

    for metric, values in metrics.items():
        summary["curves"][metric] = summarize_curve(
            rows,
            kind="geometry_vs_id",
            metric=metric,
            values=values,
            action_block=args.action_block,
        )
    summary["curves"]["id_contact_points"] = summarize_curve(
        rows,
        kind="id",
        metric="contact_points",
        values=id_data["contacts"],
        action_block=args.action_block,
    )
    summary["curves"]["geometry_contact_points"] = summarize_curve(
        rows,
        kind="geometry",
        metric="contact_points",
        values=geo_data["contacts"],
        action_block=args.action_block,
    )

    event_arrays = {
        "id_first_contact_step": id_first_contact,
        "geometry_first_contact_step": geo_first_contact,
        "first_block_divergence_ge_5_step": first_div_5,
        "first_block_divergence_ge_20_step": first_div_20,
    }
    summary["event_stats"] = {}
    for name, arr in event_arrays.items():
        valid = arr[arr >= 0]
        summary["event_stats"][name] = {
            "valid_fraction": float(len(valid) / len(arr)),
            "mean": float(np.mean(valid)) if len(valid) else float("nan"),
            "median": float(np.median(valid)) if len(valid) else float("nan"),
            "p25": float(np.percentile(valid, 25)) if len(valid) else float("nan"),
            "p75": float(np.percentile(valid, 75)) if len(valid) else float("nan"),
        }

    write_csv(output_dir / "phase4_contact_summary.csv", rows)
    (output_dir / "phase4_contact_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[contact] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
