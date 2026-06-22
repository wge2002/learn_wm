"""Phase 3 latent grounding controls for PushT LeWM.

This script separates three effects that were coupled in Phase 2:

1. same-state encoder shift: render the same dataset state under each FoV.
2. regrounded rollout error: predict with periodic real-observation grounding.
3. replay state divergence: quantify how shifted replay physics departs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from sklearn import preprocessing


DEFAULT_SHIFTS = (
    ("id", ()),
    (
        "visual",
        ("background.color", "agent.color", "block.color", "goal.color"),
    ),
    (
        "geometry",
        ("block.scale", "agent.scale", "block.shape", "goal.scale"),
    ),
)


@dataclass(frozen=True)
class WindowBatch:
    episodes: np.ndarray
    starts: np.ndarray
    states_by_k: np.ndarray
    init_states: np.ndarray
    goal_states: np.ndarray
    raw_actions: np.ndarray
    model_actions: np.ndarray


def configure_torch_threads_from_env() -> None:
    raw = os.environ.get("SWM_TORCH_THREADS")
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        print(f"[phase3] ignoring invalid SWM_TORCH_THREADS={raw!r}", flush=True)
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    print(f"[phase3] torch CPU threads set to {threads}", flush=True)


def parse_shift(raw: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in raw:
        return raw, ()
    label, values = raw.split(":", 1)
    variations = tuple(v for v in values.split(",") if v)
    return label, variations


def reset_options(variations: tuple[str, ...]) -> dict | None:
    if not variations:
        return None
    return {"variation": list(variations)}


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def sample_windows(
    dataset,
    *,
    num_samples: int,
    max_env_steps: int,
    goal_offset: int,
    action_block: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    required = max(max_env_steps, goal_offset) + 1
    lengths = np.asarray(dataset.lengths)
    valid_eps = np.flatnonzero(lengths > required)
    if len(valid_eps) == 0:
        raise ValueError(
            f"No episodes are long enough for required window length {required}"
        )

    rng = np.random.default_rng(seed)
    episodes = rng.choice(valid_eps, size=num_samples, replace=True)
    starts = np.array(
        [rng.integers(0, lengths[ep] - required) for ep in episodes],
        dtype=np.int64,
    )
    ends = starts + required
    chunks = dataset.load_chunk(episodes, starts, ends)

    if max_env_steps % action_block != 0:
        raise ValueError("max_env_steps must be divisible by action_block")

    for chunk in chunks:
        if chunk["action"].shape[0] < max_env_steps:
            raise ValueError("Loaded action chunk shorter than max_env_steps")
        if chunk["state"].shape[0] <= max_env_steps:
            raise ValueError("Loaded state chunk shorter than max_env_steps")
        if chunk["state"].shape[0] <= goal_offset:
            raise ValueError("Loaded state chunk shorter than goal_offset")

    return episodes, starts, chunks


def build_window_batch(
    dataset,
    *,
    num_samples: int,
    max_k: int,
    goal_offset: int,
    action_block: int,
    seed: int,
) -> WindowBatch:
    max_env_steps = max_k * action_block
    episodes, starts, chunks = sample_windows(
        dataset,
        num_samples=num_samples,
        max_env_steps=max_env_steps,
        goal_offset=goal_offset,
        action_block=action_block,
        seed=seed,
    )

    action_scaler = preprocessing.StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler.fit(action_data)

    state_indices = np.arange(0, max_env_steps + 1, action_block)
    states_by_k = []
    goal_states = []
    raw_actions = []
    for chunk in chunks:
        state = to_numpy(chunk["state"])
        action = to_numpy(chunk["action"])
        states_by_k.append(state[state_indices])
        goal_states.append(state[goal_offset])
        raw_actions.append(action[:max_env_steps])

    states_by_k_arr = np.asarray(states_by_k, dtype=np.float32)
    goal_states_arr = np.asarray(goal_states, dtype=np.float32)
    raw_actions_arr = np.asarray(raw_actions, dtype=np.float32)

    flat_actions = raw_actions_arr.reshape(-1, raw_actions_arr.shape[-1])
    model_actions = action_scaler.transform(flat_actions).reshape(
        num_samples, max_k, action_block * raw_actions_arr.shape[-1]
    )

    return WindowBatch(
        episodes=episodes,
        starts=starts,
        states_by_k=states_by_k_arr,
        init_states=states_by_k_arr[:, 0],
        goal_states=goal_states_arr,
        raw_actions=raw_actions_arr,
        model_actions=model_actions.astype(np.float32),
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
    raw = env.unwrapped
    if hasattr(raw, "_get_obs"):
        return np.asarray(raw._get_obs(), dtype=np.float32)
    raise AttributeError("Environment does not expose _get_obs for state capture")


def render_state_sequence(
    *,
    env_name: str,
    states_by_k: np.ndarray,
    goal_states: np.ndarray,
    variations: tuple[str, ...],
    img_size: int,
    seed: int,
) -> np.ndarray:
    n, num_k = states_by_k.shape[:2]
    frames = np.empty((n, num_k, img_size, img_size, 3), dtype=np.uint8)
    env = gym.make(
        env_name,
        max_episode_steps=max(10, num_k + 5),
        render_mode="rgb_array",
        resolution=img_size,
    )
    options = reset_options(variations)

    try:
        for i in range(n):
            env.reset(seed=seed + i, options=options)
            for k in range(num_k):
                set_state_and_goal(env, states_by_k[i, k], goal_states[i])
                frames[i, k] = env.render()
    finally:
        env.close()

    return frames


def render_replay(
    *,
    env_name: str,
    init_states: np.ndarray,
    goal_states: np.ndarray,
    raw_actions: np.ndarray,
    variations: tuple[str, ...],
    action_block: int,
    img_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, max_env_steps, _ = raw_actions.shape
    max_k = max_env_steps // action_block
    frames = np.empty((n, max_k + 1, img_size, img_size, 3), dtype=np.uint8)
    terminated = np.zeros((n, max_k + 1), dtype=bool)
    replay_states = np.empty((n, max_k + 1, 7), dtype=np.float32)

    env = gym.make(
        env_name,
        max_episode_steps=max_env_steps + 5,
        render_mode="rgb_array",
        resolution=img_size,
    )
    options = reset_options(variations)

    try:
        for i in range(n):
            env.reset(seed=seed + i, options=options)
            set_state_and_goal(env, init_states[i], goal_states[i])
            frames[i, 0] = env.render()
            replay_states[i, 0] = get_env_state(env)
            done_seen = False

            for t in range(max_env_steps):
                _, _, done, truncated, _ = env.step(raw_actions[i, t])
                done_seen = done_seen or bool(done or truncated)
                if (t + 1) % action_block == 0:
                    k = (t + 1) // action_block
                    frames[i, k] = env.render()
                    replay_states[i, k] = get_env_state(env)
                    terminated[i, k] = done_seen
    finally:
        env.close()

    return frames, terminated, replay_states


def images_to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float().div_(255.0)
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def encode_frames(
    *,
    model,
    frames: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    chunks = []
    for start in range(0, frames.shape[0], batch_size):
        end = min(start + batch_size, frames.shape[0])
        pixels = images_to_tensor(frames[start:end]).to(device=device, dtype=dtype)
        chunks.append(model.encode({"pixels": pixels})["emb"].float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.inference_mode()
def regrounded_rollout(
    *,
    model,
    frames: np.ndarray,
    true_emb: np.ndarray,
    model_actions: np.ndarray,
    interval: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if interval < 1:
        raise ValueError("interval must be >= 1")

    dtype = next(model.parameters()).dtype
    n, num_k = true_emb.shape[:2]
    max_k = num_k - 1
    pred = np.empty_like(true_emb)
    pred[:, 0] = true_emb[:, 0]

    for segment_start in range(0, max_k, interval):
        horizon = min(interval, max_k - segment_start)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            pixels = images_to_tensor(
                frames[start:end, segment_start : segment_start + 1]
            ).to(device=device, dtype=dtype)
            actions = torch.from_numpy(
                model_actions[start:end, segment_start : segment_start + horizon]
            ).to(device=device, dtype=dtype)
            rollout_out = model.rollout(
                {"pixels": pixels[:, None]},
                actions[:, None],
            )
            segment_pred = rollout_out["predicted_emb"][:, 0, 1 : horizon + 1]
            pred[
                start:end,
                segment_start + 1 : segment_start + horizon + 1,
            ] = segment_pred.float().cpu().numpy()

    return pred


def metric_arrays(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    diff = pred - true
    mse = np.mean(diff * diff, axis=-1)
    l2 = np.linalg.norm(diff, axis=-1)

    pred_t = torch.from_numpy(pred.reshape(-1, pred.shape[-1]))
    true_t = torch.from_numpy(true.reshape(-1, true.shape[-1]))
    cosine = (
        1.0
        - F.cosine_similarity(pred_t, true_t, dim=-1).numpy().reshape(pred.shape[:2])
    )
    return {"mse": mse, "l2": l2, "cosine": cosine}


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def state_metric_arrays(a: np.ndarray, b: np.ndarray) -> dict[str, np.ndarray]:
    a_core = a[..., :5]
    b_core = b[..., :5]
    agent_xy_l2 = np.linalg.norm(a_core[..., :2] - b_core[..., :2], axis=-1)
    block_xy_l2 = np.linalg.norm(a_core[..., 2:4] - b_core[..., 2:4], axis=-1)
    angle_abs = np.abs(wrap_angle(a_core[..., 4] - b_core[..., 4]))
    pose_l2 = np.sqrt(agent_xy_l2**2 + block_xy_l2**2 + angle_abs**2)
    return {
        "pose_l2": pose_l2,
        "agent_xy_l2": agent_xy_l2,
        "block_xy_l2": block_xy_l2,
        "angle_abs": angle_abs,
    }


def summarize_values(
    rows: list[dict],
    *,
    kind: str,
    shift: str,
    metric_name: str,
    values: np.ndarray,
    action_block: int,
    interval: int | None = None,
    reference: str = "",
    k_start: int = 0,
) -> dict:
    summary = {
        "mean_curve": np.nanmean(values, axis=0).tolist(),
        "median_curve": np.nanmedian(values, axis=0).tolist(),
    }
    for local_k in range(values.shape[1]):
        k = local_k + k_start
        col = values[:, local_k]
        rows.append(
            {
                "kind": kind,
                "shift": shift,
                "reference": reference,
                "interval": "" if interval is None else int(interval),
                "metric": metric_name,
                "k": k,
                "env_steps": k * action_block,
                "mean": float(np.nanmean(col)),
                "std": float(np.nanstd(col)),
                "median": float(np.nanmedian(col)),
                "p25": float(np.nanpercentile(col, 25)),
                "p75": float(np.nanpercentile(col, 75)),
                "n": int(np.isfinite(col).sum()),
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "shift",
        "reference",
        "interval",
        "metric",
        "k",
        "env_steps",
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


def parse_intervals(raw: str, max_k: int) -> list[int]:
    intervals = sorted({int(x) for x in raw.split(",") if x})
    if any(x < 1 for x in intervals):
        raise ValueError("reground intervals must be >= 1")
    if any(x > max_k for x in intervals):
        raise ValueError("reground intervals must be <= max_k")
    return intervals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--goal-offset", type=int, default=50)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lghl_phase3")
    parser.add_argument("--reground-intervals", default="1,2,3,5,10")
    parser.add_argument(
        "--shift",
        action="append",
        default=None,
        help="Shift spec LABEL:variation,variation. Defaults to id/visual/geometry.",
    )
    args = parser.parse_args()

    configure_torch_threads_from_env()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intervals = parse_intervals(args.reground_intervals, args.max_k)
    shifts = (
        [parse_shift(s) for s in args.shift]
        if args.shift
        else list(DEFAULT_SHIFTS)
    )

    start_all = time.time()
    print("[phase3] loading dataset", flush=True)
    dataset = swm.data.load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = build_window_batch(
        dataset,
        num_samples=args.num_samples,
        max_k=args.max_k,
        goal_offset=args.goal_offset,
        action_block=args.action_block,
        seed=args.seed,
    )

    print("[phase3] loading model", flush=True)
    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    metadata = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "env_name": args.env_name,
        "num_samples": args.num_samples,
        "max_k": args.max_k,
        "goal_offset": args.goal_offset,
        "action_block": args.action_block,
        "seed": args.seed,
        "device": str(device),
        "reground_intervals": intervals,
        "shifts": {label: list(vars_) for label, vars_ in shifts},
        "episodes": batch.episodes.tolist(),
        "starts": batch.starts.tolist(),
    }
    (output_dir / "phase3_metadata.json").write_text(json.dumps(metadata, indent=2))

    rows: list[dict] = []
    summary = {
        "metadata": {
            k: v for k, v in metadata.items() if k not in {"episodes", "starts"}
        },
        "same_state_encoder_shift": {},
        "regrounded_rollout_error": {},
        "replay_state_divergence": {},
    }
    same_state_emb_by_shift: dict[str, np.ndarray] = {}
    replay_state_by_shift: dict[str, np.ndarray] = {}

    for label, variations in shifts:
        shift_start = time.time()
        print(
            f"[phase3] same-state render/encode shift={label} variations={variations}",
            flush=True,
        )
        same_frames = render_state_sequence(
            env_name=args.env_name,
            states_by_k=batch.states_by_k,
            goal_states=batch.goal_states,
            variations=variations,
            img_size=args.img_size,
            seed=args.seed,
        )
        same_emb = encode_frames(
            model=model,
            frames=same_frames,
            batch_size=args.batch_size,
            device=device,
        )
        same_state_emb_by_shift[label] = same_emb
        np.savez_compressed(
            output_dir / f"phase3_{label}_same_state_embeddings.npz",
            emb=same_emb,
        )

        print(f"[phase3] replay render/encode shift={label}", flush=True)
        replay_frames, terminated, replay_states = render_replay(
            env_name=args.env_name,
            init_states=batch.init_states,
            goal_states=batch.goal_states,
            raw_actions=batch.raw_actions,
            variations=variations,
            action_block=args.action_block,
            img_size=args.img_size,
            seed=args.seed,
        )
        replay_state_by_shift[label] = replay_states
        replay_true_emb = encode_frames(
            model=model,
            frames=replay_frames,
            batch_size=args.batch_size,
            device=device,
        )

        preds_by_interval = {}
        summary["regrounded_rollout_error"][label] = {
            "terminated_fraction_by_k": terminated.mean(axis=0).tolist(),
            "intervals": {},
        }
        for interval in intervals:
            print(
                f"[phase3] regrounded rollout shift={label} interval={interval}",
                flush=True,
            )
            pred = regrounded_rollout(
                model=model,
                frames=replay_frames,
                true_emb=replay_true_emb,
                model_actions=batch.model_actions,
                interval=interval,
                batch_size=args.batch_size,
                device=device,
            )
            preds_by_interval[str(interval)] = pred
            metric_summary = {}
            for metric_name, values in metric_arrays(pred, replay_true_emb).items():
                metric_summary[metric_name] = summarize_values(
                    rows,
                    kind="regrounded_rollout_error",
                    shift=label,
                    metric_name=metric_name,
                    values=values,
                    action_block=args.action_block,
                    interval=interval,
                )
            summary["regrounded_rollout_error"][label]["intervals"][
                str(interval)
            ] = metric_summary

        np.savez_compressed(
            output_dir / f"phase3_{label}_replay_outputs.npz",
            true_emb=replay_true_emb,
            replay_states=replay_states,
            terminated=terminated,
            **{f"pred_interval_{k}": v for k, v in preds_by_interval.items()},
        )
        summary["regrounded_rollout_error"][label]["elapsed_sec"] = (
            time.time() - shift_start
        )

    if "id" in same_state_emb_by_shift:
        id_same = same_state_emb_by_shift["id"]
        for label, _ in shifts:
            if label == "id":
                continue
            metric_summary = {}
            for metric_name, values in metric_arrays(
                same_state_emb_by_shift[label], id_same
            ).items():
                metric_summary[metric_name] = summarize_values(
                    rows,
                    kind="same_state_encoder_shift",
                    shift=label,
                    reference="id_same_state",
                    metric_name=metric_name,
                    values=values,
                    action_block=args.action_block,
                )
            summary["same_state_encoder_shift"][label] = metric_summary

    if "id" in replay_state_by_shift:
        id_replay = replay_state_by_shift["id"]
        dataset_core = batch.states_by_k
        for label, _ in shifts:
            state_summary = {}
            for reference, ref_states in [
                ("id_replay", id_replay),
                ("dataset_state", dataset_core),
            ]:
                ref_summary = {}
                for metric_name, values in state_metric_arrays(
                    replay_state_by_shift[label], ref_states
                ).items():
                    ref_summary[metric_name] = summarize_values(
                        rows,
                        kind="replay_state_divergence",
                        shift=label,
                        reference=reference,
                        metric_name=metric_name,
                        values=values,
                        action_block=args.action_block,
                    )
                state_summary[reference] = ref_summary
            summary["replay_state_divergence"][label] = state_summary

    summary["elapsed_sec"] = time.time() - start_all
    write_csv(output_dir / "phase3_summary.csv", rows)
    (output_dir / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[phase3] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
