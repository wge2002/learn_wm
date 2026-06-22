"""Measure LeWM latent rollout drift on fixed PushT trajectories.

This is the Phase 2 counterpart to the behavior-level LGHL sweep. It removes
the CEM planner from the loop: given real dataset actions, replay the same
actions in the environment, encode the rendered observations, and compare those
true future latents with the model's autoregressive latent rollout.
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
import matplotlib.pyplot as plt
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
        print(f"[phase2] ignoring invalid SWM_TORCH_THREADS={raw!r}")
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    print(f"[phase2] torch CPU threads set to {threads}")


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

    for chunk in chunks:
        if chunk["action"].shape[0] < max_env_steps:
            raise ValueError("Loaded action chunk shorter than max_env_steps")
        if chunk["state"].shape[0] <= goal_offset:
            raise ValueError("Loaded state chunk shorter than goal_offset")

    if max_env_steps % action_block != 0:
        raise ValueError("max_env_steps must be divisible by action_block")

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

    init_states = []
    goal_states = []
    raw_actions = []
    for chunk in chunks:
        state = to_numpy(chunk["state"])
        action = to_numpy(chunk["action"])
        init_states.append(state[0])
        goal_states.append(state[goal_offset])
        raw_actions.append(action[:max_env_steps])

    init_states_arr = np.asarray(init_states, dtype=np.float32)
    goal_states_arr = np.asarray(goal_states, dtype=np.float32)
    raw_actions_arr = np.asarray(raw_actions, dtype=np.float32)

    flat_actions = raw_actions_arr.reshape(-1, raw_actions_arr.shape[-1])
    model_actions = action_scaler.transform(flat_actions).reshape(
        num_samples, max_k, action_block * raw_actions_arr.shape[-1]
    )

    return WindowBatch(
        episodes=episodes,
        starts=starts,
        init_states=init_states_arr,
        goal_states=goal_states_arr,
        raw_actions=raw_actions_arr,
        model_actions=model_actions.astype(np.float32),
    )


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def set_state_and_goal(env, init_state: np.ndarray, goal_state: np.ndarray) -> None:
    raw = env.unwrapped
    raw._set_goal_state(goal_state)
    if hasattr(raw, "goal_pose"):
        raw.goal_pose = np.asarray(
            [goal_state[2], goal_state[3], goal_state[4]], dtype=np.float64
        )
    raw._set_state(init_state)


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
) -> tuple[np.ndarray, np.ndarray]:
    n, max_env_steps, _ = raw_actions.shape
    max_k = max_env_steps // action_block
    frames = np.empty((n, max_k + 1, img_size, img_size, 3), dtype=np.uint8)
    terminated = np.zeros((n, max_k + 1), dtype=bool)

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
            done_seen = False

            for t in range(max_env_steps):
                _, _, done, truncated, _ = env.step(raw_actions[i, t])
                done_seen = done_seen or bool(done or truncated)
                if (t + 1) % action_block == 0:
                    k = (t + 1) // action_block
                    frames[i, k] = env.render()
                    terminated[i, k] = done_seen
    finally:
        env.close()

    return frames, terminated


def images_to_tensor(frames: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float().div_(255.0)
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats["mean"], dtype=x.dtype).view(1, 1, 3, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=x.dtype).view(1, 1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def encode_and_rollout(
    *,
    model,
    frames: np.ndarray,
    model_actions: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = next(model.parameters()).dtype
    n = frames.shape[0]
    true_chunks = []
    pred_chunks = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        pixels = images_to_tensor(frames[start:end]).to(device=device, dtype=dtype)
        actions = torch.from_numpy(model_actions[start:end]).to(
            device=device, dtype=dtype
        )

        true_info = model.encode({"pixels": pixels})
        true_emb = true_info["emb"]

        rollout_info = {
            "pixels": pixels[:, None, :1],
        }
        rollout_out = model.rollout(rollout_info, actions[:, None])
        pred_emb = rollout_out["predicted_emb"][:, 0, : true_emb.shape[1]]

        true_chunks.append(true_emb.float().cpu().numpy())
        pred_chunks.append(pred_emb.float().cpu().numpy())

    return np.concatenate(true_chunks, axis=0), np.concatenate(pred_chunks, axis=0)


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


def first_doubling(values: np.ndarray) -> float:
    if values.shape[0] <= 2 or not np.isfinite(values[1]) or values[1] <= 0:
        return float("nan")
    threshold = 2.0 * values[1]
    for k in range(1, values.shape[0]):
        if values[k] >= threshold:
            return float(k)
    return float("nan")


def summarize_metric(
    rows: list[dict],
    *,
    shift: str,
    metric_name: str,
    values: np.ndarray,
    action_block: int,
) -> dict:
    mean_curve = np.nanmean(values, axis=0)
    taus = np.array([first_doubling(row) for row in values], dtype=np.float32)
    finite_taus = taus[np.isfinite(taus)]

    for k in range(values.shape[1]):
        col = values[:, k]
        rows.append(
            {
                "kind": "rollout_error",
                "shift": shift,
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

    return {
        "metric": metric_name,
        "mean_curve": mean_curve.tolist(),
        "mean_curve_tau_k": first_doubling(mean_curve),
        "sample_tau_k_mean": float(np.nanmean(finite_taus))
        if finite_taus.size
        else float("nan"),
        "sample_tau_k_median": float(np.nanmedian(finite_taus))
        if finite_taus.size
        else float("nan"),
        "sample_tau_k_valid_fraction": float(finite_taus.size / len(taus)),
    }


def summarize_encoder_shift(
    rows: list[dict],
    *,
    shift: str,
    id_true: np.ndarray,
    shift_true: np.ndarray,
    action_block: int,
) -> None:
    metrics = metric_arrays(shift_true, id_true)
    for metric_name, values in metrics.items():
        for k in range(values.shape[1]):
            col = values[:, k]
            rows.append(
                {
                    "kind": "encoder_shift",
                    "shift": shift,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "kind",
        "shift",
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


def plot_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind, ylabel, filename in [
        ("rollout_error", "latent rollout error", "phase2_rollout_error.png"),
        ("encoder_shift", "encoder shift vs ID", "phase2_encoder_shift.png"),
    ]:
        subset = [
            r for r in rows if r["kind"] == kind and r["metric"] in {"mse", "cosine"}
        ]
        if not subset:
            continue

        metrics = ["mse", "cosine"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, metric in zip(axes, metrics):
            metric_rows = [r for r in subset if r["metric"] == metric]
            shifts = sorted({r["shift"] for r in metric_rows})
            for shift in shifts:
                group = sorted(
                    [r for r in metric_rows if r["shift"] == shift],
                    key=lambda r: r["env_steps"],
                )
                ax.plot(
                    [r["env_steps"] for r in group],
                    [r["mean"] for r in group],
                    marker="o",
                    label=shift,
                )
            ax.set_title(metric)
            ax.set_xlabel("env steps since grounding")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lghl_phase2")
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
    shifts = (
        [parse_shift(s) for s in args.shift]
        if args.shift
        else list(DEFAULT_SHIFTS)
    )

    print("[phase2] loading dataset")
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

    print("[phase2] loading model")
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
        "shifts": {label: list(vars_) for label, vars_ in shifts},
        "episodes": batch.episodes.tolist(),
        "starts": batch.starts.tolist(),
    }
    (output_dir / "phase2_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    rows: list[dict] = []
    summary = {
        "metadata": {k: v for k, v in metadata.items() if k not in {"episodes", "starts"}},
        "rollout": {},
    }
    true_by_shift: dict[str, np.ndarray] = {}
    pred_by_shift: dict[str, np.ndarray] = {}
    terminated_by_shift: dict[str, np.ndarray] = {}

    for label, variations in shifts:
        start_time = time.time()
        print(f"[phase2] rendering replay for shift={label} variations={variations}")
        frames, terminated = render_replay(
            env_name=args.env_name,
            init_states=batch.init_states,
            goal_states=batch.goal_states,
            raw_actions=batch.raw_actions,
            variations=variations,
            action_block=args.action_block,
            img_size=args.img_size,
            seed=args.seed,
        )
        print(f"[phase2] encoding and rolling out shift={label}")
        true_emb, pred_emb = encode_and_rollout(
            model=model,
            frames=frames,
            model_actions=batch.model_actions,
            batch_size=args.batch_size,
            device=device,
        )
        true_by_shift[label] = true_emb
        pred_by_shift[label] = pred_emb
        terminated_by_shift[label] = terminated

        np.savez_compressed(
            output_dir / f"phase2_{label}_embeddings.npz",
            true_emb=true_emb,
            pred_emb=pred_emb,
            terminated=terminated,
        )

        metrics = metric_arrays(pred_emb, true_emb)
        summary["rollout"][label] = {
            "terminated_fraction_by_k": terminated.mean(axis=0).tolist(),
            "metrics": {},
            "elapsed_sec": time.time() - start_time,
        }
        for metric_name, values in metrics.items():
            summary["rollout"][label]["metrics"][metric_name] = summarize_metric(
                rows,
                shift=label,
                metric_name=metric_name,
                values=values,
                action_block=args.action_block,
            )

    if "id" in true_by_shift:
        summary["encoder_shift"] = {}
        for label, _ in shifts:
            if label == "id":
                continue
            summarize_encoder_shift(
                rows,
                shift=label,
                id_true=true_by_shift["id"],
                shift_true=true_by_shift[label],
                action_block=args.action_block,
            )
            enc_metrics = metric_arrays(true_by_shift[label], true_by_shift["id"])
            summary["encoder_shift"][label] = {
                metric: {
                    "mean_curve": np.nanmean(values, axis=0).tolist(),
                    "median_curve": np.nanmedian(values, axis=0).tolist(),
                }
                for metric, values in enc_metrics.items()
            }

    write_csv(output_dir / "phase2_latent_drift_summary.csv", rows)
    (output_dir / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    plot_results(rows, output_dir)

    print(f"[phase2] wrote results to {output_dir}")


if __name__ == "__main__":
    main()
