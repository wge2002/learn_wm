"""Phase 4: connect LeWM latent drift to candidate action/cost quality.

For each sampled PushT window and shift, this script compares CEM's cost
landscape when planning from the true re-grounded latent at time k versus the
open-loop drifted latent predicted from the initial observation along the
dataset action replay.

The candidate set is shared between both latents. Metrics therefore measure
how much latent drift changes the action ranking and selected action under the
same LeWM objective.
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
        print(f"[actionq] ignoring invalid SWM_TORCH_THREADS={raw!r}", flush=True)
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    print(f"[actionq] torch CPU threads set to {threads}", flush=True)


def parse_shift(raw: str) -> tuple[str, tuple[str, ...]]:
    if ":" not in raw:
        return raw, ()
    label, values = raw.split(":", 1)
    return label, tuple(v for v in values.split(",") if v)


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


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
    total_model_steps: int,
    goal_offset: int,
    action_block: int,
    seed: int,
) -> WindowBatch:
    max_env_steps = total_model_steps * action_block
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

    raw_actions_arr = np.asarray(raw_actions, dtype=np.float32)
    flat_actions = raw_actions_arr.reshape(-1, raw_actions_arr.shape[-1])
    model_actions = action_scaler.transform(flat_actions).reshape(
        num_samples,
        total_model_steps,
        action_block * raw_actions_arr.shape[-1],
    )

    return WindowBatch(
        episodes=episodes,
        starts=starts,
        init_states=np.asarray(init_states, dtype=np.float32),
        goal_states=np.asarray(goal_states, dtype=np.float32),
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


def render_replay_and_goal(
    *,
    env_name: str,
    init_states: np.ndarray,
    goal_states: np.ndarray,
    raw_actions: np.ndarray,
    variations: tuple[str, ...],
    action_block: int,
    max_k: int,
    img_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = init_states.shape[0]
    frames = np.empty((n, max_k + 1, img_size, img_size, 3), dtype=np.uint8)
    goal_frames = np.empty((n, 1, img_size, img_size, 3), dtype=np.uint8)
    env = gym.make(
        env_name,
        max_episode_steps=max_k * action_block + 5,
        render_mode="rgb_array",
        resolution=img_size,
    )
    options = reset_options(variations)
    try:
        for i in range(n):
            env.reset(seed=seed + i, options=options)
            set_state_and_goal(env, goal_states[i], goal_states[i])
            goal_frames[i, 0] = env.render()

            set_state_and_goal(env, init_states[i], goal_states[i])
            frames[i, 0] = env.render()
            for t in range(max_k * action_block):
                env.step(raw_actions[i, t])
                if (t + 1) % action_block == 0:
                    k = (t + 1) // action_block
                    frames[i, k] = env.render()
    finally:
        env.close()
    return frames, goal_frames


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
def open_loop_pred_embeddings(
    *,
    model,
    frames: np.ndarray,
    model_actions: np.ndarray,
    max_k: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    chunks = []
    for start in range(0, frames.shape[0], batch_size):
        end = min(start + batch_size, frames.shape[0])
        pixels = images_to_tensor(frames[start:end, :1]).to(device=device, dtype=dtype)
        actions = torch.from_numpy(model_actions[start:end, :max_k]).to(
            device=device, dtype=dtype
        )
        out = model.rollout({"pixels": pixels[:, None]}, actions[:, None])
        pred = out["predicted_emb"][:, 0, : max_k + 1]
        chunks.append(pred.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def latent_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a - b) ** 2, axis=-1)


def make_candidates(
    *,
    rng: np.random.Generator,
    future_model_actions: np.ndarray,
    k: int,
    plan_horizon: int,
    num_candidates: int,
    scale: float,
) -> np.ndarray:
    n, total_model_steps, action_dim = future_model_actions.shape
    cand = rng.normal(
        loc=0.0,
        scale=scale,
        size=(n, num_candidates, plan_horizon, action_dim),
    ).astype(np.float32)
    cand[:, 0] = 0.0
    if k + plan_horizon <= total_model_steps:
        cand[:, 1] = future_model_actions[:, k : k + plan_horizon]
    return cand


def rankdata_2d(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(x.shape[0])[:, None]
    ranks[rows, order] = np.arange(x.shape[1], dtype=np.float32)[None, :]
    return ranks


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = a - a.mean(axis=1, keepdims=True)
    bb = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((aa * aa).sum(axis=1) * (bb * bb).sum(axis=1))
    out = np.full(a.shape[0], np.nan, dtype=np.float32)
    ok = denom > 0
    out[ok] = ((aa[ok] * bb[ok]).sum(axis=1) / denom[ok]).astype(np.float32)
    return out


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pearson_rows(rankdata_2d(a), rankdata_2d(b))


def topk_overlap_rows(a: np.ndarray, b: np.ndarray, k: int) -> np.ndarray:
    k = min(k, a.shape[1], b.shape[1])
    if k <= 0:
        return np.full(a.shape[0], np.nan, dtype=np.float32)
    top_a = np.argpartition(a, kth=k - 1, axis=1)[:, :k]
    top_b = np.argpartition(b, kth=k - 1, axis=1)[:, :k]
    out = np.empty(a.shape[0], dtype=np.float32)
    for i in range(a.shape[0]):
        out[i] = len(set(top_a[i].tolist()) & set(top_b[i].tolist())) / float(k)
    return out


@torch.inference_mode()
def score_candidates_from_emb(
    *,
    model,
    init_emb: np.ndarray,
    goal_emb: np.ndarray,
    candidates: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    n, num_candidates, plan_horizon, action_dim = candidates.shape
    costs = []
    # Rollout only needs the time length from pixels because emb is supplied.
    dummy_pixels = torch.zeros(
        1,
        num_candidates,
        1,
        3,
        224,
        224,
        dtype=dtype,
        device=device,
    )
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        b = end - start
        emb = torch.from_numpy(init_emb[start:end]).to(device=device, dtype=dtype)
        emb = emb[:, None, None, :].expand(b, num_candidates, 1, -1)
        goal = torch.from_numpy(goal_emb[start:end]).to(device=device, dtype=dtype)
        cand = torch.from_numpy(candidates[start:end]).to(device=device, dtype=dtype)
        pixels = dummy_pixels.expand(b, -1, -1, -1, -1, -1)
        action = torch.zeros(
            b, num_candidates, 1, action_dim, dtype=dtype, device=device
        )
        info = {
            "pixels": pixels,
            "goal": torch.zeros(
                b, 1, 3, 224, 224, dtype=dtype, device=device
            ),
            "goal_emb": goal,
            "emb": emb,
            "action": action,
        }
        costs.append(model.get_cost(info, cand).float().cpu().numpy())
    return np.concatenate(costs, axis=0)


def summarize_values(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p25": float("nan"),
            "p75": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
        "n": int(finite.size),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shift",
        "k",
        "env_steps",
        "metric",
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
    parser.add_argument("--policy", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--eval-ks", default="1,3,5,10")
    parser.add_argument("--plan-horizon", type=int, default=5)
    parser.add_argument("--goal-offset", type=int, default=50)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-candidates", type=int, default=256)
    parser.add_argument("--candidate-scale", type=float, default=1.0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lghl_phase4_action_quality")
    parser.add_argument("--shift", action="append", default=None)
    args = parser.parse_args()

    configure_torch_threads_from_env()
    eval_ks = parse_int_list(args.eval_ks)
    if any(k < 0 or k > args.max_k for k in eval_ks):
        raise ValueError("--eval-ks must be within [0, max-k]")
    if args.num_candidates < 3:
        raise ValueError("--num-candidates must be >= 3")

    shifts = (
        [parse_shift(s) for s in args.shift]
        if args.shift
        else list(DEFAULT_SHIFTS)
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[actionq] loading dataset", flush=True)
    dataset = swm.data.load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    total_model_steps = args.max_k + args.plan_horizon
    batch = sample_windows(
        dataset,
        num_samples=args.num_samples,
        total_model_steps=total_model_steps,
        goal_offset=args.goal_offset,
        action_block=args.action_block,
        seed=args.seed,
    )

    print("[actionq] loading model", flush=True)
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
        "eval_ks": eval_ks,
        "plan_horizon": args.plan_horizon,
        "goal_offset": args.goal_offset,
        "action_block": args.action_block,
        "num_candidates": args.num_candidates,
        "candidate_scale": args.candidate_scale,
        "seed": args.seed,
        "device": str(device),
        "shifts": {label: list(vars_) for label, vars_ in shifts},
        "episodes": batch.episodes.tolist(),
        "starts": batch.starts.tolist(),
    }
    (output_dir / "phase4_action_quality_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    rows: list[dict] = []
    summary = {
        "metadata": {
            k: v for k, v in metadata.items() if k not in {"episodes", "starts"}
        },
        "metrics": {},
    }

    rng = np.random.default_rng(args.seed + 17)
    start_all = time.time()
    for label, variations in shifts:
        shift_start = time.time()
        print(f"[actionq] render/encode shift={label}", flush=True)
        frames, goal_frames = render_replay_and_goal(
            env_name=args.env_name,
            init_states=batch.init_states,
            goal_states=batch.goal_states,
            raw_actions=batch.raw_actions,
            variations=variations,
            action_block=args.action_block,
            max_k=args.max_k,
            img_size=args.img_size,
            seed=args.seed,
        )
        true_emb = encode_frames(
            model=model, frames=frames, batch_size=args.batch_size, device=device
        )
        goal_emb = encode_frames(
            model=model, frames=goal_frames, batch_size=args.batch_size, device=device
        )
        pred_emb = open_loop_pred_embeddings(
            model=model,
            frames=frames,
            model_actions=batch.model_actions,
            max_k=args.max_k,
            batch_size=args.batch_size,
            device=device,
        )

        shift_summary = {}
        per_sample = {}
        for k in eval_ks:
            print(f"[actionq] score shift={label} k={k}", flush=True)
            candidates = make_candidates(
                rng=rng,
                future_model_actions=batch.model_actions,
                k=k,
                plan_horizon=args.plan_horizon,
                num_candidates=args.num_candidates,
                scale=args.candidate_scale,
            )
            true_costs = score_candidates_from_emb(
                model=model,
                init_emb=true_emb[:, k],
                goal_emb=goal_emb,
                candidates=candidates,
                batch_size=args.batch_size,
                device=device,
            )
            drift_costs = score_candidates_from_emb(
                model=model,
                init_emb=pred_emb[:, k],
                goal_emb=goal_emb,
                candidates=candidates,
                batch_size=args.batch_size,
                device=device,
            )

            best_true = true_costs.argmin(axis=1)
            best_drift = drift_costs.argmin(axis=1)
            sample_idx = np.arange(args.num_samples)
            true_best_cost = true_costs[sample_idx, best_true]
            drift_choice_true_cost = true_costs[sample_idx, best_drift]
            true_choice_drift_cost = drift_costs[sample_idx, best_true]
            drift_best_cost = drift_costs[sample_idx, best_drift]

            first_block_true = candidates[sample_idx, best_true, 0]
            first_block_drift = candidates[sample_idx, best_drift, 0]
            metrics = {
                "latent_mse": latent_mse(pred_emb[:, k], true_emb[:, k]),
                "top1_same": (best_true == best_drift).astype(np.float32),
                "first_block_l2": np.linalg.norm(
                    first_block_true - first_block_drift, axis=1
                ),
                "true_cost_regret": drift_choice_true_cost - true_best_cost,
                "drift_cost_regret_of_true_best": true_choice_drift_cost
                - drift_best_cost,
                "cost_pearson": pearson_rows(true_costs, drift_costs),
                "cost_spearman": spearman_rows(true_costs, drift_costs),
                "top5_overlap": topk_overlap_rows(true_costs, drift_costs, 5),
                "top10_overlap": topk_overlap_rows(true_costs, drift_costs, 10),
                "expert_rank_true": rankdata_2d(true_costs)[:, 1],
                "expert_rank_drift": rankdata_2d(drift_costs)[:, 1],
            }
            k_summary = {}
            for metric_name, values in metrics.items():
                stats = summarize_values(values)
                k_summary[metric_name] = stats
                rows.append(
                    {
                        "shift": label,
                        "k": k,
                        "env_steps": k * args.action_block,
                        "metric": metric_name,
                        **stats,
                    }
                )
            shift_summary[str(k)] = k_summary
            per_sample[f"k{k}"] = {
                name: values.astype(np.float32) for name, values in metrics.items()
            }

        np.savez_compressed(
            output_dir / f"phase4_action_quality_{label}.npz",
            true_emb=true_emb,
            pred_emb=pred_emb,
            goal_emb=goal_emb,
            **{
                f"{k}_{name}": values
                for k, metric_dict in per_sample.items()
                for name, values in metric_dict.items()
            },
        )
        shift_summary["elapsed_sec"] = time.time() - shift_start
        summary["metrics"][label] = shift_summary

    summary["elapsed_sec"] = time.time() - start_all
    write_csv(output_dir / "phase4_action_quality_summary.csv", rows)
    (output_dir / "phase4_action_quality_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"[actionq] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
