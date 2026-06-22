#!/usr/bin/env python3
"""Phase 5 ID-only behavior diagnostics for LeWM PushT planning.

This script keeps the evaluation condition intentionally simple:

- shift = id only
- dataset-defined starts and goals, matching scripts/plan/eval_wm.py
- richer per-episode metrics than success_rate

The main use is to separate horizon/goal credit assignment from open-loop
latent drift and CEM execution mechanics before adding visual or geometry
shift back into the picture.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


DEFAULT_HORIZONS = "1,2,3,5,8,10"
DEFAULT_RECEDING = "1,2,3,5,8,10"
DEFAULT_GOAL_OFFSETS = "5,10,15,25,50"


@dataclass(frozen=True)
class EvalBatch:
    indices: np.ndarray
    episodes: np.ndarray
    starts: np.ndarray


def configure_torch_threads_from_env() -> None:
    raw = os.environ.get("SWM_TORCH_THREADS")
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        print(f"[phase5] ignoring invalid SWM_TORCH_THREADS={raw!r}", flush=True)
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass
    print(f"[phase5] torch CPU threads set to {threads}", flush=True)


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    out = []
    for item in raw.split(","):
        value = item.strip().lower()
        if not value:
            continue
        if value in {"1", "true", "yes", "warm"}:
            out.append(True)
        elif value in {"0", "false", "no", "cold"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid bool value: {item!r}")
    return out


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def squeeze_time(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def img_transform(img_size: int, dtype: torch.dtype) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_episodes_length(dataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.asarray(lengths)


def sample_eval_batch(
    dataset,
    *,
    num_eval: int,
    goal_offset: int,
    seed: int,
) -> EvalBatch:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.asarray(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    if len(valid_indices) < num_eval:
        raise ValueError(
            f"Need {num_eval} valid starts for goal_offset={goal_offset}, "
            f"found {len(valid_indices)}"
        )

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(valid_indices, size=num_eval, replace=False))
    rows = dataset.get_row_data(chosen)
    return EvalBatch(
        indices=chosen,
        episodes=to_numpy(rows[col_name]).astype(np.int64),
        starts=to_numpy(rows["step_idx"]).astype(np.int64),
    )


def extract_init_goal(dataset, batch: EvalBatch, goal_offset: int) -> tuple[dict, dict]:
    data = dataset.load_chunk(
        batch.episodes,
        batch.starts,
        batch.starts + goal_offset + 1,
    )
    init_lists: dict[str, list[np.ndarray]] = {}
    goal_lists: dict[str, list[np.ndarray]] = {}

    for ep in data:
        for col in dataset.column_names:
            if col.startswith("goal"):
                continue
            value = ep[col]
            if col.startswith("pixels") and torch.is_tensor(value):
                value = value.permute(0, 2, 3, 1)
            if not isinstance(value, (torch.Tensor, np.ndarray)):
                continue
            arr = to_numpy(value)
            init_lists.setdefault(col, []).append(arr[0])
            goal_lists.setdefault(col, []).append(arr[-1])

    init_state = {key: np.stack(values) for key, values in init_lists.items()}
    goal_state = {}
    for key, values in goal_lists.items():
        goal_state["goal" if key == "pixels" else f"goal_{key}"] = np.stack(values)
    return init_state, goal_state


def set_goal_pose_from_state(env, state: np.ndarray) -> None:
    if hasattr(env, "goal_pose") and state is not None and len(state) >= 5:
        env.goal_pose = np.asarray([state[2], state[3], state[4]])


def prepare_world_from_dataset(
    world,
    *,
    init_state: dict,
    goal_state: dict,
) -> dict:
    world.reset(seed=init_state.get("seed"), options=None)

    goal_states = goal_state.get("goal_state")
    init_states = init_state.get("state")
    if init_states is not None:
        for i, env in enumerate(world.envs.envs):
            raw = env.unwrapped
            if goal_states is not None and hasattr(raw, "_set_goal_state"):
                raw._set_goal_state(goal_states[i])
                set_goal_pose_from_state(raw, goal_states[i])
            if hasattr(raw, "_set_state"):
                raw._set_state(init_states[i])

    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for key, value in src.items():
            if key in {"pixels", "goal"}:
                continue
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()

    refresh_rendered_images(world.envs.envs, world.infos, init_state, goal_state)
    return {key: world.infos[key].copy() for key in goal_state if key in world.infos}


def refresh_rendered_images(envs, infos: dict, init_state: dict, goal_state: dict) -> None:
    if "pixels" not in infos or "state" not in init_state:
        return
    init_states = init_state["state"]
    goal_states = goal_state.get("goal_state")
    can_refresh_goal = "goal" in infos and goal_states is not None

    for i, env in enumerate(envs):
        raw = env.unwrapped
        if not hasattr(raw, "_set_state") or not hasattr(raw, "render"):
            continue
        if can_refresh_goal:
            set_goal_pose_from_state(raw, goal_states[i])
            raw._set_state(goal_states[i])
            goal_img = raw.render()
            infos["goal"][i, 0] = goal_img
            if hasattr(raw, "_goal"):
                raw._goal = goal_img
        raw._set_state(init_states[i])
        infos["pixels"][i, 0] = raw.render()


def make_process(dataset, keys: list[str]) -> dict[str, preprocessing.StandardScaler]:
    process = {}
    for key in keys:
        if key not in dataset.column_names or key == "pixels":
            continue
        scaler = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(key)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        scaler.fit(col_data)
        process[key] = scaler
        if key != "action":
            process[f"goal_{key}"] = scaler
    return process


def state_metrics(goal_state: np.ndarray, cur_state: np.ndarray) -> tuple[np.ndarray, ...]:
    full_l2 = np.linalg.norm(goal_state - cur_state, axis=1)
    pos_l2 = np.linalg.norm(goal_state[:, :4] - cur_state[:, :4], axis=1)
    angle_abs = np.abs(wrap_angle(goal_state[:, 4] - cur_state[:, 4]))
    return full_l2, pos_l2, angle_abs


def summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p25": float("nan"), "p75": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p25": float(np.percentile(finite, 25)),
        "p75": float(np.percentile(finite, 75)),
    }


def run_detailed_eval(
    *,
    model,
    dataset,
    batch: EvalBatch,
    process: dict,
    transform: dict,
    env_name: str,
    img_size: int,
    goal_offset: int,
    eval_budget: int,
    horizon: int,
    receding: int,
    action_block: int,
    warm_start: bool,
    solver_args: argparse.Namespace,
    device: str,
) -> tuple[dict, list[dict]]:
    init_state, goal_state = extract_init_goal(dataset, batch, goal_offset)
    config = swm.PlanConfig(
        horizon=horizon,
        receding_horizon=receding,
        action_block=action_block,
        warm_start=warm_start,
    )
    solver = swm.solver.CEMSolver(
        model=model,
        batch_size=solver_args.solver_batch_size,
        num_samples=solver_args.solver_num_samples,
        var_scale=solver_args.solver_var_scale,
        n_steps=solver_args.solver_n_steps,
        topk=solver_args.solver_topk,
        device=device,
        seed=solver_args.seed,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    world = swm.World(
        env_name=env_name,
        num_envs=len(batch.episodes),
        image_shape=(img_size, img_size),
        max_episode_steps=2 * eval_budget,
    )
    world.set_policy(policy)

    try:
        goal_snapshot = prepare_world_from_dataset(
            world, init_state=init_state, goal_state=goal_state
        )
        goal_arr = squeeze_time(goal_snapshot["goal_state"])
        cur_arr = squeeze_time(world.infos["state"])
        init_dist, init_pos, init_angle = state_metrics(goal_arr, cur_arr)

        n = len(batch.episodes)
        success = np.zeros(n, dtype=bool)
        time_to_success = np.full(n, np.nan, dtype=np.float32)
        final_dist = init_dist.copy()
        final_pos = init_pos.copy()
        final_angle = init_angle.copy()
        best_dist = init_dist.copy()
        best_pos = init_pos.copy()
        best_angle = init_angle.copy()
        distance_sum = np.zeros(n, dtype=np.float64)
        reward_sum = np.zeros(n, dtype=np.float64)
        steps = np.zeros(n, dtype=np.int32)
        action_l2_sum = np.zeros(n, dtype=np.float64)
        action_l2_count = np.zeros(n, dtype=np.int32)
        action_delta_sum = np.zeros(n, dtype=np.float64)
        action_delta_count = np.zeros(n, dtype=np.int32)
        prev_action = None
        step_counter = {"value": 0}

        def on_step(run_world) -> None:
            nonlocal prev_action
            step_counter["value"] += 1
            active = ~success

            run_world.infos.update(deepcopy(goal_snapshot))
            cur = squeeze_time(run_world.infos["state"])
            dist, pos, angle = state_metrics(goal_arr, cur)

            final_dist[:] = dist
            final_pos[:] = pos
            final_angle[:] = angle
            best_dist[:] = np.minimum(best_dist, dist)
            best_pos[:] = np.minimum(best_pos, pos)
            best_angle[:] = np.minimum(best_angle, angle)

            distance_sum[active] += dist[active]
            if run_world.rewards is not None:
                reward_sum[active] += np.asarray(run_world.rewards)[active]
            steps[active] += 1

            action = squeeze_time(run_world.infos.get("action"))
            if action is not None:
                action_l2 = np.linalg.norm(action, axis=1)
                action_l2_sum[active] += action_l2[active]
                action_l2_count[active] += 1
                if prev_action is not None:
                    delta = np.linalg.norm(action - prev_action, axis=1)
                    action_delta_sum[active] += delta[active]
                    action_delta_count[active] += 1
                prev_action = action.copy()

            newly_success = active & np.asarray(run_world.terminateds, dtype=bool)
            if newly_success.any():
                success[newly_success] = True
                time_to_success[newly_success] = step_counter["value"]

        world._run(max_steps=eval_budget, mode="wait", on_step=on_step)
    finally:
        world.close()

    with np.errstate(divide="ignore", invalid="ignore"):
        mean_step_distance = distance_sum / np.maximum(steps, 1)
        mean_action_l2 = action_l2_sum / np.maximum(action_l2_count, 1)
        mean_action_delta_l2 = action_delta_sum / np.maximum(action_delta_count, 1)

    episode_rows = []
    for i in range(len(batch.episodes)):
        episode_rows.append(
            {
                "episode_id": i,
                "dataset_index": int(batch.indices[i]),
                "dataset_episode": int(batch.episodes[i]),
                "dataset_start": int(batch.starts[i]),
                "goal_offset": goal_offset,
                "horizon": horizon,
                "receding_horizon": receding,
                "warm_start": str(warm_start).lower(),
                "success": int(success[i]),
                "time_to_success": time_to_success[i],
                "steps_executed": int(steps[i]),
                "initial_distance": init_dist[i],
                "final_distance": final_dist[i],
                "best_distance": best_dist[i],
                "final_pos_l2": final_pos[i],
                "best_pos_l2": best_pos[i],
                "final_angle_abs": final_angle[i],
                "best_angle_abs": best_angle[i],
                "distance_auc": distance_sum[i],
                "mean_step_distance": mean_step_distance[i],
                "episode_return": reward_sum[i],
                "mean_action_l2": mean_action_l2[i],
                "mean_action_delta_l2": mean_action_delta_l2[i],
            }
        )

    final_stats = summarize(final_dist)
    best_stats = summarize(best_dist)
    tts_stats = summarize(time_to_success)
    summary = {
        "shift": "id",
        "goal_offset": goal_offset,
        "horizon": horizon,
        "receding_horizon": receding,
        "action_block": action_block,
        "plan_horizon_env_steps": horizon * action_block,
        "verify_gap_env_steps": receding * action_block,
        "warm_start": str(warm_start).lower(),
        "num_eval": len(batch.episodes),
        "success_rate": float(success.mean() * 100.0),
        "mean_final_distance": final_stats["mean"],
        "median_final_distance": final_stats["median"],
        "p25_final_distance": final_stats["p25"],
        "p75_final_distance": final_stats["p75"],
        "mean_best_distance": best_stats["mean"],
        "median_best_distance": best_stats["median"],
        "p25_best_distance": best_stats["p25"],
        "p75_best_distance": best_stats["p75"],
        "mean_time_to_success": tts_stats["mean"],
        "median_time_to_success": tts_stats["median"],
        "mean_episode_return": float(np.mean(reward_sum)),
        "mean_distance_auc": float(np.mean(distance_sum)),
        "mean_step_distance": float(np.mean(mean_step_distance)),
        "mean_action_l2": float(np.mean(mean_action_l2)),
        "mean_action_delta_l2": float(np.mean(mean_action_delta_l2)),
        "mean_steps_executed": float(np.mean(steps)),
    }
    return summary, episode_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def receding_for_horizon(
    *,
    horizon: int,
    mode: str,
    receding_values: list[int],
) -> list[int]:
    if mode == "diagonal":
        return [horizon]
    values = [r for r in receding_values if r <= horizon]
    if mode in {"all", "list"}:
        return values
    raise ValueError(f"unknown receding mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env-name", default="swm/PushT-v1")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offsets", default=DEFAULT_GOAL_OFFSETS)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizons", default=DEFAULT_HORIZONS)
    parser.add_argument("--receding", default=DEFAULT_RECEDING)
    parser.add_argument(
        "--receding-mode",
        choices=["diagonal", "all", "list"],
        default="diagonal",
    )
    parser.add_argument("--warm-start", default="true")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--solver-batch-size", type=int, default=1)
    parser.add_argument("--solver-num-samples", type=int, default=300)
    parser.add_argument("--solver-var-scale", type=float, default=1.0)
    parser.add_argument("--solver-n-steps", type=int, default=30)
    parser.add_argument("--solver-topk", type=int, default=30)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--output-dir", default="outputs/lghl_phase5_id_behavior")
    args = parser.parse_args()

    configure_torch_threads_from_env()
    horizons = parse_int_list(args.horizons)
    receding_values = parse_int_list(args.receding)
    goal_offsets = parse_int_list(args.goal_offsets)
    warm_values = parse_bool_list(args.warm_start)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "env_name": args.env_name,
        "num_eval": args.num_eval,
        "goal_offsets": goal_offsets,
        "eval_budget": args.eval_budget,
        "horizons": horizons,
        "receding": receding_values,
        "receding_mode": args.receding_mode,
        "warm_start": warm_values,
        "action_block": args.action_block,
        "seed": args.seed,
        "device": args.device,
        "solver": {
            "batch_size": args.solver_batch_size,
            "num_samples": args.solver_num_samples,
            "var_scale": args.solver_var_scale,
            "n_steps": args.solver_n_steps,
            "topk": args.solver_topk,
        },
    }
    (output_dir / "phase5_id_behavior_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("[phase5] loading dataset", flush=True)
    dataset = swm.data.load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        keys_to_cache=["action", "proprio", "state"],
    )
    process = make_process(dataset, ["action", "proprio", "state"])

    print("[phase5] loading model", flush=True)
    model = swm.wm.utils.load_pretrained(args.policy)
    if args.bf16:
        model = model.to(torch.bfloat16)
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    img_dtype = torch.bfloat16 if args.bf16 else torch.float32
    transform = {
        "pixels": img_transform(args.img_size, img_dtype),
        "goal": img_transform(args.img_size, img_dtype),
    }

    summary_rows: list[dict] = []
    episode_rows: list[dict] = []
    batch_cache: dict[int, EvalBatch] = {}
    start_all = time.time()

    for goal_offset in goal_offsets:
        batch_cache[goal_offset] = sample_eval_batch(
            dataset,
            num_eval=args.num_eval,
            goal_offset=goal_offset,
            seed=args.seed,
        )
        for horizon in horizons:
            if horizon * args.action_block > args.eval_budget:
                print(
                    f"[phase5] skip H={horizon}: plan exceeds eval_budget",
                    flush=True,
                )
                continue
            for receding in receding_for_horizon(
                horizon=horizon,
                mode=args.receding_mode,
                receding_values=receding_values,
            ):
                for warm in warm_values:
                    label = (
                        f"goal={goal_offset} H={horizon} R={receding} "
                        f"warm={warm}"
                    )
                    print(f"[phase5] running {label}", flush=True)
                    start = time.time()
                    summary, episodes = run_detailed_eval(
                        model=model,
                        dataset=dataset,
                        batch=batch_cache[goal_offset],
                        process=process,
                        transform=transform,
                        env_name=args.env_name,
                        img_size=args.img_size,
                        goal_offset=goal_offset,
                        eval_budget=args.eval_budget,
                        horizon=horizon,
                        receding=receding,
                        action_block=args.action_block,
                        warm_start=warm,
                        solver_args=args,
                        device=args.device,
                    )
                    elapsed = time.time() - start
                    summary["elapsed_sec"] = elapsed
                    summary_rows.append(summary)
                    episode_rows.extend(episodes)
                    write_csv(output_dir / "phase5_id_behavior_summary.csv", summary_rows)
                    write_csv(output_dir / "phase5_id_behavior_episodes.csv", episode_rows)
                    print(
                        f"[phase5] done {label}: "
                        f"success={summary['success_rate']:.1f} "
                        f"best={summary['mean_best_distance']:.3f} "
                        f"final={summary['mean_final_distance']:.3f} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

    metadata["elapsed_sec"] = time.time() - start_all
    (output_dir / "phase5_id_behavior_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
