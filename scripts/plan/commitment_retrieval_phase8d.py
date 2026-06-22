"""Phase 8d: retrieval-based (non-parametric) commitment sub-goal proposer.

Phase 8c killed the hand-crafted geometric sub-goal: it hurts. The oracle (peek
at the expert mid latent) helps a lot but is circular. This script tests the
middle ground -- a realizable, non-circular proposer that NEVER sees this
sample's future: a retrieval database built from OTHER expert windows mapping

    (z_state, z_goal)  ->  z_{state + delta}

At eval, for the planner's drifted latent z_k and goal, retrieve the nearest
(z_state, z_goal) in the DB and use its forward latent as the commitment
sub-goal. If this recovers a meaningful fraction of the oracle credit benefit,
a learned proposer is worth training; if it recovers ~0, the waypoint is hard to
predict from (state, goal) alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_action_quality_phase4 as aq  # noqa: E402
import latent_drift_phase3 as p3  # noqa: E402
import commitment_subgoal_phase8b as p8b  # noqa: E402


def build_db(args, model, device):
    dataset = aq.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = p3.build_window_batch(
        dataset, num_samples=args.db_size, max_k=2,
        goal_offset=args.goal_offset, action_block=args.action_block,
        seed=args.db_seed,
    )
    # z at model steps 0,1,2 (id render, goal-T present)
    frames = p3.render_state_sequence(
        env_name=args.env_name, states_by_k=batch.states_by_k,
        goal_states=batch.goal_states, variations=(),
        img_size=args.img_size, seed=args.db_seed,
    )
    z012 = p3.encode_frames(model=model, frames=frames,
                            batch_size=args.batch_size, device=device)  # (N,3,D)
    # goal latent
    goal_frames = p3.render_state_sequence(
        env_name=args.env_name, states_by_k=batch.goal_states[:, None],
        goal_states=batch.goal_states, variations=(),
        img_size=args.img_size, seed=args.db_seed,
    )
    z_goal = p3.encode_frames(model=model, frames=goal_frames,
                              batch_size=args.batch_size, device=device)[:, 0]
    key = np.concatenate([z012[:, 0], z_goal], axis=1).astype(np.float32)  # (N,2D)
    return key, {1: z012[:, 1].astype(np.float32), 2: z012[:, 2].astype(np.float32)}


def retrieve(query_key, db_key, db_vals, delta, m, block=1024):
    k_sq = (db_key * db_key).sum(1)
    out = np.empty((query_key.shape[0], db_vals[delta].shape[1]), dtype=np.float32)
    for s in range(0, query_key.shape[0], block):
        q = query_key[s : s + block]
        d2 = (q * q).sum(1)[:, None] - 2 * q @ db_key.T + k_sq[None, :]
        idx = np.argpartition(d2, kth=m - 1, axis=1)[:, :m]
        out[s : s + block] = db_vals[delta][idx].mean(axis=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8d_retrieval")
    ap.add_argument("--num-samples", type=int, default=120)
    ap.add_argument("--db-size", type=int, default=3000)
    ap.add_argument("--db-seed", type=int, default=123)
    ap.add_argument("--eval-ks", default="0,5,8")
    ap.add_argument("--plan-horizons", default="3,5")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--knn-m", type=int, default=4)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--num-candidates", type=int, default=256)
    ap.add_argument("--candidate-scale", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--shift", default="id")
    args = ap.parse_args()

    aq.configure_torch_threads_from_env()
    eval_ks = aq.parse_int_list(args.eval_ks)
    plan_hs = aq.parse_int_list(args.plan_horizons)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variations = dict(aq.DEFAULT_SHIFTS)[args.shift]
    t0 = time.time()
    K_total = max(eval_ks) + max(plan_hs)

    device = torch.device(args.device)
    dataset = aq.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = aq.sample_windows(
        dataset, num_samples=args.num_samples, total_model_steps=K_total,
        goal_offset=args.goal_offset, action_block=args.action_block, seed=args.seed,
    )
    model = aq.swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print("[8d] building retrieval DB", flush=True)
    db_key, db_vals = build_db(args, model, device)

    print(f"[8d] render/encode eval to K_total={K_total}", flush=True)
    frames, goal_frames = aq.render_replay_and_goal(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        variations=variations, action_block=args.action_block,
        max_k=K_total, img_size=args.img_size, seed=args.seed,
    )
    true_emb = aq.encode_frames(model=model, frames=frames, batch_size=args.batch_size, device=device)
    goal_emb = aq.encode_frames(model=model, frames=goal_frames, batch_size=args.batch_size, device=device)[:, 0]
    pred_emb = aq.open_loop_pred_embeddings(
        model=model, frames=frames, model_actions=batch.model_actions,
        max_k=K_total, batch_size=args.batch_size, device=device)

    rng = np.random.default_rng(args.seed + 17)
    sidx = np.arange(args.num_samples)
    rows = []
    for k in eval_ks:
        for H in plan_hs:
            mid = max(1, H // 2)
            candidates = aq.make_candidates(
                rng=rng, future_model_actions=batch.model_actions, k=k,
                plan_horizon=H, num_candidates=args.num_candidates,
                scale=args.candidate_scale)
            traj_true = p8b.rollout_pred(model, true_emb[:, k], candidates, device, args.batch_size)
            traj_drift = p8b.rollout_pred(model, pred_emb[:, k], candidates, device, args.batch_size)
            ref_choice = p8b.costs_from_traj(traj_true, goal_emb, None, mid, 0.0).argmin(1)
            true_term = p8b.costs_from_traj(traj_true, goal_emb, None, mid, 0.0)

            query_key = np.concatenate([pred_emb[:, k], goal_emb], axis=1).astype(np.float32)
            sg_retr = retrieve(query_key, db_key, db_vals, mid, args.knn_m)
            sgs = {"oracle": true_emb[:, k + mid], "retrieval": sg_retr}

            def record(name, lam, sg):
                c = p8b.costs_from_traj(traj_drift, goal_emb, sg, mid, lam)
                ranks = aq.rankdata_2d(c)
                choice = c.argmin(1)
                rows.append({
                    "k": k, "plan_h": H, "subgoal": name, "lam": lam,
                    "expert_rank": float(ranks[:, 1].mean()),
                    "top1_vs_ideal": float(np.mean(choice == ref_choice)),
                    "regret": float(np.mean(true_term[sidx, choice] - true_term[sidx, ref_choice])),
                    "subgoal_mse_to_oracle": float(np.mean((sg - true_emb[:, k + mid]) ** 2)) if sg is not None else 0.0,
                })

            record("none", 0.0, None)
            for name, sg in sgs.items():
                record(name, args.lam, sg)
            print(f"[8d] k={k} H={H} done", flush=True)

    with (out_dir / "phase8d_retrieval_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "plan_h", "subgoal", "lam",
                            "expert_rank", "top1_vs_ideal", "regret", "subgoal_mse_to_oracle"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "phase8d_retrieval_summary.json").write_text(
        json.dumps({"meta": vars(args), "rows": rows, "elapsed_sec": time.time() - t0}, indent=2))

    def g(k, H, n, f):
        m = [r for r in rows if r["k"] == k and r["plan_h"] == H and r["subgoal"] == n]
        return m[0][f] if m else float("nan")
    print(f"\n[8d] expert_rank (/{args.num_candidates}, lower=better), drift init, lam={args.lam}:")
    print("  %-7s %-10s %-10s %-12s %-s" % ("k/H", "baseline", "oracle", "retrieval", "retr_recover%"))
    for k in eval_ks:
        for H in plan_hs:
            b = g(k, H, "none", "expert_rank"); o = g(k, H, "oracle", "expert_rank"); r = g(k, H, "retrieval", "expert_rank")
            rec = 100 * (b - r) / (b - o) if (b - o) != 0 else float("nan")
            print("  %-7s %-10.1f %-10.1f %-12.1f %.0f%%" % ("k%dH%d" % (k, H), b, o, r, rec))
    print(f"[8d] wrote {out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
