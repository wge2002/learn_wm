"""Phase 8c: de-circularized commitment sub-goal.

Phase 8b's oracle sub-goal was the expert's own mid-trajectory latent, which is
circular (the expert action is the one that reaches it). This script replaces it
with sub-goals that NEVER peek at the expert future, built only from the latent
the planner actually has (drifted) + the goal + the ID manifold bank:

  - interp_goal(alpha): manifold-projected interpolation toward the goal,
        subgoal = proj_to_bank( (1-alpha)*z_cur + alpha*goal_emb )
    a non-circular "head this far toward the goal, on-manifold" waypoint.
  - interp_raw(alpha): same without manifold projection (ablation).
  - oracle: kept as the circular ceiling for reference.

If expert_rank still improves with interp_goal (the expert makes real progress,
so a toward-goal waypoint should rank it well non-circularly), the credit benefit
of a commitment anchor is real, not an artifact of peeking at the expert path.
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
import commitment_subgoal_phase8b as p8b  # noqa: E402


def knn_project(x, bank, m, block=2048):
    bank_sq = (bank * bank).sum(1)
    out = np.empty_like(x)
    for s in range(0, x.shape[0], block):
        q = x[s : s + block]
        d2 = (q * q).sum(1)[:, None] - 2 * q @ bank.T + bank_sq[None, :]
        idx = np.argpartition(d2, kth=m - 1, axis=1)[:, :m]
        out[s : s + block] = bank[idx].mean(axis=1)
    return out.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--bank-path", default="outputs/lghl_phase7_manifold/id_latent_bank.npy")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8c_decircular")
    ap.add_argument("--num-samples", type=int, default=120)
    ap.add_argument("--eval-ks", default="0,5,8")
    ap.add_argument("--plan-horizons", default="2,3,5")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--alphas", default="0.3,0.5")
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--num-candidates", type=int, default=256)
    ap.add_argument("--candidate-scale", type=float, default=1.0)
    ap.add_argument("--knn-m", type=int, default=8)
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
    alphas = [float(x) for x in args.alphas.split(",") if x]
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

    print(f"[8c] render/encode to K_total={K_total}", flush=True)
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
    bank = np.load(args.bank_path).astype(np.float32)

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

            z_cur = pred_emb[:, k]  # drifted latent the planner actually has
            subgoals = {"oracle": true_emb[:, k + mid]}  # circular ceiling
            for a in alphas:
                lin = ((1 - a) * z_cur + a * goal_emb).astype(np.float32)
                subgoals[f"interp_goal_a{a}"] = knn_project(lin, bank, args.knn_m)
                subgoals[f"interp_raw_a{a}"] = lin

            def record(name, lam, sg):
                c = p8b.costs_from_traj(traj_drift, goal_emb, sg, mid, lam)
                ranks = aq.rankdata_2d(c)
                choice = c.argmin(1)
                rows.append({
                    "k": k, "plan_h": H, "subgoal": name, "lam": lam,
                    "expert_rank": float(ranks[:, 1].mean()),
                    "top1_vs_ideal": float(np.mean(choice == ref_choice)),
                    "regret": float(np.mean(true_term[sidx, choice] - true_term[sidx, ref_choice])),
                })

            record("none", 0.0, None)  # baseline
            for name, sg in subgoals.items():
                record(name, args.lam, sg)
            print(f"[8c] k={k} H={H} done", flush=True)

    with (out_dir / "phase8c_decircular_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["k", "plan_h", "subgoal", "lam",
                            "expert_rank", "top1_vs_ideal", "regret"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "phase8c_decircular_summary.json").write_text(
        json.dumps({"meta": vars(args), "rows": rows, "elapsed_sec": time.time() - t0}, indent=2))

    names = ["none", "oracle"] + [f"interp_goal_a{a}" for a in alphas] + [f"interp_raw_a{a}" for a in alphas]
    print(f"\n[8c] expert_rank (/{args.num_candidates}, lower=better), drift init, lam={args.lam}:")
    print("  " + "k/H".ljust(7) + "".join("%-16s" % n for n in names))
    for k in eval_ks:
        for H in plan_hs:
            cells = []
            for n in names:
                m = [r for r in rows if r["k"] == k and r["plan_h"] == H and r["subgoal"] == n]
                cells.append("%-16.1f" % m[0]["expert_rank"] if m else "%-16s" % "-")
            print("  " + ("k%dH%d" % (k, H)).ljust(7) + "".join(cells))
    print(f"[8c] wrote {out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
