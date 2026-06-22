"""Phase 8h: downstream action-quality eval of the trained anchor proposer.

Reuses the Phase 8d eval but adds a "trained" sub-goal produced by the Phase 8g
proposer from the grounded latent + goal. Headline: expert_rank recovery vs the
oracle ceiling at k=0 (the realistic grounded-replanning regime), compared to
the retrieval baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_action_quality_phase4 as aq  # noqa: E402
import latent_drift_phase3 as p3  # noqa: E402
import commitment_subgoal_phase8b as p8b  # noqa: E402
import commitment_retrieval_phase8d as p8d  # noqa: E402
from anchor_train_phase8g import Proposer  # noqa: E402


def load_proposer(path, device):
    ck = torch.load(path, map_location=device)
    m = Proposer(dim=ck["dim"], codebook=ck.get("codebook", 256),
                 discrete=ck["discrete"]).to(device).eval()
    m.load_state_dict(ck["state"])
    return m


@torch.no_grad()
def proposer_waypoint(model, z, g, device):
    zt = torch.from_numpy(z).to(device); gt = torch.from_numpy(g).to(device)
    w, _ = model(zt, gt, hard=True)
    return w.cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--proposer-dir", default="outputs/lghl_phase8g_proposer")
    ap.add_argument("--proposer-tag", default="discrete_C256")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8h_eval")
    ap.add_argument("--num-samples", type=int, default=120)
    ap.add_argument("--db-size", type=int, default=3000)
    ap.add_argument("--db-seed", type=int, default=123)
    ap.add_argument("--eval-ks", default="0")
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
    args = ap.parse_args()

    aq.configure_torch_threads_from_env()
    eval_ks = aq.parse_int_list(args.eval_ks)
    plan_hs = aq.parse_int_list(args.plan_horizons)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    variations = ()
    K_total = max(eval_ks) + max(plan_hs)
    device = torch.device(args.device)

    dataset = aq.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    batch = aq.sample_windows(
        dataset, num_samples=args.num_samples, total_model_steps=K_total,
        goal_offset=args.goal_offset, action_block=args.action_block, seed=args.seed)
    model = aq.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True

    db_key, db_vals = p8d.build_db(args, model, device)
    frames, goal_frames = aq.render_replay_and_goal(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        variations=variations, action_block=args.action_block,
        max_k=K_total, img_size=args.img_size, seed=args.seed)
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
            prop = load_proposer(Path(args.proposer_dir) / f"proposer_{args.proposer_tag}_d{mid}.pt", device)
            candidates = aq.make_candidates(
                rng=rng, future_model_actions=batch.model_actions, k=k,
                plan_horizon=H, num_candidates=args.num_candidates, scale=args.candidate_scale)
            traj_true = p8b.rollout_pred(model, true_emb[:, k], candidates, device, args.batch_size)
            traj_drift = p8b.rollout_pred(model, pred_emb[:, k], candidates, device, args.batch_size)
            ref_choice = p8b.costs_from_traj(traj_true, goal_emb, None, mid, 0.0).argmin(1)
            true_term = p8b.costs_from_traj(traj_true, goal_emb, None, mid, 0.0)

            z_in = pred_emb[:, k]  # at k=0 this is the grounded latent
            query_key = np.concatenate([z_in, goal_emb], axis=1).astype(np.float32)
            sgs = {
                "oracle": true_emb[:, k + mid],
                "retrieval": p8d.retrieve(query_key, db_key, db_vals, mid, args.knn_m),
                "trained": proposer_waypoint(prop, z_in, goal_emb, device),
            }

            def record(name, lam, sg):
                c = p8b.costs_from_traj(traj_drift, goal_emb, sg, mid, lam)
                ranks = aq.rankdata_2d(c); choice = c.argmin(1)
                rows.append({"k": k, "plan_h": H, "subgoal": name, "lam": lam,
                    "expert_rank": float(ranks[:, 1].mean()),
                    "top1_vs_ideal": float(np.mean(choice == ref_choice)),
                    "waypoint_mse_to_oracle": float(np.mean((sg - true_emb[:, k + mid]) ** 2)) if sg is not None else 0.0})
            record("none", 0.0, None)
            for name, sg in sgs.items():
                record(name, args.lam, sg)
            print(f"[8h] k={k} H={H} done", flush=True)

    (out_dir / "phase8h_eval_summary.json").write_text(json.dumps({"meta": vars(args), "rows": rows}, indent=2))

    def g(k, H, n, f):
        m = [r for r in rows if r["k"] == k and r["plan_h"] == H and r["subgoal"] == n]
        return m[0][f] if m else float("nan")
    print(f"\n[8h] expert_rank (/{args.num_candidates}, lower=better), proposer={args.proposer_tag}:")
    print("  %-7s %-9s %-9s %-11s %-10s | %s" % ("k/H", "baseline", "oracle", "retrieval", "trained", "trained_recover%"))
    for k in eval_ks:
        for H in plan_hs:
            b = g(k, H, "none", "expert_rank"); o = g(k, H, "oracle", "expert_rank")
            r = g(k, H, "retrieval", "expert_rank"); t = g(k, H, "trained", "expert_rank")
            rec = 100 * (b - t) / (b - o) if (b - o) != 0 else float("nan")
            print("  %-7s %-9.1f %-9.1f %-11.1f %-10.1f | %.0f%%" % ("k%dH%d" % (k, H), b, o, r, t, rec))
    print(f"[8h] wrote {out_dir}")


if __name__ == "__main__":
    main()
