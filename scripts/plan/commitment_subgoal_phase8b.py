"""Phase 8b: commitment anchor as an intermediate planning sub-goal (oracle).

Phase 8 ruled out using a discrete anchor to *replace* the rollout latent
(precision-bound). The reframed commitment anchor instead adds a low-dim
intermediate sub-goal to the planner's terminal-only cost, keeping the precise
continuous latent. This script measures the oracle ceiling of that idea on the
action-quality proxy.

For a planning state at step k with horizon H, candidates are scored with

    cost = (1-lam) * MSE(pred_emb[:,H], goal_emb)            # terminal (baseline)
         +    lam  * MSE(pred_emb[:,mid], subgoal_emb)       # commitment term

where subgoal_emb is the TRUE latent the expert reaches at the mid step (oracle),
optionally quantized to a codebook. lam=0 reproduces the terminal-only baseline.

Candidate index 1 is seeded with the expert future action chunk, so a good cost
ranks it near the top. The headline question: under the DRIFTED open-loop latent
(realistic planning condition), does adding the commitment term (lam>0) lower the
expert action's rank and raise agreement with the ideal (true-latent) planner,
especially at short H where terminal-only credit is starved (Phase 5)?
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


@torch.inference_mode()
def rollout_pred(model, init_emb, candidates, device, batch_size):
    """Full predicted-emb trajectory for candidates from a given init latent.

    init_emb: (n, D); candidates: (n, S, H, A). Returns (n, S, H+1, D).
    """
    dtype = next(model.parameters()).dtype
    n, S, H, A = candidates.shape
    out = np.empty((n, S, H + 1, init_emb.shape[-1]), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        b = e - s
        emb = torch.from_numpy(init_emb[s:e]).to(device=device, dtype=dtype)
        emb = emb[:, None, None, :].expand(b, S, 1, -1)
        dummy_pixels = torch.zeros(b, S, 1, 3, 224, 224, dtype=dtype, device=device)
        cand = torch.from_numpy(candidates[s:e]).to(device=device, dtype=dtype)
        info = model.rollout({"pixels": dummy_pixels, "emb": emb}, cand)
        out[s:e] = info["predicted_emb"].float().cpu().numpy()
    return out


def costs_from_traj(traj, goal_emb, subgoal_emb, mid, lam):
    """traj: (n,S,H+1,D); goal_emb,subgoal_emb: (n,D). Returns (n,S)."""
    term = np.mean((traj[:, :, -1] - goal_emb[:, None]) ** 2, axis=-1)
    if lam <= 0:
        return term
    sub = np.mean((traj[:, :, mid] - subgoal_emb[:, None]) ** 2, axis=-1)
    return (1 - lam) * term + lam * sub


def make_quantizer(centroids):
    c_sq = (centroids * centroids).sum(1)

    def q(x):
        d2 = (x * x).sum(1)[:, None] - 2 * x @ centroids.T + c_sq[None, :]
        return centroids[d2.argmin(1)]
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--bank-path", default="outputs/lghl_phase7_manifold/id_latent_bank.npy")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8b_commitment")
    ap.add_argument("--num-samples", type=int, default=120)
    ap.add_argument("--eval-ks", default="0,5,8")
    ap.add_argument("--plan-horizons", default="2,3,5")
    ap.add_argument("--lams", default="0,0.25,0.5,0.75")
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--num-candidates", type=int, default=256)
    ap.add_argument("--candidate-scale", type=float, default=1.0)
    ap.add_argument("--subgoal-codebook", type=int, default=2048)
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
    lams = [float(x) for x in args.lams.split(",") if x]
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

    print(f"[8b] render/encode true_emb to K_total={K_total}", flush=True)
    frames, goal_frames = aq.render_replay_and_goal(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        variations=variations, action_block=args.action_block,
        max_k=K_total, img_size=args.img_size, seed=args.seed,
    )
    true_emb = aq.encode_frames(model=model, frames=frames,
                                batch_size=args.batch_size, device=device)
    goal_emb = aq.encode_frames(model=model, frames=goal_frames,
                                batch_size=args.batch_size, device=device)[:, 0]
    pred_emb = aq.open_loop_pred_embeddings(
        model=model, frames=frames, model_actions=batch.model_actions,
        max_k=K_total, batch_size=args.batch_size, device=device)

    quant = None
    if args.subgoal_codebook > 0:
        from sklearn.cluster import MiniBatchKMeans
        bank = np.load(args.bank_path).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=args.subgoal_codebook, batch_size=4096,
                             n_init=3, max_iter=100, random_state=0).fit(bank)
        quant = make_quantizer(km.cluster_centers_.astype(np.float32))

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
            subgoal_oracle = true_emb[:, k + mid]
            subgoals = {"oracle": subgoal_oracle}
            if quant is not None:
                subgoals[f"disc{args.subgoal_codebook}"] = quant(subgoal_oracle).astype(np.float32)

            traj_true = rollout_pred(model, true_emb[:, k], candidates, device, args.batch_size)
            traj_drift = rollout_pred(model, pred_emb[:, k], candidates, device, args.batch_size)

            # ideal planner reference = true latent, terminal-only
            ref_choice = costs_from_traj(traj_true, goal_emb, None, mid, 0.0).argmin(1)

            for sg_name, sg in subgoals.items():
                for lam in lams:
                    for init_name, traj in (("true", traj_true), ("drift", traj_drift)):
                        if lam == 0 and sg_name != list(subgoals)[0]:
                            continue  # lam=0 identical across subgoal variants
                        c = costs_from_traj(traj, goal_emb, sg, mid, lam)
                        ranks = aq.rankdata_2d(c)  # (n,S) ascending: 0=best
                        expert_rank = float(ranks[:, 1].mean())  # candidate 1 = expert
                        choice = c.argmin(1)
                        top1_vs_ideal = float(np.mean(choice == ref_choice))
                        # regret in TRUE terminal cost of the chosen action
                        true_term = costs_from_traj(traj_true, goal_emb, None, mid, 0.0)
                        regret = float(np.mean(true_term[sidx, choice]
                                               - true_term[sidx, ref_choice]))
                        rows.append({
                            "shift": args.shift, "k": k, "plan_h": H, "mid": mid,
                            "subgoal": sg_name if lam > 0 else "none",
                            "lam": lam, "init": init_name,
                            "expert_rank": expert_rank,
                            "top1_vs_ideal": top1_vs_ideal, "regret": regret,
                        })
            print(f"[8b] k={k} H={H} done", flush=True)

    with (out_dir / "phase8b_commitment_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shift", "k", "plan_h", "mid", "subgoal",
                            "lam", "init", "expert_rank", "top1_vs_ideal", "regret"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "phase8b_commitment_summary.json").write_text(
        json.dumps({"meta": vars(args), "rows": rows,
                    "elapsed_sec": time.time() - t0}, indent=2))

    # headline: drifted-init, oracle subgoal, expert_rank vs lam (lower=better)
    print("\n[8b] DRIFTED-init expert_rank (lower=planner prefers good action), oracle subgoal:")
    print("      lam=0 is terminal-only baseline. num_candidates=%d" % args.num_candidates)
    for k in eval_ks:
        for H in plan_hs:
            cells = []
            for lam in lams:
                sgn = "oracle" if lam > 0 else "none"
                m = [r for r in rows if r["k"] == k and r["plan_h"] == H
                     and r["init"] == "drift" and r["lam"] == lam
                     and r["subgoal"] == sgn]
                cells.append(f"lam{lam}={m[0]['expert_rank']:.1f}" if m else "-")
            print(f"  k={k} H={H}: " + "  ".join(cells))
    print(f"[8b] wrote {out_dir} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
