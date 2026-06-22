"""Phase 7 (C): does projecting the drifted latent back onto the data manifold
restore the planner's candidate ranking?

This is the decision-level test of the "restoring force" idea. It reuses the
Phase 4 action-quality machinery: for each window/shift/k it scores a shared
candidate set under three latents and compares each to the TRUE-latent ranking:

  - true   : score from the re-grounded true latent z_k        (reference)
  - drift  : score from the open-loop predicted latent zhat_k  (baseline degr.)
  - proj_* : score from zhat_k projected back toward the ID latent manifold
             (PCA-denoise / kNN soft-blend), using only a generic ID latent
             bank (never the specific ground-truth future).

If a projection moves top1/top5/Spearman agreement (vs true) back up from the
drift baseline toward 1.0, the manifold-projection restoring force helps the
planner, not just the raw latent MSE.
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
import latent_manifold_phase7 as mani  # noqa: E402


def build_projectors(bank: np.ndarray, pca_ds, alphas, knn_m):
    mean, evecs, evals = mani.pca_fit(bank)
    projs = {}
    for d in pca_ds:
        projs[f"pca{d}"] = ("pca", d)
    for a in alphas:
        projs[f"knnsoft{a}"] = ("knn", a)
    projs["knnsnap"] = ("knn", 1.0)
    return mean, evecs, bank, projs


def apply_projector(zh, kind, param, mean, evecs, bank, knn_m, nn_cache):
    if kind == "pca":
        return mani.pca_denoise(zh, mean, evecs, int(param))
    # knn soft blend toward kNN-mean (param=alpha); alpha=1.0 with m=1 == snap
    if nn_cache[0] is None:
        nn_cache[0] = mani.knn_indices(zh, bank, knn_m)
    nn = nn_cache[0]
    if param == 1.0 and knn_m >= 1:
        knn_mean = bank[nn].mean(axis=1)
    else:
        knn_mean = bank[nn].mean(axis=1)
    return (1 - param) * zh + param * knn_mean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--bank-path", required=True)
    ap.add_argument("--output-dir", default="outputs/lghl_phase7_action_proj")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--eval-ks", default="1,5,10")
    ap.add_argument("--plan-horizon", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--num-candidates", type=int, default=256)
    ap.add_argument("--candidate-scale", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--pca-ds", default="20,50")
    ap.add_argument("--alphas", default="0.3,0.6")
    ap.add_argument("--knn-m", type=int, default=8)
    ap.add_argument("--shift", action="append", default=None)
    args = ap.parse_args()

    aq.configure_torch_threads_from_env()
    eval_ks = aq.parse_int_list(args.eval_ks)
    shifts = (
        [aq.parse_shift(s) for s in args.shift] if args.shift
        else list(aq.DEFAULT_SHIFTS)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pca_ds = [int(x) for x in args.pca_ds.split(",") if x]
    alphas = [float(x) for x in args.alphas.split(",") if x]

    bank = np.load(args.bank_path).astype(np.float32)
    mean, evecs, bank, projs = build_projectors(bank, pca_ds, alphas, args.knn_m)
    print(f"[phase7C] bank {bank.shape}, projectors {list(projs)}", flush=True)

    device = torch.device(args.device)
    dataset = aq.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    total_model_steps = args.max_k + args.plan_horizon
    batch = aq.sample_windows(
        dataset, num_samples=args.num_samples,
        total_model_steps=total_model_steps, goal_offset=args.goal_offset,
        action_block=args.action_block, seed=args.seed,
    )
    model = aq.swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    rng = np.random.default_rng(args.seed + 17)
    rows: list[dict] = []
    summary = {"meta": {
        "bank_path": args.bank_path, "num_samples": args.num_samples,
        "eval_ks": eval_ks, "plan_horizon": args.plan_horizon,
        "num_candidates": args.num_candidates, "projectors": list(projs),
        "seed": args.seed,
    }, "results": {}}
    t0 = time.time()

    for label, variations in shifts:
        print(f"[phase7C] render/encode shift={label}", flush=True)
        frames, goal_frames = aq.render_replay_and_goal(
            env_name=args.env_name, init_states=batch.init_states,
            goal_states=batch.goal_states, raw_actions=batch.raw_actions,
            variations=variations, action_block=args.action_block,
            max_k=args.max_k, img_size=args.img_size, seed=args.seed,
        )
        true_emb = aq.encode_frames(model=model, frames=frames,
                                    batch_size=args.batch_size, device=device)
        goal_emb = aq.encode_frames(model=model, frames=goal_frames,
                                    batch_size=args.batch_size, device=device)
        pred_emb = aq.open_loop_pred_embeddings(
            model=model, frames=frames, model_actions=batch.model_actions,
            max_k=args.max_k, batch_size=args.batch_size, device=device)

        summary["results"][label] = {}
        for k in eval_ks:
            candidates = aq.make_candidates(
                rng=rng, future_model_actions=batch.model_actions, k=k,
                plan_horizon=args.plan_horizon,
                num_candidates=args.num_candidates, scale=args.candidate_scale)
            true_costs = aq.score_candidates_from_emb(
                model=model, init_emb=true_emb[:, k], goal_emb=goal_emb,
                candidates=candidates, batch_size=args.batch_size, device=device)

            # build latent variants: drift + each projection of drift
            variants = {"drift": pred_emb[:, k].astype(np.float32)}
            nn_cache = [None]
            for name, (kind, param) in projs.items():
                variants[f"proj_{name}"] = apply_projector(
                    pred_emb[:, k].astype(np.float32), kind, param,
                    mean, evecs, bank, args.knn_m, nn_cache).astype(np.float32)

            best_true = true_costs.argmin(axis=1)
            sidx = np.arange(args.num_samples)
            true_best_cost = true_costs[sidx, best_true]

            k_res = {"latent_mse": {}, "top1_same": {}, "top5_overlap": {},
                     "cost_spearman": {}, "true_cost_regret": {}}
            for name, emb_v in variants.items():
                v_costs = aq.score_candidates_from_emb(
                    model=model, init_emb=emb_v, goal_emb=goal_emb,
                    candidates=candidates, batch_size=args.batch_size,
                    device=device)
                best_v = v_costs.argmin(axis=1)
                lat_mse = float(np.mean((emb_v - true_emb[:, k]) ** 2))
                top1 = float(np.mean(best_true == best_v))
                top5 = float(np.nanmean(aq.topk_overlap_rows(true_costs, v_costs, 5)))
                spr = float(np.nanmean(aq.spearman_rows(true_costs, v_costs)))
                regret = float(np.mean(true_costs[sidx, best_v] - true_best_cost))
                k_res["latent_mse"][name] = lat_mse
                k_res["top1_same"][name] = top1
                k_res["top5_overlap"][name] = top5
                k_res["cost_spearman"][name] = spr
                k_res["true_cost_regret"][name] = regret
                rows.append({"shift": label, "k": k, "variant": name,
                             "latent_mse": lat_mse, "top1_same": top1,
                             "top5_overlap": top5, "cost_spearman": spr,
                             "true_cost_regret": regret})
            summary["results"][label][str(k)] = k_res
            print(f"[phase7C] {label} k={k} done", flush=True)

    with (out_dir / "phase7_action_proj_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shift", "k", "variant", "latent_mse",
                            "top1_same", "top5_overlap", "cost_spearman",
                            "true_cost_regret"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "phase7_action_proj_summary.json").write_text(json.dumps(summary, indent=2))

    print("[phase7C] top1_same / top5_overlap / spearman vs TRUE, k=10:")
    for label, _ in shifts:
        r = summary["results"][label].get("10")
        if not r:
            continue
        print(f"  {label}:")
        for name in ["drift"] + [f"proj_{n}" for n in projs]:
            print(f"    {name:14} mse={r['latent_mse'][name]:.4f} "
                  f"top1={r['top1_same'][name]:.2f} top5={r['top5_overlap'][name]:.2f} "
                  f"spr={r['cost_spearman'][name]:.3f} regret={r['true_cost_regret'][name]:.1f}")
    print(f"[phase7C] elapsed {summary['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
