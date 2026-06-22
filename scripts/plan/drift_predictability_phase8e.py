"""Phase 8e: is open-loop drift magnitude predictable WITHOUT ground truth?

Direction (2) -- uncertainty-discounted planning -- requires that the planner can
estimate how much to distrust the rollout latent at step k using only quantities
available at test time (no true z_k). This script tests that premise on the saved
Phase 3 ID rollout latents.

target   : drift_k = mean_d (zhat_k - z_k)^2          (unknown at test)
features : k, ||zhat_k||, ||zhat_k - zhat_{k-1}|| (step size), ||zhat_0||,
           ||zhat_k - zhat_0|| (distance travelled), cumulative |action|

Two questions:
  - pooled: can a regressor predict drift across (sample,k) (R^2)?
  - within-k: at FIXED k, is the per-sample drift variation predictable, or is
    drift essentially a function of k alone (=> a simple k-schedule suffices)?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-dirs", required=True, help="comma-separated seed dirs")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8e_drift_pred")
    ap.add_argument("--num-samples", type=int, default=600)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dirs = [Path(x) for x in args.phase3_dirs.split(",") if x]

    # rebuild model_actions per seed (seeds match phase3 runs by dir name suffix)
    dataset = p3.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )

    feats_all, targ_all, k_all = [], [], []
    for d in dirs:
        seed = int(str(d).rstrip("/").split("seed")[-1])
        rep = np.load(d / "phase3_id_replay_outputs.npz")
        true = rep["true_emb"].astype(np.float32)
        pred = rep["pred_interval_10"].astype(np.float32)
        n, K, D = true.shape
        batch = p3.build_window_batch(
            dataset, num_samples=n, max_k=args.max_k,
            goal_offset=args.goal_offset, action_block=args.action_block, seed=seed,
        )
        act_mag = np.linalg.norm(batch.model_actions, axis=-1)  # (n, max_k)
        for k in range(1, K):
            drift = np.mean((pred[:, k] - true[:, k]) ** 2, axis=-1)  # (n,)
            f = np.stack([
                np.full(n, k, np.float32),
                np.linalg.norm(pred[:, k], axis=1),
                np.linalg.norm(pred[:, k] - pred[:, k - 1], axis=1),
                np.linalg.norm(pred[:, 0], axis=1),
                np.linalg.norm(pred[:, k] - pred[:, 0], axis=1),
                act_mag[:, :k].sum(1),
            ], axis=1)
            feats_all.append(f); targ_all.append(drift); k_all.append(np.full(n, k))

    X = np.concatenate(feats_all); y = np.concatenate(targ_all); ks = np.concatenate(k_all)
    fnames = ["k", "pred_norm", "step_size", "init_norm", "dist_travelled", "cum_action"]

    # pooled R^2 (5-fold CV)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, n_jobs=-1, random_state=0)
    yhat = cross_val_predict(rf, X, y, cv=5, n_jobs=-1)
    pooled_r2 = float(r2_score(y, yhat))

    # baseline: predict drift = mean drift at that k (k-schedule only)
    kmean = {k: y[ks == k].mean() for k in np.unique(ks)}
    y_konly = np.array([kmean[k] for k in ks])
    kschedule_r2 = float(r2_score(y, y_konly))

    # within-k R^2 (does per-sample variation predict beyond the k-mean?)
    within = {}
    for k in np.unique(ks):
        mask = ks == k
        Xk = X[mask][:, 1:]  # drop k feature
        yk = y[mask]
        if yk.std() < 1e-9:
            continue
        yhk = cross_val_predict(
            RandomForestRegressor(n_estimators=200, max_depth=6, n_jobs=-1, random_state=0),
            Xk, yk, cv=5, n_jobs=-1)
        within[int(k)] = {
            "r2": float(r2_score(yk, yhk)),
            "cv_drift_over_mean": float(yk.std() / yk.mean()),
        }

    # per-feature correlation with drift (pooled)
    corr = {f: float(np.corrcoef(X[:, i], y)[0, 1]) for i, f in enumerate(fnames)}

    summary = {
        "n_rows": int(len(y)),
        "pooled_r2_all_features": pooled_r2,
        "kschedule_only_r2": kschedule_r2,
        "within_k_r2": within,
        "feature_correlation_with_drift": corr,
    }
    (out_dir / "phase8e_drift_pred_summary.json").write_text(json.dumps(summary, indent=2))

    print("[8e] drift predictability (ID open-loop):")
    print(f"  pooled R^2 (all features)     = {pooled_r2:.3f}")
    print(f"  k-schedule-only R^2 (drift~k) = {kschedule_r2:.3f}")
    print(f"  => extra explainable beyond k = {pooled_r2 - kschedule_r2:.3f}")
    print("  within-k R^2 (per-sample variation predictable at fixed k?):")
    for k, v in within.items():
        print(f"    k={k:2d}  R2={v['r2']:+.3f}  drift CoV={v['cv_drift_over_mean']:.2f}")
    print("  feature correlations with drift:")
    for f, c in corr.items():
        print(f"    {f:16s} {c:+.3f}")
    print(f"[8e] wrote {out_dir}")


if __name__ == "__main__":
    main()
