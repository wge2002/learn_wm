"""Phase 7 (F): is the on-manifold ID drift a lag/sluggish-prediction error or a
scrambled wrong-branch error?

Phase 7B found the ID open-loop drift is largely *on* the data manifold (a
valid-looking but wrong latent), so a manifold-projection restoring force does
not recover it. This script characterizes *what kind* of wrong state it is.

For each window i and open-loop step k, it finds which true latent z_{i,j} in the
same window the prediction zhat_{i,k} is closest to:

    nearest_j(i,k) = argmin_j || zhat_{i,k} - z_{i,j} ||

- if nearest_j is systematically < k, the model predicts too little motion
  (lag / inertia / regression toward recent past) -> a dynamics-magnitude problem
  fixable by better long-horizon dynamics, not by re-grounding alone.
- if nearest_j ~ k but distance is large, the error is off-trajectory (wrong
  branch / scrambled).
- offset = nearest_j - k summarizes the timing bias.

Pure numpy over saved Phase 3 rollout latents; can run over several seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def analyze_dir(phase3_dir: Path, shifts) -> dict:
    out = {}
    for shift in shifts:
        path = phase3_dir / f"phase3_{shift}_replay_outputs.npz"
        if not path.exists():
            continue
        rep = np.load(path)
        true = rep["true_emb"].astype(np.float32)  # (n, K, D)
        pred = rep["pred_interval_10"].astype(np.float32)
        n, K, D = true.shape
        # pairwise dist zhat_{i,k} to z_{i,j}: (n, K_pred, K_true)
        # ||a-b||^2 = |a|^2 + |b|^2 - 2 a.b
        res = {}
        for k in range(1, K):
            zh = pred[:, k]  # (n, D)
            d2 = (
                (zh * zh).sum(1)[:, None]
                + (true * true).sum(2)
                - 2 * np.einsum("nd,nkd->nk", zh, true)
            )  # (n, K_true)
            nearest_j = d2.argmin(axis=1)  # (n,)
            offset = nearest_j - k
            dist_to_k = np.sqrt(np.maximum(d2[:, k], 0))
            dist_to_nearest = np.sqrt(np.maximum(d2.min(axis=1), 0))
            res[k] = {
                "mean_nearest_j": float(nearest_j.mean()),
                "mean_offset": float(offset.mean()),
                "median_offset": float(np.median(offset)),
                "frac_lag": float(np.mean(offset < 0)),
                "frac_ontime": float(np.mean(offset == 0)),
                "frac_overshoot": float(np.mean(offset > 0)),
                "mean_dist_to_k": float(dist_to_k.mean()),
                "mean_dist_to_nearest": float(dist_to_nearest.mean()),
                "onmanifold_gap": float(
                    (dist_to_k.mean() - dist_to_nearest.mean())
                ),
            }
        out[shift] = res
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase3-dirs", required=True,
                    help="comma-separated phase3 output dirs (multi-seed)")
    ap.add_argument("--output-dir", default="outputs/lghl_phase7_lag")
    ap.add_argument("--shifts", default="id,visual,geometry")
    args = ap.parse_args()

    dirs = [Path(x) for x in args.phase3_dirs.split(",") if x]
    shifts = [s for s in args.shifts.split(",") if s]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_dir = [analyze_dir(d, shifts) for d in dirs]
    per_dir = [p for p in per_dir if p]

    # aggregate across dirs (seeds)
    rows = []
    agg = {}
    fields = ["mean_nearest_j", "mean_offset", "frac_lag", "frac_ontime",
              "frac_overshoot", "mean_dist_to_k", "mean_dist_to_nearest",
              "onmanifold_gap"]
    shift_set = sorted({s for p in per_dir for s in p})
    for shift in shift_set:
        agg[shift] = {}
        ks = sorted(per_dir[0][shift].keys())
        for k in ks:
            agg[shift][k] = {}
            for f in fields:
                vals = np.array([p[shift][k][f] for p in per_dir if shift in p])
                agg[shift][k][f + "_mean"] = float(vals.mean())
                agg[shift][k][f + "_std"] = float(vals.std())
            rows.append({"shift": shift, "k": k,
                         **{f: agg[shift][k][f + "_mean"] for f in fields}})

    with (out_dir / "phase7_lag_summary.csv").open("w", newline="") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["shift", "k"] + fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out_dir / "phase7_lag_summary.json").write_text(
        json.dumps({"n_dirs": len(per_dir), "agg": agg}, indent=2))

    print(f"[phase7F] aggregated over {len(per_dir)} seed dir(s)")
    print("k -> mean nearest true-j (where the open-loop prediction actually lands)")
    for shift in shift_set:
        print(f"  {shift}:")
        for k in sorted(agg[shift]):
            a = agg[shift][k]
            print(f"    k={k:2d}  nearest_j={a['mean_nearest_j_mean']:.2f} "
                  f"offset={a['mean_offset_mean']:+.2f} "
                  f"lag/ontime/over={a['frac_lag_mean']:.2f}/"
                  f"{a['frac_ontime_mean']:.2f}/{a['frac_overshoot_mean']:.2f} "
                  f"dist_k={a['mean_dist_to_k_mean']:.3f} "
                  f"dist_near={a['mean_dist_to_nearest_mean']:.3f}")
    print(f"[phase7F] wrote {out_dir/'phase7_lag_summary.csv'}")


if __name__ == "__main__":
    main()
