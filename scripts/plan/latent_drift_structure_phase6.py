"""Phase 6: structural diagnostics of LeWM latent drift.

Phases 2-5 measured *how much* the open-loop predicted latent drifts away from
the true encoded latent (MSE / L2 / cosine / cost-rank / regret). They never
measured *how* it drifts. This script characterizes the error vector

    delta_k = zhat_k - z_k

so we can pick the right fix before moving to the solution phase:

1. bias vs diffusion: how much of the squared error energy is a consistent
   additive offset (E[delta]) vs sample-dependent scatter. High bias fraction =>
   a single learned/estimated correction vector removes most of it (cheap fix).
   High diffusion => need more frequent re-grounding or uncertainty-aware
   planning (planner-level fix).
2. dimension concentration: is the error concentrated in a few latent dims
   (participation ratio, top-10 share) or spread across all 192.
3. norm collapse vs explosion: does zhat shrink toward a mean/prior or blow up
   relative to the true latent norm.
4. direction stability across k: is the bias direction consistent over the
   horizon (one correction direction works for all k) or rotating.
5. natural-scale normalization: express drift and encoder-shift relative to the
   natural spread of true ID latents, so "MSE = 0.5" becomes interpretable.

This is a pure post-hoc analysis of the Phase 3 saved rollout latents
(`true_emb`, `pred_interval_{1,2,3,5,10}`, `*_same_state_embeddings.npz`). It
reuses the exact same n=200, seed=42, goal_offset=50 windows, so the structural
numbers are directly comparable to Phase 2/3 magnitude numbers and introduce no
new sampling noise.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


SHIFTS = ("id", "visual", "geometry")


def load_replay(phase3_dir: Path, shift: str) -> dict[str, np.ndarray]:
    path = phase3_dir / f"phase3_{shift}_replay_outputs.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files}


def load_same_state(phase3_dir: Path, shift: str) -> np.ndarray:
    path = phase3_dir / f"phase3_{shift}_same_state_embeddings.npz"
    return np.load(path)["emb"]


def natural_spread(true_emb: np.ndarray) -> np.ndarray:
    """Per-k mean-over-dim variance of true latent across samples.

    Same units as the mean-over-dim MSE used elsewhere, so drift / spread is a
    clean dimensionless ratio. Shape in: (n, K, D). Out: (K,).
    """
    var_per_dim = true_emb.var(axis=0)  # (K, D)
    return var_per_dim.mean(axis=-1)  # (K,)


def error_structure(delta: np.ndarray) -> dict[str, np.ndarray]:
    """Structural stats of an error tensor delta of shape (n, K, D), per k."""
    n, K, D = delta.shape
    # per-dim consistent bias and per-dim energy
    bias = delta.mean(axis=0)  # (K, D)
    energy_per_dim = (delta * delta).mean(axis=0)  # (K, D) = E_n[delta_d^2]

    e_total = energy_per_dim.mean(axis=-1)  # (K,) mean-over-dim MSE
    e_bias = (bias * bias).mean(axis=-1)  # (K,) energy of consistent offset
    with np.errstate(invalid="ignore", divide="ignore"):
        bias_fraction = np.where(e_total > 0, e_bias / e_total, np.nan)

    # dimension concentration over per-dim energy
    q = energy_per_dim  # (K, D)
    q_sum = q.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        qhat = np.where(q_sum > 0, q / q_sum, np.nan)  # (K, D) sum to 1
    participation_ratio = 1.0 / np.nansum(qhat * qhat, axis=-1)  # (K,) in [1, D]
    sorted_q = np.sort(q, axis=-1)[:, ::-1]
    top10_share = sorted_q[:, :10].sum(axis=-1) / np.where(
        q_sum[:, 0] > 0, q_sum[:, 0], np.nan
    )

    # direction stability of the consistent bias across k
    bias_norm = np.linalg.norm(bias, axis=-1)  # (K,)
    cos_adj = np.full(K, np.nan)
    cos_ref = np.full(K, np.nan)
    ref = bias[-1]  # final-k bias direction
    ref_norm = np.linalg.norm(ref)
    for k in range(K):
        if k >= 1 and bias_norm[k] > 0 and bias_norm[k - 1] > 0:
            cos_adj[k] = float(
                bias[k] @ bias[k - 1] / (bias_norm[k] * bias_norm[k - 1])
            )
        if bias_norm[k] > 0 and ref_norm > 0:
            cos_ref[k] = float(bias[k] @ ref / (bias_norm[k] * ref_norm))

    return {
        "e_total": e_total,
        "bias_fraction": bias_fraction,
        "participation_ratio": participation_ratio,
        "top10_dim_share": top10_share,
        "bias_norm": bias_norm,
        "bias_cos_adjacent_k": cos_adj,
        "bias_cos_to_final_k": cos_ref,
    }


def norm_behavior(pred: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    pred_norm = np.linalg.norm(pred, axis=-1).mean(axis=0)  # (K,)
    true_norm = np.linalg.norm(true, axis=-1).mean(axis=0)  # (K,)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(true_norm > 0, pred_norm / true_norm, np.nan)
    return {"pred_norm": pred_norm, "true_norm": true_norm, "norm_ratio": ratio}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase3-dir",
        default="outputs/lghl_phase3_n200_k10_goal50",
        help="Directory holding phase3 *_replay_outputs.npz files.",
    )
    parser.add_argument("--output-dir", default="outputs/lghl_phase6_structure")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument(
        "--intervals",
        default="1,2,3,5,10",
        help="Re-grounding intervals present in phase3 npz (10 = pure open-loop).",
    )
    args = parser.parse_args()

    phase3_dir = Path(args.phase3_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = [int(x) for x in args.intervals.split(",") if x]

    replay = {s: load_replay(phase3_dir, s) for s in SHIFTS}
    id_true = replay["id"]["true_emb"]
    id_spread = natural_spread(id_true)  # (K,)
    K = id_true.shape[1]

    rows: list[dict] = []
    summary: dict = {
        "meta": {
            "phase3_dir": str(phase3_dir),
            "intervals": intervals,
            "action_block": args.action_block,
            "shape": list(id_true.shape),
            "id_natural_spread_per_k": id_spread.tolist(),
        },
        "drift_structure": {},
        "encoder_shift_normalized": {},
    }

    def emit(analysis, shift, interval, metrics: dict[str, np.ndarray]):
        for k in range(K):
            row = {
                "analysis": analysis,
                "shift": shift,
                "interval": "" if interval is None else interval,
                "k": k,
                "env_steps": k * args.action_block,
            }
            for name, arr in metrics.items():
                row[name] = float(arr[k])
            rows.append(row)

    # --- drift structure for each shift x interval ---
    for shift in SHIFTS:
        true = replay[shift]["true_emb"]
        spread = natural_spread(true)
        summary["drift_structure"][shift] = {}
        for interval in intervals:
            key = f"pred_interval_{interval}"
            if key not in replay[shift]:
                continue
            pred = replay[shift][key]
            delta = pred - true
            struct = error_structure(delta)
            norms = norm_behavior(pred, true)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.where(
                    spread > 0, struct["e_total"] / spread, np.nan
                )
            metrics = {
                **struct,
                **norms,
                "natural_spread": spread,
                "drift_over_spread": rel,
            }
            emit("drift_structure", shift, interval, metrics)
            summary["drift_structure"][shift][str(interval)] = {
                name: np.asarray(arr).tolist() for name, arr in metrics.items()
            }

    # --- encoder-shift normalized by natural spread (same-state) ---
    id_same = load_same_state(phase3_dir, "id")
    same_spread = natural_spread(id_same)
    for shift in ("visual", "geometry"):
        same = load_same_state(phase3_dir, shift)
        delta = same - id_same
        e_total = (delta * delta).mean(axis=0).mean(axis=-1)  # (K,)
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.where(same_spread > 0, e_total / same_spread, np.nan)
        struct = error_structure(delta)
        metrics = {
            "e_total": e_total,
            "natural_spread": same_spread,
            "shift_over_spread": rel,
            "bias_fraction": struct["bias_fraction"],
            "participation_ratio": struct["participation_ratio"],
            "top10_dim_share": struct["top10_dim_share"],
        }
        emit("encoder_shift_normalized", shift, None, metrics)
        summary["encoder_shift_normalized"][shift] = {
            name: np.asarray(arr).tolist() for name, arr in metrics.items()
        }

    # --- write outputs ---
    fieldnames = [
        "analysis", "shift", "interval", "k", "env_steps",
        "e_total", "drift_over_spread", "shift_over_spread", "natural_spread",
        "bias_fraction", "bias_norm", "bias_cos_adjacent_k", "bias_cos_to_final_k",
        "participation_ratio", "top10_dim_share",
        "pred_norm", "true_norm", "norm_ratio",
    ]
    csv_path = out_dir / "phase6_structure_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    (out_dir / "phase6_structure_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    # --- console headline for open-loop (interval=10) ---
    print("[phase6] open-loop (interval=10) drift structure, key k:")
    hdr = f"{'shift':9} {'k':>2} {'mse':>7} {'drift/spread':>12} {'bias_frac':>9} {'PR':>6} {'norm_ratio':>10} {'cos_adj':>7}"
    print(hdr)
    for shift in SHIFTS:
        d = summary["drift_structure"][shift]["10"]
        for k in (1, 5, 10):
            print(
                f"{shift:9} {k:>2} {d['e_total'][k]:7.4f} "
                f"{d['drift_over_spread'][k]:12.3f} {d['bias_fraction'][k]:9.3f} "
                f"{d['participation_ratio'][k]:6.1f} {d['norm_ratio'][k]:10.3f} "
                f"{d['bias_cos_adjacent_k'][k]:7.3f}"
            )
    print("[phase6] encoder-shift normalized by natural latent spread (k=0):")
    for shift in ("visual", "geometry"):
        e = summary["encoder_shift_normalized"][shift]
        print(
            f"  {shift:9} mse={e['e_total'][0]:.4f} "
            f"shift/spread={e['shift_over_spread'][0]:.3f} "
            f"bias_frac={e['bias_fraction'][0]:.3f} PR={e['participation_ratio'][0]:.1f}"
        )
    print(f"[phase6] wrote {csv_path}")


if __name__ == "__main__":
    main()
