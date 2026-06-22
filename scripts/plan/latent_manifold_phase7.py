"""Phase 7 (A+B): latent-manifold geometry and training-free projection recovery.

Phase 6 showed open-loop drift is isotropic, dimension-spread diffusion. The
central hypothesis from the idea discussion is that this diffusion pushes the
predicted latent *off the data manifold* (into directions the encoder never
produces), and that the most natural fix is a "restoring force" that projects
the prediction back onto the manifold (the mechanism behind discrete/codebook
world models).

This script tests that hypothesis without any training:

A. manifold geometry of true ID latents
   - anisotropy: mean pairwise cosine of random true latents
   - PCA spectrum -> effective dimension (participation ratio of eigenvalues)

B. off-manifold decomposition + projection recovery of the saved open-loop
   drift (Phase 3 `pred_interval_10` vs `true_emb`)
   - energy-capture curves: what fraction of the *error* delta = zhat - z lives
     in the top-d PCA subspace of the data manifold, vs what fraction of the
     natural latent fluctuation does. If the error is spread far more broadly
     than the data -> off-manifold.
   - projection recovery: snap zhat back toward the manifold (PCA-denoise / kNN)
     and measure residual MSE to the true z_k. Large recovery => the drift is
     mostly off-manifold junk that a restoring force removes. Small recovery =>
     the drift is an on-manifold wrong state and projection cannot save it.
   - NN density: distance from zhat to nearest true latent vs distance from a
     true latent to its nearest neighbor (is zhat in a sparse, off-data region).

The latent bank is built from the same ID-rendered dataset states the rollout
uses (reuses Phase 3 window/render/encode helpers), so the manifold matches the
rollout latent distribution. Bank build is the only GPU step; everything else is
numpy and reused by Phase 7 Job C.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402


def build_bank(args, device) -> np.ndarray:
    import stable_worldmodel as swm
    import torch

    print("[phase7] loading dataset for bank", flush=True)
    dataset = swm.data.load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = p3.build_window_batch(
        dataset,
        num_samples=args.bank_windows,
        max_k=args.max_k,
        goal_offset=args.goal_offset,
        action_block=args.action_block,
        seed=args.bank_seed,
    )
    print("[phase7] loading model for bank", flush=True)
    model = swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    frames = p3.render_state_sequence(
        env_name=args.env_name,
        states_by_k=batch.states_by_k,
        goal_states=batch.goal_states,
        variations=(),  # id render
        img_size=args.img_size,
        seed=args.bank_seed,
    )
    emb = p3.encode_frames(
        model=model, frames=frames, batch_size=args.batch_size, device=device
    )
    bank = emb.reshape(-1, emb.shape[-1]).astype(np.float32)
    return bank


def pca_fit(bank: np.ndarray):
    mean = bank.mean(axis=0)
    X = bank - mean
    cov = (X.T @ X) / (X.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)  # ascending
    evals = evals[::-1]
    evecs = evecs[:, ::-1]
    evals = np.clip(evals, 0, None)
    return mean, evecs, evals


def effective_dim(evals: np.ndarray) -> float:
    s = evals.sum()
    return float((s * s) / (evals * evals).sum()) if s > 0 else float("nan")


def mean_pairwise_cosine(bank: np.ndarray, n_pairs: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, bank.shape[0], size=(n_pairs, 2))
    a = bank[idx[:, 0]]
    b = bank[idx[:, 1]]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    ok = den > 0
    return float((num[ok] / den[ok]).mean())


def energy_capture(vectors: np.ndarray, evecs: np.ndarray, ds: list[int]) -> dict:
    """Fraction of squared norm of each vector captured by top-d PCA dirs.

    vectors: (N, D) (already mean-subtracted if appropriate). Returns mean
    fraction over N for each d.
    """
    coeff = vectors @ evecs  # (N, D) energy per component
    energy = coeff * coeff
    total = energy.sum(axis=1)
    out = {}
    for d in ds:
        cap = energy[:, :d].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(total > 0, cap / total, np.nan)
        out[d] = float(np.nanmean(frac))
    return out


def pca_denoise(x: np.ndarray, mean, evecs, d: int) -> np.ndarray:
    Vd = evecs[:, :d]
    return mean + (x - mean) @ Vd @ Vd.T


def knn_indices(query: np.ndarray, bank: np.ndarray, m: int, block: int = 2048):
    """Return indices of m nearest bank rows for each query row (L2)."""
    bank_sq = (bank * bank).sum(axis=1)  # (M,)
    out = np.empty((query.shape[0], m), dtype=np.int64)
    for s in range(0, query.shape[0], block):
        q = query[s : s + block]
        d2 = (q * q).sum(axis=1)[:, None] - 2 * q @ bank.T + bank_sq[None, :]
        out[s : s + block] = np.argpartition(d2, kth=m - 1, axis=1)[:, :m]
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument(
        "--phase3-dir", default="outputs/lghl_phase3_n200_k10_goal50"
    )
    ap.add_argument("--output-dir", default="outputs/lghl_phase7_manifold")
    ap.add_argument("--bank-path", default=None, help="precomputed bank .npy")
    ap.add_argument("--bank-windows", type=int, default=1500)
    ap.add_argument("--bank-seed", type=int, default=7)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--knn-m", type=int, default=8)
    ap.add_argument("--alphas", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--pca-ds", default="2,5,10,20,30,50,80,120,192")
    ap.add_argument("--eval-ks", default="1,5,10")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    import torch

    device = torch.device(args.device)

    # --- bank ---
    if args.bank_path and Path(args.bank_path).exists():
        bank = np.load(args.bank_path).astype(np.float32)
        print(f"[phase7] loaded bank {bank.shape} from {args.bank_path}", flush=True)
    else:
        bank = build_bank(args, device)
        np.save(out_dir / "id_latent_bank.npy", bank)
        print(f"[phase7] built bank {bank.shape}", flush=True)

    pca_ds = [int(x) for x in args.pca_ds.split(",") if x]
    pca_ds = [min(d, bank.shape[1]) for d in pca_ds]
    eval_ks = [int(x) for x in args.eval_ks.split(",") if x]
    alphas = [float(x) for x in args.alphas.split(",") if x]

    mean, evecs, evals = pca_fit(bank)
    eff_dim = effective_dim(evals)
    aniso_cos = mean_pairwise_cosine(bank, n_pairs=200000, seed=1)

    # cumulative variance captured by top-d
    cum_var = (np.cumsum(evals) / evals.sum()).tolist()

    summary = {
        "meta": {
            "bank_shape": list(bank.shape),
            "phase3_dir": args.phase3_dir,
            "pca_ds": pca_ds,
            "eval_ks": eval_ks,
            "alphas": alphas,
            "knn_m": args.knn_m,
        },
        "geometry": {
            "effective_dim_participation_ratio": eff_dim,
            "ambient_dim": int(bank.shape[1]),
            "mean_pairwise_cosine": aniso_cos,
            "top_eigenvalues": evals[:20].tolist(),
            "cum_variance_at_ds": {d: cum_var[d - 1] for d in pca_ds},
        },
        "off_manifold": {},
        "projection_recovery": {},
        "nn_density": {},
    }
    rows: list[dict] = []

    # natural latent fluctuation energy capture (baseline manifold shape)
    nat = bank - mean
    nat_capture = energy_capture(nat, evecs, pca_ds)

    for shift in ("id", "visual", "geometry"):
        rep = np.load(
            Path(args.phase3_dir) / f"phase3_{shift}_replay_outputs.npz"
        )
        true = rep["true_emb"].astype(np.float32)  # (n, K, D)
        pred = rep["pred_interval_10"].astype(np.float32)
        n, K, D = true.shape

        summary["off_manifold"][shift] = {}
        summary["projection_recovery"][shift] = {}
        summary["nn_density"][shift] = {}

        for k in eval_ks:
            zt = true[:, k]  # (n, D)
            zh = pred[:, k]
            delta = zh - zt
            raw_mse = mse(zh, zt)

            # off-manifold: energy capture of the error vs natural fluctuation
            err_capture = energy_capture(delta, evecs, pca_ds)
            summary["off_manifold"][shift][str(k)] = {
                "raw_mse": raw_mse,
                "error_energy_capture_by_d": err_capture,
                "natural_energy_capture_by_d": nat_capture,
            }
            for d in pca_ds:
                rows.append({
                    "analysis": "off_manifold", "shift": shift, "k": k,
                    "param": d, "method": "energy_capture",
                    "value": err_capture[d],
                    "baseline": nat_capture[d], "raw_mse": raw_mse,
                })

            # projection recovery: PCA-denoise sweep d
            rec = {"pca_denoise": {}, "knn_snap": {}, "knn_soft": {}}
            for d in pca_ds:
                zproj = pca_denoise(zh, mean, evecs, d)
                rec["pca_denoise"][d] = mse(zproj, zt)
                rows.append({
                    "analysis": "projection_recovery", "shift": shift, "k": k,
                    "param": d, "method": "pca_denoise",
                    "value": mse(zproj, zt), "baseline": raw_mse,
                })
            # kNN snap (1-NN) and soft blends toward kNN mean
            nn = knn_indices(zh, bank, args.knn_m)
            nn1 = bank[nn[:, 0]]
            rec["knn_snap"][1] = mse(nn1, zt)
            rows.append({
                "analysis": "projection_recovery", "shift": shift, "k": k,
                "param": 1, "method": "knn_snap",
                "value": mse(nn1, zt), "baseline": raw_mse,
            })
            knn_mean = bank[nn].mean(axis=1)
            for a in alphas:
                blend = (1 - a) * zh + a * knn_mean
                rec["knn_soft"][a] = mse(blend, zt)
                rows.append({
                    "analysis": "projection_recovery", "shift": shift, "k": k,
                    "param": a, "method": "knn_soft",
                    "value": mse(blend, zt), "baseline": raw_mse,
                })
            summary["projection_recovery"][shift][str(k)] = {
                "raw_mse": raw_mse, **rec,
            }

            # NN density: zhat-to-bank vs true-to-bank
            d_pred = np.linalg.norm(zh - nn1, axis=1).mean()
            nn_true = knn_indices(zt, bank, 2)  # nearest is itself-ish; take 2nd
            d_true = np.linalg.norm(zt - bank[nn_true[:, 1]], axis=1).mean()
            summary["nn_density"][shift][str(k)] = {
                "mean_pred_to_nn": float(d_pred),
                "mean_true_to_nn": float(d_true),
                "ratio": float(d_pred / d_true) if d_true > 0 else float("nan"),
            }
            rows.append({
                "analysis": "nn_density", "shift": shift, "k": k,
                "param": "", "method": "pred_to_nn", "value": float(d_pred),
                "baseline": float(d_true),
            })

    csv_path = out_dir / "phase7_manifold_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["analysis", "shift", "k", "param", "method",
                        "value", "baseline", "raw_mse"],
            extrasaction="ignore",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary["elapsed_sec"] = time.time() - t0
    (out_dir / "phase7_manifold_summary.json").write_text(json.dumps(summary, indent=2))

    # headline
    print(f"[phase7] effective_dim={eff_dim:.1f}/{bank.shape[1]}  "
          f"mean_pairwise_cosine={aniso_cos:.3f}")
    print("[phase7] off-manifold + recovery (open-loop, k=10):")
    for shift in ("id", "visual", "geometry"):
        om = summary["off_manifold"][shift].get("10")
        pr = summary["projection_recovery"][shift].get("10")
        nd = summary["nn_density"][shift].get("10")
        if not om:
            continue
        d_eff = pca_ds[min(range(len(pca_ds)), key=lambda i: abs(pca_ds[i]-30))]
        best_pca = min(pr["pca_denoise"].values())
        print(f"  {shift:9} raw_mse={pr['raw_mse']:.4f} "
              f"err_cap@d{d_eff}={om['error_energy_capture_by_d'][d_eff]:.3f} "
              f"(nat={om['natural_energy_capture_by_d'][d_eff]:.3f}) "
              f"best_pca_recover_mse={best_pca:.4f} "
              f"knn_snap_mse={pr['knn_snap'][1]:.4f} "
              f"nn_ratio={nd['ratio']:.2f}")
    print(f"[phase7] wrote {csv_path}  ({summary['elapsed_sec']:.1f}s)")


if __name__ == "__main__":
    main()
