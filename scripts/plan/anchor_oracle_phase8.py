"""Phase 8 (anchor probe 1): oracle ceiling of a discrete commitment anchor.

The "commitment anchor" idea is a dedicated discrete code, emitted every G model
steps, that re-anchors the open-loop latent rollout to bound the random-walk
drift measured in Phases 6/7. Before training an anchor predictor, this script
measures the *ceiling*: assume a perfect anchor predictor that outputs the
discrete code of the true latent at each anchor point, and ask

  - does periodic discrete re-anchoring actually bound open-loop drift?
  - how large a codebook C is needed (discreteness precision cost)?
  - how far is discrete (codebook) from continuous (exact-latent) re-anchoring?

It reuses the Phase 3 windows/rollout exactly (n=200, seed=42, goal_offset=50),
so drift numbers are directly comparable. Three rollout conditions at matched G:

  open-loop        : never re-anchor (== Phase 3 interval=10 baseline)
  anchor_continuous: every G steps reseed rollout with the *exact* true latent
                     z_{seg} (C=inf ceiling for any latent re-anchoring)
  anchor_C<C>      : every G steps reseed with quantize(z_{seg}) under a size-C
                     k-means codebook fit on the 22k ID latent bank

The codebook quantization is the only thing standing in for a real anchor; the
*which true latent* is given (oracle), so this is an upper bound, not a method.
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
import latent_drift_phase3 as p3  # noqa: E402


@torch.inference_mode()
def anchored_rollout(
    *, model, true_emb, model_actions, anchor_interval, quantize, device,
    batch_size,
):
    """Open-loop rollout reseeded every `anchor_interval` steps with a (possibly
    quantized) true latent. Mirrors p3.regrounded_rollout but seeds each segment
    from a latent instead of a re-encoded frame."""
    dtype = next(model.parameters()).dtype
    n, num_k, D = true_emb.shape
    max_k = num_k - 1
    pred = np.empty_like(true_emb)
    pred[:, 0] = true_emb[:, 0]

    for seg in range(0, max_k, anchor_interval):
        horizon = min(anchor_interval, max_k - seg)
        seed_latent = quantize(true_emb[:, seg])  # (n, D)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            emb = torch.from_numpy(seed_latent[s:e]).to(device=device, dtype=dtype)
            emb = emb[:, None, None, :]  # (b, S=1, H=1, D)
            acts = torch.from_numpy(
                model_actions[s:e, seg : seg + horizon]
            ).to(device=device, dtype=dtype)
            dummy_pixels = torch.zeros(
                e - s, 1, 1, 3, 224, 224, dtype=dtype, device=device
            )
            out = model.rollout({"pixels": dummy_pixels, "emb": emb}, acts[:, None])
            seg_pred = out["predicted_emb"][:, 0, 1 : horizon + 1]
            pred[s:e, seg + 1 : seg + horizon + 1] = seg_pred.float().cpu().numpy()
    return pred


def make_quantizer(centroids):
    c_sq = (centroids * centroids).sum(1)

    def q(x):
        d2 = (x * x).sum(1)[:, None] - 2 * x @ centroids.T + c_sq[None, :]
        idx = d2.argmin(1)
        return centroids[idx]

    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--bank-path", default="outputs/lghl_phase7_manifold/id_latent_bank.npy")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8_anchor_oracle")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--anchor-intervals", default="2,3,5")
    ap.add_argument("--codebook-sizes", default="128,512,2048,8192")
    ap.add_argument("--shift", default="id")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = [int(x) for x in args.anchor_intervals.split(",") if x]
    Cs = [int(x) for x in args.codebook_sizes.split(",") if x]
    variations = dict(p3.DEFAULT_SHIFTS)[args.shift]
    t0 = time.time()

    print("[phase8] dataset + windows", flush=True)
    dataset = p3.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = p3.build_window_batch(
        dataset, num_samples=args.num_samples, max_k=args.max_k,
        goal_offset=args.goal_offset, action_block=args.action_block,
        seed=args.seed,
    )
    device = torch.device(args.device)
    model = p3.swm.wm.utils.load_pretrained(args.policy)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f"[phase8] render/encode true_emb shift={args.shift}", flush=True)
    frames = p3.render_replay(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        variations=variations, action_block=args.action_block,
        img_size=args.img_size, seed=args.seed,
    )[0]
    true_emb = p3.encode_frames(
        model=model, frames=frames, batch_size=args.batch_size, device=device
    )
    spread = (true_emb.var(axis=0).mean(axis=-1))  # (K,) natural spread per k

    # codebooks
    from sklearn.cluster import MiniBatchKMeans
    bank = np.load(args.bank_path).astype(np.float32)
    quantizers = {"continuous": (lambda x: x)}
    codebooks = {}
    for C in Cs:
        print(f"[phase8] fitting codebook C={C}", flush=True)
        km = MiniBatchKMeans(n_clusters=C, batch_size=4096, n_init=3,
                             max_iter=100, random_state=0)
        km.fit(bank)
        cent = km.cluster_centers_.astype(np.float32)
        codebooks[C] = cent
        quantizers[f"C{C}"] = make_quantizer(cent)
        # quantization reconstruction error on bank (precision floor)
        q = make_quantizer(cent)
        rec = float(np.mean((q(bank) - bank) ** 2))
        print(f"[phase8]   C={C} bank recon mse={rec:.4f}", flush=True)

    def mse_curve(pred):
        return np.mean((pred - true_emb) ** 2, axis=(0, 2))  # (K,)

    rows = []
    summary = {"meta": {
        "shift": args.shift, "num_samples": args.num_samples,
        "anchor_intervals": intervals, "codebook_sizes": Cs,
        "bank_path": args.bank_path,
    }, "open_loop": {}, "anchored": {}}

    # open-loop baseline (anchor only at k=0)
    print("[phase8] open-loop baseline", flush=True)
    ol = anchored_rollout(
        model=model, true_emb=true_emb, model_actions=batch.model_actions,
        anchor_interval=args.max_k, quantize=quantizers["continuous"],
        device=device, batch_size=args.batch_size)
    ol_mse = mse_curve(ol)
    summary["open_loop"] = {"mse_by_k": ol_mse.tolist()}
    for k in range(args.max_k + 1):
        rows.append({"condition": "open_loop", "anchor_interval": args.max_k,
                     "codebook": "none", "k": k, "env_steps": k * args.action_block,
                     "mse": float(ol_mse[k]),
                     "drift_over_spread": float(ol_mse[k] / spread[k]) if spread[k] > 0 else float("nan")})

    for G in intervals:
        summary["anchored"][str(G)] = {}
        for qname, q in quantizers.items():
            pred = anchored_rollout(
                model=model, true_emb=true_emb,
                model_actions=batch.model_actions, anchor_interval=G,
                quantize=q, device=device, batch_size=args.batch_size)
            mc = mse_curve(pred)
            summary["anchored"][str(G)][qname] = {"mse_by_k": mc.tolist()}
            for k in range(args.max_k + 1):
                rows.append({
                    "condition": "anchored", "anchor_interval": G,
                    "codebook": qname, "k": k, "env_steps": k * args.action_block,
                    "mse": float(mc[k]),
                    "drift_over_spread": float(mc[k] / spread[k]) if spread[k] > 0 else float("nan")})
            print(f"[phase8] G={G} {qname}: mse@k10={mc[args.max_k]:.4f}", flush=True)

    with (out_dir / "phase8_anchor_oracle_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "anchor_interval",
                            "codebook", "k", "env_steps", "mse", "drift_over_spread"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary["elapsed_sec"] = time.time() - t0
    summary["bank_recon_mse"] = {str(C): float(np.mean((make_quantizer(codebooks[C])(bank) - bank) ** 2)) for C in Cs}
    (out_dir / "phase8_anchor_oracle_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[phase8] drift MSE @ k=10 (lower=better). open-loop baseline = %.4f" % ol_mse[args.max_k])
    print("%8s | %10s %s" % ("interval", "continuous", " ".join("C%d" % C for C in Cs)))
    for G in intervals:
        a = summary["anchored"][str(G)]
        cells = " ".join("%6.4f" % a[f"C{C}"]["mse_by_k"][args.max_k] for C in Cs)
        print("%8d | %10.4f %s" % (G, a["continuous"]["mse_by_k"][args.max_k], cells))
    print(f"[phase8] wrote {out_dir} ({summary['elapsed_sec']:.0f}s)")


if __name__ == "__main__":
    main()
