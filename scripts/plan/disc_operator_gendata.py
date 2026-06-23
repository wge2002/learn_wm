"""Build latent-transition sequences on the frozen LeWM latent for the
discrete-operator experiment.

Reads the stored ``pixels`` column of pusht_expert_train.h5 directly (no env
render needed -- ID condition, stored obs == true render), samples expert
windows, encodes z_0..z_K (pooled CLS-192) via the frozen LeWM encoder, and
builds the matching model actions (action_block raw steps StandardScaled and
concatenated, exactly as LeWM eval).

Output npz: z (N, K+1, D)  a (N, K, action_block*raw_adim)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn import preprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402  (reuse encode_frames + thread cfg)


def sample_windows(f, *, num_samples, max_k, action_block, seed):
    ep_off = f["ep_offset"][:]
    ep_len = f["ep_len"][:]
    max_env = max_k * action_block
    required = max_env + 1
    valid = np.flatnonzero(ep_len > required)
    rng = np.random.default_rng(seed)
    eps = rng.choice(valid, size=num_samples, replace=True)
    starts = np.array([rng.integers(0, ep_len[e] - required) for e in eps], dtype=np.int64)
    abs_start = ep_off[eps] + starts  # absolute row in flat arrays
    state_idx = np.arange(0, max_env + 1, action_block)  # K+1 frame offsets
    return abs_start, state_idx, max_env


def gather(f, abs_start, state_idx, max_env, action_scaler, action_block, batch_read=256):
    """Return frames (N, K+1, 224,224,3) uint8 and model actions (N, K, ab*adim)."""
    pixels = f["pixels"]
    action = f["action"]
    N = abs_start.shape[0]
    K1 = state_idx.shape[0]
    frames = np.empty((N, K1, 224, 224, 3), dtype=np.uint8)
    raw_adim = action.shape[1]
    model_actions = np.empty((N, K1 - 1, action_block * raw_adim), dtype=np.float32)
    for s in range(0, N, batch_read):
        e = min(s + batch_read, N)
        for j in range(s, e):
            a0 = int(abs_start[j])
            frames[j] = pixels[a0 + state_idx]  # (K1,224,224,3)
            raw = action[a0:a0 + max_env]       # (max_env, adim)
            scaled = action_scaler.transform(raw).reshape(K1 - 1, action_block * raw_adim)
            model_actions[j] = scaled
    return frames, model_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--h5", default="/home/jovyan/.stable_worldmodel/pusht_expert_train.h5")
    ap.add_argument("--output", default="outputs/disc_operator/latent_seq.npz")
    ap.add_argument("--num-samples", type=int, default=40000)
    ap.add_argument("--val-samples", type=int, default=4000)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--batch-size", type=int, default=256, help="encode batch")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    f = h5py.File(args.h5, "r")
    # fit StandardScaler on all actions (drop NaN), as in build_window_batch
    act_all = f["action"][:]
    act_all = act_all[~np.isnan(act_all).any(axis=1)]
    scaler = preprocessing.StandardScaler().fit(act_all)

    def build(n, seed, tag):
        abs_start, state_idx, max_env = sample_windows(
            f, num_samples=n, max_k=args.max_k, action_block=args.action_block, seed=seed)
        frames, macts = gather(f, abs_start, state_idx, max_env, scaler, args.action_block)
        print(f"[gendata] {tag}: encoding frames {frames.shape}", flush=True)
        z = p3.encode_frames(model=model, frames=frames,
                             batch_size=args.batch_size, device=device)  # (n,K1,D)
        return z.astype(np.float32), macts.astype(np.float32)

    z_tr, a_tr = build(args.num_samples, args.seed, "train")
    z_va, a_va = build(args.val_samples, args.val_seed, "val")
    np.savez_compressed(out, z=z_tr, a=a_tr, z_val=z_va, a_val=a_va)
    print(f"[gendata] saved {out} z={z_tr.shape} a={a_tr.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
