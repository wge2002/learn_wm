"""Phase 8f: build (z, g, w1, w2) latent triples for training the anchor proposer.

z  = grounded latent at a sampled expert state
g  = latent of that window's goal state (goal_offset ahead)
w1 = true latent 1 model-step ahead   (mid sub-goal for short H)
w2 = true latent 2 model-steps ahead  (mid sub-goal for longer H)

Saved to npz, reused by the proposer trainer. Train/val split by seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402


def build(args, model, device, n, seed):
    dataset = build.dataset
    batch = p3.build_window_batch(
        dataset, num_samples=n, max_k=2, goal_offset=args.goal_offset,
        action_block=args.action_block, seed=seed)
    frames = p3.render_state_sequence(
        env_name=args.env_name, states_by_k=batch.states_by_k,
        goal_states=batch.goal_states, variations=(), img_size=args.img_size, seed=seed)
    z012 = p3.encode_frames(model=model, frames=frames, batch_size=args.batch_size, device=device)
    gframes = p3.render_state_sequence(
        env_name=args.env_name, states_by_k=batch.goal_states[:, None],
        goal_states=batch.goal_states, variations=(), img_size=args.img_size, seed=seed)
    g = p3.encode_frames(model=model, frames=gframes, batch_size=args.batch_size, device=device)[:, 0]
    return (z012[:, 0].astype(np.float32), g.astype(np.float32),
            z012[:, 1].astype(np.float32), z012[:, 2].astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--output", default="outputs/lghl_phase8f_anchor_data/anchor_triples.npz")
    ap.add_argument("--train-size", type=int, default=40000)
    ap.add_argument("--val-size", type=int, default=4000)
    ap.add_argument("--train-seed", type=int, default=2024)
    ap.add_argument("--val-seed", type=int, default=777)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()
    build.dataset = p3.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f"[8f] building train n={args.train_size}", flush=True)
    ztr, gtr, w1tr, w2tr = build(args, model, device, args.train_size, args.train_seed)
    print(f"[8f] building val n={args.val_size}", flush=True)
    zva, gva, w1va, w2va = build(args, model, device, args.val_size, args.val_seed)
    np.savez_compressed(out, z_train=ztr, g_train=gtr, w1_train=w1tr, w2_train=w2tr,
                        z_val=zva, g_val=gva, w1_val=w1va, w2_val=w2va)
    print(f"[8f] saved {out} ({time.time()-t0:.0f}s) shapes z_train={ztr.shape}")


if __name__ == "__main__":
    main()
