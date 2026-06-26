"""Contact-labeled eval set for Step B (regime MoE).

Same latent + action format as the disc_operator 40k training npz
(z: (N,K+1,192), a: (N,K,action_block*raw_adim)), PLUS a per-block contact
signal so we can measure whether the MoE gate aligns with contact unsupervised.

Reuses Step A replay (env contact) + Phase 3 windowing/encode.
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
import regime_existence_stepA as stepA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name",
                    default="/home/jovyan/.stable_worldmodel/pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/regime_stepB/eval_contact.npz")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    dataset = p3.swm.data.load_dataset(
        args.dataset_name,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = p3.build_window_batch(
        dataset, num_samples=args.num_samples, max_k=args.max_k,
        goal_offset=args.goal_offset, action_block=args.action_block, seed=args.seed)

    print("[stepB-data] replay for contact + frames", flush=True)
    frames, contact_max, contact_frac, _states, _term = stepA.replay_windows(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        action_block=args.action_block, img_size=args.img_size, seed=args.seed)

    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f"[stepB-data] encoding frames {frames.shape}", flush=True)
    z = p3.encode_frames(model=model, frames=frames,
                         batch_size=args.batch_size, device=device)

    np.savez_compressed(
        out,
        z=z.astype(np.float32),
        a=batch.model_actions.astype(np.float32),
        contact_frac=contact_frac.astype(np.float32),
        contact_max=contact_max.astype(np.float32),
    )
    print(f"[stepB-data] saved {out} z={z.shape} a={batch.model_actions.shape} "
          f"contact_rate={(contact_frac[:,1:]>0).mean():.3f} ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
