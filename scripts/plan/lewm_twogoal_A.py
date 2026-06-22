"""LeWM two-goal (A) — first cut: does a two-goal junction create SEPARATED
branches in LeWM's real latent? (existence check before building the proposer.)

For each sampled PushT start state we define two distinct goals (A, B). Using
LeWM's own dynamics + cost (random-shooting planning), we imagine the latent
future toward A and toward B:
  - sample S random action sequences, score each with model.get_cost toward goal,
    take the top-k best, read their imagined final latents (model's predicted_emb).
  - z_A = top-k futures toward A; z_B = top-k toward B.

Metric (LeWM analog of Stage 1c sep/within):
  sep      = ||mean(z_A) - mean(z_B)||           (cross-goal separation)
  within   = mean within-goal scatter of z_A,z_B (same-goal planning spread)
  sep/within >> 1  => the two goals map to genuinely separated latent branches
                      => a goal-agnostic continuous predictor would blur between
                      them; discrete commitment has something to commit to.
  sep/within ~ 1   => no exploitable multimodal structure in LeWM latent here.

Reuses latent_drift_phase3 for load/sample/render/encode. Writes outputs/lewm_twogoal_A/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402


@torch.inference_mode()
def imagine_toward_goal(model, start_frames, goal_frames, n_samples, horizon,
                        madim, topk, device, rng):
    """Random-shooting plan toward goal; return top-k imagined final latents.
    start_frames/goal_frames: (B,H,W,3) uint8. madim = action_block*action_dim
    (the model consumes actions in blocks). Returns (B, topk, D)."""
    dtype = next(model.parameters()).dtype
    B = start_frames.shape[0]
    # (B,1,C,H,W) single grounding frame, broadcast to S candidates
    px1 = p3.images_to_tensor(start_frames[:, None]).to(device=device, dtype=dtype)   # (B,1,C,H,W)
    gl1 = p3.images_to_tensor(goal_frames[:, None]).to(device=device, dtype=dtype)    # (B,1,C,H,W)
    pixels = px1[:, None].expand(B, n_samples, 1, *px1.shape[2:]).contiguous()        # (B,S,1,C,H,W)
    acts = torch.from_numpy(
        rng.uniform(-1, 1, size=(B, n_samples, horizon, madim)).astype(np.float32)
    ).to(device=device, dtype=dtype)
    # get_cost does goal={k:v[:,0]} then encodes -> needs goal (B,1,1,C,H,W) so the
    # [:,0] strip leaves a 5D (B,T=1,C,H,W) for encode.
    info = {"pixels": pixels, "goal": gl1[:, None], "action": acts}  # get_cost pops 'action'
    cost = model.get_cost(info, acts)                       # (B,S)
    pred = info["predicted_emb"]                            # (B,S,T+1,D)
    final = pred[:, :, -1, :].float()                       # (B,S,D)
    order = torch.argsort(cost, dim=1)[:, :topk]            # best topk per item
    sel = torch.gather(final, 1, order[..., None].expand(B, topk, final.shape[-1]))
    return sel.cpu().numpy()                                # (B,topk,D)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--output-dir", default="outputs/lewm_twogoal_A")
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    p3.configure_torch_threads_from_env()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    dataset = p3.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True

    # sample starts + their real goals; goalB = shuffled (a different valid goal)
    batch = p3.build_window_batch(
        dataset, num_samples=args.n_starts, max_k=2, goal_offset=args.goal_offset,
        action_block=args.action_block, seed=args.seed)
    starts = batch.states_by_k[:, 0]                       # (N,7)
    goalA = batch.goal_states                              # (N,7)
    perm = rng.permutation(len(goalA))
    goalB = goalA[perm]                                    # different goal per start
    action_dim = int(np.asarray(dataset.get_col_data("action")).reshape(len(
        np.asarray(dataset.get_col_data("state"))), -1).shape[1])

    sf = p3.render_state_sequence(env_name=args.env_name, states_by_k=starts[:, None],
                                  goal_states=goalA, variations=(), img_size=args.img_size, seed=args.seed)[:, 0]
    gAf = p3.render_state_sequence(env_name=args.env_name, states_by_k=goalA[:, None],
                                   goal_states=goalA, variations=(), img_size=args.img_size, seed=args.seed)[:, 0]
    gBf = p3.render_state_sequence(env_name=args.env_name, states_by_k=goalB[:, None],
                                   goal_states=goalB, variations=(), img_size=args.img_size, seed=args.seed)[:, 0]
    madim = args.action_block * action_dim                 # model consumes action blocks
    print(f"[2gA] starts={len(sf)} action_dim={action_dim} madim={madim} S={args.n_samples} H={args.horizon}", flush=True)

    zA = imagine_toward_goal(model, sf, gAf, args.n_samples, args.horizon, madim, args.topk, device, rng)
    zB = imagine_toward_goal(model, sf, gBf, args.n_samples, args.horizon, madim, args.topk, device, rng)
    # also normalize latents (z-score over all) so distances are comparable
    allz = np.concatenate([zA.reshape(-1, zA.shape[-1]), zB.reshape(-1, zB.shape[-1])], 0)
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    zA = (zA - mu) / sd; zB = (zB - mu) / sd

    seps, withins, betweens = [], [], []
    for i in range(len(sf)):
        a, b = zA[i], zB[i]                                # (topk, D)
        ma, mb = a.mean(0), b.mean(0)
        sep = np.linalg.norm(ma - mb)
        within = 0.5 * (np.linalg.norm(a - ma, axis=1).mean() + np.linalg.norm(b - mb, axis=1).mean())
        mid = 0.5 * (ma + mb)
        between = min(np.linalg.norm(mid - ma), np.linalg.norm(mid - mb)) / (sep / 2 + 1e-9)
        seps.append(sep); withins.append(within); betweens.append(between)
    seps, withins = np.array(seps), np.array(withins)
    ratio = seps / (withins + 1e-9)

    R = {
        "n_starts": int(len(sf)), "n_samples": args.n_samples, "horizon": args.horizon,
        "topk": args.topk, "action_dim": action_dim,
        "mean_sep": float(seps.mean()), "mean_within": float(withins.mean()),
        "median_sep_over_within": float(np.median(ratio)),
        "mean_sep_over_within": float(ratio.mean()),
        "frac_separated(ratio>2)": float((ratio > 2).mean()),
        "mean_midpoint_betweenness": float(np.mean(betweens)),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "twogoal_A_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== LeWM two-goal (A): separated branches in real latent? ===",
        f"starts={R['n_starts']} S={R['n_samples']} H={R['horizon']} topk={R['topk']} ({R['elapsed_sec']}s)",
        "",
        f"sep (cross-goal)            : {R['mean_sep']:.3f}",
        f"within (same-goal scatter)  : {R['mean_within']:.3f}",
        f"sep/within  median / mean   : {R['median_sep_over_within']:.2f} / {R['mean_sep_over_within']:.2f}",
        f"frac starts separated (>2)  : {R['frac_separated(ratio>2)']:.2f}",
        f"midpoint between-ness       : {R['mean_midpoint_betweenness']:.2f}  (~1 = midpoint sits between branches)",
        "",
        "read: sep/within >> 1 => two goals map to separated LeWM-latent branches",
        "  (goal-agnostic continuous predictor would blur to the between-state).",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[2gA] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
