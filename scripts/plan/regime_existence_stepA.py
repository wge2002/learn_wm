"""Step A (existence) for the dynamic-discrete-regime direction.

Direction doc: docs/knowledge/direction_discrete_regime_from_lewm.md

Question (pure analysis, no training): does a discrete *regime* structure live in
the ALREADY-TRAINED LeWM transition f, even though SIGReg forces the marginal p(z)
to be structureless? Theory says discreteness cannot live in the state z (isotropic
Gaussian by construction) but CAN live in the transition f(z,a), because contact
physics is piecewise (free move / contact / push / release).

Method:
  1. Replay PushT expert windows in the env (ID condition), capturing per-block
     frames, the contact signal (n_contact_points), and state.
  2. Encode frames -> latent z along the genuine expert trajectory.
  3. For each interior step, read f's LOCAL BEHAVIOR with the trained predictor:
       - residual direction  d_t = f(z_hist, a_t) - z_t      (cheap, "how it moves")
       - local Jacobian      J_t = d f / d z_t (last frame)   (the local operator)
  4. Cluster three feature families and compare:
       - z_t            (CONTROL; expect no structure under SIGReg)
       - residual d_t   (dynamics)
       - Jacobian J_t   (local operator)
     Measure structure (silhouette) and alignment to contact (NMI / ARI / purity).

Go/kill (from the doc):
  f forms a few clusters that align with contact while z does not  -> existence holds
  (green). f is also structureless -> direction shrinks / honest negative.

Output: <out>/summary.json, <out>/summary.csv, optional <out>/stepA_clusters.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402  reuse model/data/encode helpers


# --------------------------------------------------------------------------- #
# replay: frames + contact + state, all from one stepped expert replay
# --------------------------------------------------------------------------- #
def replay_windows(
    *,
    env_name: str,
    init_states: np.ndarray,
    goal_states: np.ndarray,
    raw_actions: np.ndarray,
    action_block: int,
    img_size: int,
    seed: int,
):
    """Step expert actions; capture per-block frame, contact (max/mean over the
    block's env steps), and state. Returns
        frames        (N, K+1, H, W, 3) uint8
        contact_max   (N, K+1) float   max n_contact_points in the block ending here
        contact_frac  (N, K+1) float   fraction of env steps with contact>0
        states        (N, K+1, 7)
        terminated    (N, K+1) bool
    """
    n, max_env_steps, _ = raw_actions.shape
    max_k = max_env_steps // action_block
    frames = np.empty((n, max_k + 1, img_size, img_size, 3), dtype=np.uint8)
    contact_max = np.zeros((n, max_k + 1), dtype=np.float32)
    contact_frac = np.zeros((n, max_k + 1), dtype=np.float32)
    states = np.empty((n, max_k + 1, 7), dtype=np.float32)
    terminated = np.zeros((n, max_k + 1), dtype=bool)

    env = gym.make(
        env_name,
        max_episode_steps=max_env_steps + 5,
        render_mode="rgb_array",
        resolution=img_size,
    )
    try:
        for i in range(n):
            env.reset(seed=seed + i)
            p3.set_state_and_goal(env, init_states[i], goal_states[i])
            frames[i, 0] = env.render()
            states[i, 0] = p3.get_env_state(env)
            done_seen = False
            blk_contacts: list[float] = []
            for t in range(max_env_steps):
                _, _, done, truncated, _ = env.step(raw_actions[i, t])
                done_seen = done_seen or bool(done or truncated)
                ncp = float(getattr(env.unwrapped, "n_contact_points", 0))
                blk_contacts.append(ncp)
                if (t + 1) % action_block == 0:
                    k = (t + 1) // action_block
                    frames[i, k] = env.render()
                    states[i, k] = p3.get_env_state(env)
                    terminated[i, k] = done_seen
                    arr = np.asarray(blk_contacts)
                    contact_max[i, k] = float(arr.max())
                    contact_frac[i, k] = float((arr > 0).mean())
                    blk_contacts = []
    finally:
        env.close()
    return frames, contact_max, contact_frac, states, terminated


# --------------------------------------------------------------------------- #
# predictor local behavior
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def predictor_residual(model, emb_win, act_emb_win):
    """emb_win (B, HS, D), act_emb_win (B, HS, A) -> next-z pred (B, D)."""
    return model.predict(emb_win, act_emb_win)[:, -1]


def jacobian_features(model, emb_win, act_emb_win, topm, device):
    """Local Jacobian d f / d z_t (last history frame), summarized by its singular
    value spectrum (top-m, scale-invariant operator signature). emb_win (B,HS,D)."""
    from torch.func import jacrev, vmap

    B, HS, D = emb_win.shape
    fixed = emb_win[:, :-1].contiguous()  # (B, HS-1, D) context, held fixed
    last = emb_win[:, -1].contiguous()    # (B, D) the variable

    def single(z_last, ctx, a_win):
        win = torch.cat([ctx, z_last[None]], dim=0)[None]  # (1, HS, D)
        return model.predict(win, a_win[None])[0, -1]      # (D,)

    jac = vmap(jacrev(single), in_dims=(0, 0, 0))(last, fixed, act_emb_win)  # (B,D,D)
    svals = torch.linalg.svdvals(jac.float())  # (B, D) sorted desc
    return svals[:, :topm].cpu().numpy()


# --------------------------------------------------------------------------- #
# clustering analysis
# --------------------------------------------------------------------------- #
def analyze_feature(name, X, contact_label, k_list, pca_dim, seed):
    """Standardize -> PCA -> KMeans over k_list. Returns per-k structure +
    contact-alignment metrics and the best-silhouette pick."""
    Xs = StandardScaler().fit_transform(X)
    if pca_dim and pca_dim < Xs.shape[1]:
        Xs = PCA(n_components=pca_dim, random_state=seed).fit_transform(Xs)
    rows = []
    best = None
    for k in k_list:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xs)
        lab = km.labels_
        # subsample silhouette for speed if large
        if len(Xs) > 5000:
            ridx = np.random.default_rng(seed).choice(len(Xs), 5000, replace=False)
            sil = float(silhouette_score(Xs[ridx], lab[ridx]))
        else:
            sil = float(silhouette_score(Xs, lab))
        nmi = float(normalized_mutual_info_score(contact_label, lab))
        ari = float(adjusted_rand_score(contact_label, lab))
        # purity: each cluster -> majority contact class
        purity = 0.0
        for c in np.unique(lab):
            m = lab == c
            vals, cnts = np.unique(contact_label[m], return_counts=True)
            purity += cnts.max()
        purity = float(purity / len(lab))
        # contact rate per cluster (spread => clusters separate contact)
        crate = [float(contact_label[lab == c].mean()) for c in np.unique(lab)]
        row = {
            "feature": name, "k": int(k), "silhouette": sil,
            "contact_nmi": nmi, "contact_ari": ari, "contact_purity": purity,
            "cluster_contact_rate_min": float(min(crate)),
            "cluster_contact_rate_max": float(max(crate)),
        }
        rows.append(row)
        if best is None or sil > best["silhouette"]:
            best = row
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name",
                    default="/home/jovyan/.stable_worldmodel/pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--num-samples", type=int, default=400, help="expert windows")
    ap.add_argument("--max-k", type=int, default=10, help="model steps per window")
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128, help="encode batch")
    ap.add_argument("--jac-batch", type=int, default=64, help="jacobian vmap batch")
    ap.add_argument("--max-jac-samples", type=int, default=2000)
    ap.add_argument("--jac-topm", type=int, default=32)
    ap.add_argument("--no-jacobian", action="store_true")
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--k-list", default="2,3,4,5,6,8")
    ap.add_argument("--contact-thresh", type=float, default=0.0,
                    help="block is 'contact' if contact_frac > thresh")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="outputs/regime_stepA")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    k_list = [int(x) for x in args.k_list.split(",") if x]
    t0 = time.time()

    print("[stepA] loading dataset", flush=True)
    dataset = p3.swm.data.load_dataset(
        args.dataset_name,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"],
    )
    batch = p3.build_window_batch(
        dataset,
        num_samples=args.num_samples,
        max_k=args.max_k,
        goal_offset=args.goal_offset,
        action_block=args.action_block,
        seed=args.seed,
    )

    print("[stepA] replay (frames + contact + state)", flush=True)
    frames, contact_max, contact_frac, _states, _term = replay_windows(
        env_name=args.env_name,
        init_states=batch.init_states,
        goal_states=batch.goal_states,
        raw_actions=batch.raw_actions,
        action_block=args.action_block,
        img_size=args.img_size,
        seed=args.seed,
    )

    print("[stepA] loading model", flush=True)
    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    HS = int(getattr(model.predictor, "num_frames", 3))

    print(f"[stepA] encoding frames {frames.shape}", flush=True)
    z = p3.encode_frames(model=model, frames=frames,
                         batch_size=args.batch_size, device=device)  # (N,K+1,D)
    N, K1, D = z.shape
    print(f"[stepA] z={z.shape} HS={HS}", flush=True)

    # action embeddings per model step: model_actions (N, max_k, A_raw)
    macts = torch.from_numpy(batch.model_actions).to(device)
    with torch.inference_mode():
        act_emb = model.action_encoder(macts)  # (N, max_k, A_emb)
    z_t = torch.from_numpy(z).to(device)

    # ---- build interior-step records: predict z_{k+1} from history ending at k ---
    z_feat, resid_feat, contact_lab = [], [], []
    for n in range(N):
        for k in range(HS - 1, args.max_k):  # need HS history frames; predict k+1
            lo = k - HS + 1
            emb_win = z_t[n, lo:k + 1]              # (HS, D)
            a_win = act_emb[n, lo:k + 1]            # (HS, A_emb)
            z_feat.append(z[n, k])
            # contact during the predicted block (k+1)
            contact_lab.append(int(contact_frac[n, k + 1] > args.contact_thresh))
            resid_feat.append((emb_win, a_win, z_t[n, k]))

    contact_label = np.asarray(contact_lab)
    z_feat = np.asarray(z_feat, dtype=np.float32)
    M = len(z_feat)
    print(f"[stepA] interior records M={M} contact_rate={contact_label.mean():.3f}",
          flush=True)

    # residual in batches
    print("[stepA] computing residual directions", flush=True)
    resid = np.empty((M, D), dtype=np.float32)
    bs = 1024
    for s in range(0, M, bs):
        e = min(s + bs, M)
        ew = torch.stack([resid_feat[j][0] for j in range(s, e)], dim=0)  # (b,HS,D)
        aw = torch.stack([resid_feat[j][1] for j in range(s, e)], dim=0)
        zc = torch.stack([resid_feat[j][2] for j in range(s, e)], dim=0)  # (b,D)
        pred = predictor_residual(model, ew, aw)
        resid[s:e] = (pred - zc).float().cpu().numpy()
    # direction-normalize residual (regime = direction of motion, not action scale)
    resid_dir = resid / (np.linalg.norm(resid, axis=1, keepdims=True) + 1e-8)

    # jacobian features on a subsample
    jac_svals = None
    if not args.no_jacobian:
        sub = np.arange(M)
        if M > args.max_jac_samples:
            sub = np.random.default_rng(args.seed).choice(
                M, args.max_jac_samples, replace=False)
        print(f"[stepA] computing Jacobian SV spectra on {len(sub)} samples",
              flush=True)
        feats = []
        for s in range(0, len(sub), args.jac_batch):
            idx = sub[s:s + args.jac_batch]
            ew = torch.stack([resid_feat[j][0] for j in idx], dim=0)  # (b,HS,D)
            aw = torch.stack([resid_feat[j][1] for j in idx], dim=0)
            feats.append(jacobian_features(model, ew, aw, args.jac_topm, device))
        jac_svals = np.concatenate(feats, axis=0)
        jac_sub_idx = sub

    # ---- clustering comparison ----
    print("[stepA] clustering: z (control) vs residual vs jacobian", flush=True)
    all_rows = []
    bests = {}
    rows, best = analyze_feature("z_control", z_feat, contact_label,
                                 k_list, args.pca_dim, args.seed)
    all_rows += rows; bests["z_control"] = best
    rows, best = analyze_feature("residual_dir", resid_dir, contact_label,
                                 k_list, args.pca_dim, args.seed)
    all_rows += rows; bests["residual_dir"] = best
    if jac_svals is not None:
        rows, best = analyze_feature("jacobian_svals", jac_svals,
                                     contact_label[jac_sub_idx],
                                     k_list, min(args.pca_dim, args.jac_topm),
                                     args.seed)
        all_rows += rows; bests["jacobian_svals"] = best

    # raw arrays for later curated figures (not committed to Git by default)
    np.savez_compressed(
        out / "stepA_features.npz",
        z_feat=z_feat, resid_dir=resid_dir, contact_label=contact_label,
        jac_svals=jac_svals if jac_svals is not None else np.empty(0),
        jac_sub_idx=jac_sub_idx if jac_svals is not None else np.empty(0),
    )

    # ---- verdict heuristic ----
    zc = bests["z_control"]
    def beats(b):
        return (b["silhouette"] > zc["silhouette"] + 0.05 and
                b["contact_nmi"] > zc["contact_nmi"] + 0.02)
    f_keys = [k for k in ("residual_dir", "jacobian_svals") if k in bests]
    green = any(beats(bests[k]) for k in f_keys)
    verdict = "GREEN: regime structure in f, aligned to contact, beyond z" if green \
        else "RED/AMBER: f not clearly more structured/contact-aligned than z"

    summary = {
        "direction": "direction_discrete_regime_from_lewm.md / Step A existence",
        "policy": args.policy,
        "config": {
            "num_samples": args.num_samples, "max_k": args.max_k,
            "action_block": args.action_block, "HS": HS, "D": int(D),
            "M_records": int(M), "contact_rate": float(contact_label.mean()),
            "contact_thresh": args.contact_thresh, "k_list": k_list,
            "pca_dim": args.pca_dim, "jac_topm": args.jac_topm,
            "max_jac_samples": args.max_jac_samples,
            "jacobian": not args.no_jacobian,
        },
        "best_by_feature": bests,
        "verdict": verdict,
        "verdict_green": bool(green),
        "elapsed_sec": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    print("\n========== Step A summary ==========", flush=True)
    for kf, b in bests.items():
        print(f"  {kf:16s} best k={b['k']} sil={b['silhouette']:+.3f} "
              f"contact_nmi={b['contact_nmi']:.3f} purity={b['contact_purity']:.3f} "
              f"crate[{b['cluster_contact_rate_min']:.2f},"
              f"{b['cluster_contact_rate_max']:.2f}]", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"  wrote {out}/summary.json ({summary['elapsed_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
