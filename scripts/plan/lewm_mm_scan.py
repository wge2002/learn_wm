"""LeWM domain multimodality scan (step B): does the EXISTING PushT expert data
contain natural multimodality -- states where, for ~the same (current state, goal),
the future diverges into two+ separated routes?

This decides strategy: if PushT demos are richly multimodal, we can build the
commitment test from real data; if near-unimodal (as the diagnosis suggested),
the two-goal construction (A) is the only path -- itself an important finding.

Method (raw state space, cheap -- no rendering/encoding):
  - per frame: feature = (current state, episode-goal=final block pose), future =
    state H steps ahead (within the same episode).
  - for sampled anchors, find cross-episode neighbors with SMALL (state,goal)
    distance (~same situation), collect their futures, and test whether the
    futures form >=2 separated clusters (2-GMM BIC vs 1, and spread ratio).
  - report: fraction of anchors that are genuine multimodal junctions, and how
    separated the branches are.

Writes outputs/lewm_mm_scan/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_worldmodel as swm  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--output-dir", default="outputs/lewm_mm_scan")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--horizon", type=int, default=15, help="future lookahead (steps)")
    ap.add_argument("--n-anchors", type=int, default=1500)
    ap.add_argument("--knn", type=int, default=40)
    ap.add_argument("--sg-radius-pct", type=float, default=5.0,
                    help="keep neighbors within this percentile of (state,goal) dist")
    ap.add_argument("--goal-cols", default="", help="state dims used as goal (e.g. 2,3,4=block pose); empty=full")
    ap.add_argument("--fut-cols", default="", help="state dims for future-divergence; empty=full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ds = swm.data.load_dataset(args.dataset_name, cache_dir=args.cache_dir,
                               keys_to_load=["state", "episode_idx", "step_idx"],
                               keys_to_cache=["state"])
    cols = ds.column_names
    ep_col = "episode_idx" if "episode_idx" in cols else "ep_idx"
    st_col = "step_idx" if "step_idx" in cols else "step"
    state = np.asarray(ds.get_col_data("state"), np.float32)
    ep = np.asarray(ds.get_col_data(ep_col)).reshape(-1)
    step = np.asarray(ds.get_col_data(st_col)).reshape(-1)
    state = state.reshape(len(ep), -1)
    D = state.shape[1]
    print(f"[mm] frames={len(state)} state_dim={D} episodes={len(np.unique(ep))}", flush=True)

    # per-episode bookkeeping: index ranges + goal = final state of the episode
    order = np.lexsort((step, ep))
    state, ep, step = state[order], ep[order], step[order]
    uniq, starts = np.unique(ep, return_index=True)
    ep_end = np.append(starts[1:], len(ep))                     # exclusive end per episode
    goal_of_ep = {int(uniq[i]): state[ep_end[i] - 1].copy() for i in range(len(uniq))}
    goal = np.stack([goal_of_ep[int(e)] for e in ep])
    ep_end_of_frame = np.empty(len(ep), np.int64)
    for i in range(len(uniq)):
        ep_end_of_frame[starts[i]:ep_end[i]] = ep_end[i]

    # future = state H ahead within same episode (skip frames too close to the end)
    fut_idx = np.arange(len(ep)) + args.horizon
    valid = fut_idx < ep_end_of_frame
    future = np.full_like(state, np.nan)
    future[valid] = state[fut_idx[valid]]

    # normalize state & goal dims for distance
    mu, sd = state.mean(0), state.std(0) + 1e-6
    gcols = [int(x) for x in args.goal_cols.split(",")] if args.goal_cols else list(range(D))
    fcols = [int(x) for x in args.fut_cols.split(",")] if args.fut_cols else list(range(D))
    ns = (state - mu) / sd; ng = ((goal - mu) / sd)[:, gcols]
    feat = np.concatenate([ns, ng], 1)                          # (full state, goal[gcols])
    nf = ((future - mu) / sd)[:, fcols]                         # future divergence on fcols
    print(f"[mm] goal_cols={gcols} fut_cols={fcols}", flush=True)

    from sklearn.neighbors import NearestNeighbors
    from sklearn.mixture import GaussianMixture

    pool = np.where(valid)[0]
    anchors = rng.choice(pool, size=min(args.n_anchors, len(pool)), replace=False)
    nn = NearestNeighbors(n_neighbors=args.knn + 1).fit(feat[pool])

    # radius = percentile of (state,goal) NN distances -> "same situation"
    dist_a, _ = nn.kneighbors(feat[anchors])
    radius = np.percentile(dist_a[:, 1:], args.sg_radius_pct)
    print(f"[mm] H={args.horizon} knn={args.knn} sg-radius(p{args.sg_radius_pct})={radius:.3f}", flush=True)

    n_mm, bimodal_flags, sep_ratios, n_used = 0, [], [], 0
    examples = []
    for a in anchors:
        d, j = nn.kneighbors(feat[a][None], n_neighbors=args.knn + 1)
        d, j = d[0], j[0]
        keep = pool[j[(d <= radius) & (pool[j] != -1)]]
        keep = keep[ep[keep] != ep[a]]                          # cross-episode only
        keep = np.append(keep, a)
        if len(keep) < 8:
            continue
        n_used += 1
        F = nf[keep]                                            # their futures (normalized)
        F = F[~np.isnan(F).any(1)]
        if len(F) < 8:
            continue
        # spread: how far apart are the futures vs within-cluster scatter?
        F = F + rng.normal(0, 1e-4, F.shape).astype(np.float32)   # break exact duplicates
        try:
            g1 = GaussianMixture(1, covariance_type="diag", reg_covar=1e-3, random_state=0).fit(F)
            g2 = GaussianMixture(2, covariance_type="diag", reg_covar=1e-3, random_state=0).fit(F)
            is_bi = g2.bic(F) < g1.bic(F)
        except ValueError:
            continue
        if is_bi:
            c = g2.means_; lab = g2.predict(F)
            inter = np.linalg.norm(c[0] - c[1])
            intra = np.mean([np.linalg.norm(F[lab == b] - c[b], axis=1).mean()
                             for b in (0, 1) if (lab == b).sum() > 1] or [1e-9])
            ratio = inter / (intra + 1e-9)
        else:
            ratio = 0.0
        bimodal_flags.append(is_bi); sep_ratios.append(ratio)
        if is_bi and ratio > 2.0:
            n_mm += 1
            if len(examples) < 5:
                examples.append({"anchor_ep": int(ep[a]), "step": int(step[a]),
                                 "n_neighbors": int(len(keep)), "sep_ratio": float(ratio)})

    R = {
        "n_frames": int(len(state)), "state_dim": int(D),
        "horizon": args.horizon, "knn": args.knn, "sg_radius": float(radius),
        "anchors_used": int(n_used),
        "frac_bimodal_BIC": float(np.mean(bimodal_flags)) if bimodal_flags else 0.0,
        "frac_strong_multimodal(sep>2)": float(n_mm / max(n_used, 1)),
        "median_sep_ratio": float(np.median(sep_ratios)) if sep_ratios else 0.0,
        "p90_sep_ratio": float(np.percentile(sep_ratios, 90)) if sep_ratios else 0.0,
        "examples": examples,
    }
    (out / "mm_scan_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== LeWM/PushT expert-data multimodality scan (raw state) ===",
        f"frames={R['n_frames']} state_dim={R['state_dim']} anchors_used={R['anchors_used']} "
        f"H={R['horizon']} sg_radius={R['sg_radius']:.3f}",
        "",
        f"fraction of junctions BIC-bimodal           : {R['frac_bimodal_BIC']:.2f}",
        f"fraction STRONG multimodal (sep/within>2)   : {R['frac_strong_multimodal(sep>2)']:.2f}",
        f"median sep/within ratio                     : {R['median_sep_ratio']:.2f}",
        f"p90 sep/within ratio                        : {R['p90_sep_ratio']:.2f}",
        "",
        "read: high strong-multimodal fraction => PushT demos ARE naturally multimodal",
        "  (build commitment test from real data). near-zero => domain is ~unimodal,",
        "  the two-goal construction (A) is the only path (itself a key finding).",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[mm] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
