"""Direction B veto gate: does PushT have modelable stochasticity, and does it
live along the error-amplification directions?

Generative-LeWM (Direction B) posits z' = f(z,a) + Sigma(z)^{1/2} eps with an
innovation whose covariance is shaped by the amplification law. That is only a
model, not a prior-on-noise, if (1) the true transition has nonzero conditional
variance given (z,a), and (2) that variance is anisotropic and aligns with the
top error-amplification directions (else there is nothing structured to model).

We approximate the conditional variance by grouping near-duplicate physical
state/action conditions from the probe rollouts and measuring the spread of
their next latents.  The strict audit uses *absolute* radii in standardized
condition space.  A percentile cut is deliberately not supported: it always
selects some groups, even when none of them are genuinely close, and therefore
confounds deterministic local drift with conditional variance.

Kill Direction B if median conditional std / latent scale < 0.02 (effectively
deterministic), the conditional covariance is near-isotropic (top/mean
eigenvalue ratio < 2, nothing anisotropic to allocate), OR the fraction of
conditional variance inside the top amplification span is less than twice its
random-subspace baseline.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_worldmodel as swm  # noqa: E402

HS = 3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--num', type=int, default=4000)
    ap.add_argument(
        '--knn',
        type=int,
        default=12,
        help='number of non-self neighbors required inside every radius',
    )
    ap.add_argument(
        '--radius',
        type=float,
        nargs='+',
        required=True,
        help=(
            'one or more absolute kNN-radius thresholds in globally '
            'standardized condition space'
        ),
    )
    ap.add_argument(
        '--key-mode',
        choices=('current', 'history'),
        default='current',
        help=(
            'current=(s_t,a_t), matching the smoke test; history conditions on '
            'the full model history and is the stricter hidden-velocity audit'
        ),
    )
    ap.add_argument(
        '--step',
        type=int,
        default=2,
        help='prediction step after the initial history at which to form tuples',
    )
    ap.add_argument('--amp-rank', type=int, default=3)
    ap.add_argument('--amp-num', type=int, default=64)
    ap.add_argument(
        '--min-groups',
        type=int,
        default=25,
        help='minimum anchor groups required to issue a PASS/KILL verdict',
    )
    ap.add_argument(
        '--keep-duplicate-trajectories',
        action='store_true',
        help=(
            'retain exact duplicate sampled windows; strict audits remove them '
            'by default because the probe sampler draws with replacement'
        ),
    )
    ap.add_argument(
        '--disjoint-groups',
        action='store_true',
        help=(
            'greedily keep non-overlapping neighbor groups, ordered from '
            'smallest to largest kNN radius, to avoid pseudo-replication'
        ),
    )
    ap.add_argument('--out', required=True)
    return ap.parse_args()


def condition_key(states, actions, t, mode):
    if mode == 'current':
        return np.concatenate([states[:, t], actions[:, t]], axis=1)
    start = t - HS + 1
    return np.concatenate(
        [
            states[:, start : t + 1].reshape(len(states), -1),
            actions[:, start : t + 1].reshape(len(actions), -1),
        ],
        axis=1,
    )


def covariance_for_radius(znext, idx, kth_radius, radius, disjoint):
    candidates = np.flatnonzero(kth_radius <= radius)
    if disjoint:
        used = set()
        selected = []
        for anchor in candidates[np.argsort(kth_radius[candidates])]:
            members = idx[anchor]
            if any(int(member) in used for member in members):
                continue
            selected.append(anchor)
            used.update(int(member) for member in members)
        anchors = np.asarray(selected, dtype=np.int64)
    else:
        anchors = candidates
    if len(anchors) == 0:
        return candidates, anchors, None, np.empty(0, dtype=np.float64)

    cov_sum = np.zeros((znext.shape[1], znext.shape[1]), dtype=np.float64)
    cond_stds = np.empty(len(anchors), dtype=np.float64)
    for j, anchor in enumerate(anchors):
        grp = znext[idx[anchor]].astype(np.float64, copy=False)
        centered = grp - grp.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(1, len(grp) - 1)
        cov_sum += cov
        cond_stds[j] = np.sqrt(np.trace(cov))
    return candidates, anchors, cov_sum / len(anchors), cond_stds


def amplification_bases(model, z, a, t, rank, num, dev):
    count = min(num, len(z))
    zt = torch.from_numpy(z[:count, t - HS + 1 : t + 1]).to(dev)
    at = torch.from_numpy(a[:count, t - HS + 1 : t + 1]).to(dev)
    aet = model.action_encoder(at)
    bases = []
    for i in range(count):
        zl = zt[i, -1].detach().requires_grad_(True)

        def g(x):
            context = torch.cat([zt[i, :-1], x.unsqueeze(0)], 0).unsqueeze(0)
            return model.predict(context, aet[i].unsqueeze(0))[0, -1]

        jac = torch.autograd.functional.jacobian(g, zl, vectorize=True)
        u, _, _ = torch.linalg.svd(jac, full_matrices=False)
        bases.append(u[:, :rank].detach().cpu().numpy())
    return bases


def summarize_radius(
    radius,
    num_candidates,
    anchors,
    mean_cov,
    cond_stds,
    kth_radius,
    znext_scale,
    amp_bases,
    amp_rank,
    latent_dim,
    min_groups,
):
    base = {
        'radius': float(radius),
        'n_candidate_groups': int(num_candidates),
        'n_groups': int(len(anchors)),
        'status': 'ok' if len(anchors) >= min_groups else 'insufficient_groups',
    }
    if mean_cov is None or len(anchors) < min_groups:
        base.update(
            {
                'cond_std_med': None,
                'cond_std_ratio': None,
                'cond_cov_anisotropy': None,
                'condvar_in_amp_span_med': None,
                'amp_alignment_enrichment': None,
                'verdict_gate': 'INCONCLUSIVE',
                'reason': f'fewer than {min_groups} strict-radius groups',
            }
        )
        return base

    eigvals, eigvecs = np.linalg.eigh(mean_cov)
    eigvals = eigvals[::-1].clip(min=0)
    top_condvec = eigvecs[:, -1]
    trace = float(np.trace(mean_cov))
    cond_std = float(np.median(cond_stds))
    ratio = cond_std / (znext_scale + 1e-12)
    anisotropy = float(eigvals[0] / (eigvals.mean() + 1e-12))

    variance_fractions = []
    leading_overlaps = []
    for basis in amp_bases:
        variance_fractions.append(
            float(np.trace(basis.T @ mean_cov @ basis) / (trace + 1e-12))
        )
        leading_overlaps.append(float(np.linalg.norm(basis.T @ top_condvec) ** 2))
    variance_fraction = float(np.median(variance_fractions))
    leading_overlap = float(np.median(leading_overlaps))
    random_baseline = amp_rank / latent_dim
    enrichment = variance_fraction / random_baseline

    if ratio < 0.02:
        verdict, reason = 'KILL', 'effectively deterministic'
    elif anisotropy < 2.0:
        verdict, reason = 'KILL', 'conditional covariance is near-isotropic'
    elif enrichment < 2.0:
        verdict, reason = 'KILL', 'variance is not enriched in amplification span'
    else:
        verdict, reason = 'PASS', 'structured stochasticity'

    base.update(
        {
            'kth_radius_med_selected': float(np.median(kth_radius[anchors])),
            'kth_radius_max_selected': float(np.max(kth_radius[anchors])),
            'cond_std_med': cond_std,
            'cond_std_p25': float(np.percentile(cond_stds, 25)),
            'cond_std_p75': float(np.percentile(cond_stds, 75)),
            'znext_scale': znext_scale,
            'cond_std_ratio': ratio,
            'cond_cov_anisotropy': anisotropy,
            'condvar_in_amp_span_med': variance_fraction,
            'leading_condvec_in_amp_span_med': leading_overlap,
            'random_subspace_baseline': random_baseline,
            'amp_alignment_enrichment': enrichment,
            'verdict_gate': verdict,
            'reason': reason,
        }
    )
    return base


def main():
    args = parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = swm.wm.utils.load_pretrained(args.policy).to(dev).eval()
    model.requires_grad_(False)
    with np.load(args.data) as data:
        z = data['z'][: args.num].astype(np.float32)  # (N,L,D)
        a = data['a'][: args.num].astype(np.float32)  # (N,L,Araw)
        s = data['states'][: args.num].astype(np.float32)  # (N,L,S)
    num_input = len(z)
    if not args.keep_duplicate_trajectories:
        trajectory_key = np.concatenate(
            [s.reshape(len(s), -1), a.reshape(len(a), -1)], axis=1
        )
        _, unique_idx = np.unique(
            trajectory_key, axis=0, return_index=True
        )
        unique_idx.sort()
        z, a, s = z[unique_idx], a[unique_idx], s[unique_idx]
    N, L, D = z.shape
    if not 0 < args.knn < N:
        raise ValueError(f'--knn must be in [1, {N - 1}], got {args.knn}')
    if args.amp_rank < 1 or args.amp_rank > D:
        raise ValueError(f'--amp-rank must be in [1, {D}], got {args.amp_rank}')
    if args.amp_num < 1:
        raise ValueError(f'--amp-num must be positive, got {args.amp_num}')
    if args.min_groups < 1:
        raise ValueError(
            f'--min-groups must be positive, got {args.min_groups}'
        )
    if any(radius <= 0 for radius in args.radius):
        raise ValueError(f'all --radius values must be positive: {args.radius}')

    # transition tuples at a mid frame: (true_state, action) -> next latent
    t = HS - 1 + args.step
    if t - HS + 1 < 0 or t + 1 >= L:
        raise ValueError(
            f'--step={args.step} selects t={t}, but trajectory length is {L}'
        )
    znext = z[:, t + 1]
    key_raw = condition_key(s, a, t, args.key_mode)
    key_mean = key_raw.mean(axis=0)
    key_std = key_raw.std(axis=0)
    active = key_std > 1e-6
    if not np.any(active):
        raise ValueError('all condition-key dimensions are constant')
    key = (key_raw[:, active] - key_mean[active]) / key_std[active]

    # group by nearest neighbors in (state,action): spread of next latent = cond var
    from scipy.spatial import cKDTree

    tree = cKDTree(key)
    dists, idx = tree.query(key, k=args.knn + 1)  # includes self
    kth_radius = dists[:, -1]
    znext_scale = float(
        np.linalg.norm(znext - znext.mean(axis=0), axis=1).mean()
    )
    amp_bases = amplification_bases(
        model,
        z,
        a,
        t,
        args.amp_rank,
        args.amp_num,
        dev,
    )

    results = []
    for radius in sorted(set(args.radius)):
        candidates, anchors, mean_cov, cond_stds = covariance_for_radius(
            znext, idx, kth_radius, radius, args.disjoint_groups
        )
        results.append(
            summarize_radius(
                radius,
                len(candidates),
                anchors,
                mean_cov,
                cond_stds,
                kth_radius,
                znext_scale,
                amp_bases,
                args.amp_rank,
                D,
                args.min_groups,
            )
        )

    res = {
        'policy': args.policy,
        'data': args.data,
        'num_input_transitions': num_input,
        'num_transitions': N,
        'duplicate_trajectories_removed': num_input - N,
        'trajectory_length': L,
        'latent_dim': D,
        'key_mode': args.key_mode,
        'key_dim_raw': int(key_raw.shape[1]),
        'key_dim_active': int(active.sum()),
        'step': args.step,
        'knn': args.knn,
        'disjoint_groups': args.disjoint_groups,
        'amp_rank': args.amp_rank,
        'amp_num': len(amp_bases),
        'znext_scale': znext_scale,
        'kth_radius_quantiles': {
            str(q): float(np.quantile(kth_radius, q))
            for q in (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0)
        },
        'results': results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2) + '\n')

    print(
        args.policy,
        f'N={N} key={args.key_mode}/{key.shape[1]}D '
        f'k={args.knn} radius_q01={np.quantile(kth_radius, 0.01):.4f} '
        f'q50={np.quantile(kth_radius, 0.5):.4f}',
    )
    for row in results:
        if row['status'] != 'ok':
            print(
                f"radius={row['radius']:.4f} groups={row['n_groups']} "
                f"-> INCONCLUSIVE ({row['reason']})"
            )
            continue
        print(
            f"radius={row['radius']:.4f} groups={row['n_groups']} "
            f"cond_std/scale={row['cond_std_ratio']:.4f} "
            f"aniso={row['cond_cov_anisotropy']:.2f} "
            f"amp_fraction={row['condvar_in_amp_span_med']:.4f} "
            f"enrichment={row['amp_alignment_enrichment']:.2f}x "
            f"-> {row['verdict_gate']} ({row['reason']})"
        )


if __name__ == '__main__':
    main()
