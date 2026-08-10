#!/usr/bin/env python3
"""Audit the input-side intrinsic dimension used by the controlled-metric plan.

The D=192 LeWM encoder receives observations from a deterministic PushT system.
Before attributing representation freedom to an overcomplete latent, measure the
dimension of the physical variables at the model's observation cadence.  This
script reports global PCA ranks and two local intrinsic-dimension estimates for
four nested descriptions:

* current physical state (7 coordinates),
* three-frame physical history (21 coordinates),
* current state plus its five-action block (17 coordinates), and
* three controlled frames (51 coordinates).

The action-inclusive descriptions are not encoder inputs.  They are included
because they upper-bound the controlled transition context used by the
predictor and match the 17-D/51-D keys used by the conditional-variance audit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=50_000)
    parser.add_argument('--local-samples', type=int, default=12_000)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--knn', type=int, default=20)
    parser.add_argument('--local-pca-neighbors', type=int, default=64)
    parser.add_argument('--local-pca-anchors', type=int, default=1_000)
    parser.add_argument('--seed', type=int, default=20260810)
    return parser.parse_args()


def sample_contexts(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.dataset, 'r') as handle:
        lengths = np.asarray(handle['ep_len'], dtype=np.int64)
        offsets = np.asarray(handle['ep_offset'], dtype=np.int64)
        state_ds = handle['state']
        action_ds = handle['action']

        span = args.history * args.frameskip
        valid_counts = np.maximum(lengths - span + 1, 0)
        valid_episodes = np.flatnonzero(valid_counts > 0)
        if len(valid_episodes) == 0:
            raise ValueError(f'no episode is long enough for span={span}')
        weights = valid_counts[valid_episodes].astype(np.float64)
        weights /= weights.sum()
        episodes = rng.choice(
            valid_episodes,
            size=args.num_samples,
            replace=True,
            p=weights,
        )
        starts = np.asarray(
            [rng.integers(0, valid_counts[ep]) for ep in episodes],
            dtype=np.int64,
        )

        bases = offsets[episodes] + starts
        state_indices = bases[:, None] + (
            np.arange(args.history, dtype=np.int64)[None, :] * args.frameskip
        )
        unique_state, inverse_state = np.unique(
            state_indices.reshape(-1), return_inverse=True
        )
        state_history = np.asarray(state_ds[unique_state])[inverse_state].reshape(
            args.num_samples, args.history, state_ds.shape[1]
        )

        action_indices = state_indices[:, :, None] + np.arange(
            args.frameskip, dtype=np.int64
        )[None, None, :]
        unique_action, inverse_action = np.unique(
            action_indices.reshape(-1), return_inverse=True
        )
        action_history = np.asarray(action_ds[unique_action])[inverse_action]
        action_history = action_history.reshape(
            args.num_samples,
            args.history,
            args.frameskip * action_ds.shape[1],
        )

    controlled = np.concatenate([state_history, action_history], axis=-1)
    return {
        'current_state': state_history[:, -1],
        'state_history': state_history.reshape(args.num_samples, -1),
        'current_controlled': controlled[:, -1],
        'controlled_history': controlled.reshape(args.num_samples, -1),
    }


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    keep = std > 1e-10
    if not np.any(keep):
        raise ValueError('all input coordinates are constant')
    normalized = (values[:, keep].astype(np.float64) - mean[keep]) / std[keep]
    return normalized, mean, std


def pca_summary(values: np.ndarray) -> dict[str, object]:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1].clip(min=0.0)
    total = eigenvalues.sum()
    fractions = eigenvalues / max(total, 1e-30)
    cumulative = np.cumsum(fractions)

    def rank_at(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold, side='left') + 1)

    participation = float(total**2 / max(np.square(eigenvalues).sum(), 1e-30))
    entropy = -float(np.sum(fractions * np.log(fractions + 1e-30)))
    return {
        'participation_rank': participation,
        'entropy_effective_rank': float(np.exp(entropy)),
        'pca_rank_95': rank_at(0.95),
        'pca_rank_99': rank_at(0.99),
        'pca_rank_999': rank_at(0.999),
        'eigenvalue_fraction': fractions.tolist(),
    }


def local_id_summary(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    num_samples: int,
    knn: int,
    pca_neighbors: int,
    pca_anchors: int,
) -> dict[str, object]:
    count = min(num_samples, len(values))
    selected = rng.choice(len(values), size=count, replace=False)
    sample = values[selected]
    neighbors = max(knn + 1, pca_neighbors + 1)
    nn = NearestNeighbors(n_neighbors=neighbors, n_jobs=-1)
    nn.fit(sample)
    distances, indices = nn.kneighbors(sample, return_distance=True)
    distances = distances[:, 1:]

    duplicate_fraction = float(np.mean(distances[:, 0] <= 1e-12))
    r = distances[:, :knn]
    valid = (r[:, 0] > 1e-12) & (r[:, -1] > r[:, 0])
    r_valid = r[valid]
    mle = (knn - 1) / np.maximum(
        np.log(r_valid[:, -1, None] / r_valid[:, :-1]).sum(axis=1),
        1e-12,
    )
    mle = mle[np.isfinite(mle)]
    twonn_log_ratio = np.log(r_valid[:, 1] / r_valid[:, 0])
    twonn_log_ratio = twonn_log_ratio[twonn_log_ratio > 0]
    twonn = (
        float(1.0 / twonn_log_ratio.mean())
        if len(twonn_log_ratio)
        else float('nan')
    )

    anchor_count = min(pca_anchors, count)
    anchors = rng.choice(count, size=anchor_count, replace=False)
    local_ranks = []
    for anchor in anchors:
        local = sample[indices[anchor, 1 : pca_neighbors + 1]]
        local -= local.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(local, full_matrices=False, compute_uv=False)
        energy = np.square(singular)
        cumulative = np.cumsum(energy) / max(energy.sum(), 1e-30)
        local_ranks.append(int(np.searchsorted(cumulative, 0.95) + 1))

    return {
        'sample_count': count,
        'duplicate_neighbor_fraction': duplicate_fraction,
        'mle_knn': knn,
        'mle_valid_count': int(len(mle)),
        'mle_id_median': float(np.median(mle)),
        'mle_id_trimmed_mean': float(
            np.mean(np.clip(mle, np.percentile(mle, 5), np.percentile(mle, 95)))
        ),
        'twonn_id': twonn,
        'local_pca_neighbors': pca_neighbors,
        'local_pca_rank95_median': float(np.median(local_ranks)),
        'local_pca_rank95_p25': float(np.percentile(local_ranks, 25)),
        'local_pca_rank95_p75': float(np.percentile(local_ranks, 75)),
    }


def main() -> None:
    args = parse_args()
    if args.knn < 3:
        raise ValueError('--knn must be at least 3')
    if args.local_pca_neighbors <= args.knn:
        raise ValueError('--local-pca-neighbors must exceed --knn')

    contexts = sample_contexts(args)
    rng = np.random.default_rng(args.seed + 1)
    result: dict[str, object] = {
        'config': {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        'theoretical': {
            'latent_ambient_dim': 192,
            'physical_state_dim': 7,
            'state_history_upper_bound': 7 * args.history,
            'action_block_dim': 2 * args.frameskip,
            'controlled_history_ambient_dim': (
                (7 + 2 * args.frameskip) * args.history
            ),
        },
        'descriptions': {},
    }

    for name, raw in contexts.items():
        normalized, _, std = standardize(raw)
        summary = {
            'ambient_dim': int(raw.shape[1]),
            'nonconstant_dim': int(normalized.shape[1]),
            'constant_dim': int(np.sum(std <= 1e-10)),
        }
        summary.update(pca_summary(normalized))
        summary.update(
            local_id_summary(
                normalized,
                rng=rng,
                num_samples=args.local_samples,
                knn=args.knn,
                pca_neighbors=args.local_pca_neighbors,
                pca_anchors=args.local_pca_anchors,
            )
        )
        result['descriptions'][name] = summary
        print(
            f'{name:20s} ambient={summary["ambient_dim"]:2d} '
            f'PCA99={summary["pca_rank_99"]:2d} '
            f'PR={summary["participation_rank"]:5.2f} '
            f'MLE={summary["mle_id_median"]:5.2f} '
            f'localPCA95={summary["local_pca_rank95_median"]:4.1f}',
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
