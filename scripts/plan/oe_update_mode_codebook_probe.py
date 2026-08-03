"""Oracle-routing ceiling for discrete CEM update-correction modes.

Continuous frozen-head regression can reduce update magnitude error while
still averaging incompatible correction directions.  This probe asks whether
the missing corrections instead admit a small *discrete* codebook.

For every outer held-out-state fold, correction residuals from training states
are clustered with deterministic spherical k-means.  A held-out example is
then routed by an oracle to the codebook entry that minimizes its update error;
the zero correction is always available as a safety option.  This routing is
not deployable.  It is the ceiling test that decides whether implementing a
latent/planner-feature router is worthwhile.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from oe_update_corrector_probe import (
    EPS,
    load_trace,
    paired_bootstrap,
    state_aggregate,
    update_metrics,
)


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(
        np.linalg.norm(values, axis=1, keepdims=True),
        EPS,
    )


def spherical_kmeans(
    values: np.ndarray,
    *,
    clusters: int,
    seed: int,
    iterations: int = 100,
    restarts: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    if clusters < 1 or clusters > len(values):
        raise ValueError(f'invalid K={clusters} for {len(values)} rows')
    direction = unit_rows(values)
    best_score = -np.inf
    best_labels = None
    rng = np.random.default_rng(seed)
    for restart in range(restarts):
        first = int(rng.integers(len(direction)))
        selected = [first]
        # Farthest-first initialization in cosine distance.
        nearest = direction @ direction[first]
        for _ in range(1, clusters):
            probabilities = np.maximum(1.0 - nearest, 1e-6)
            probabilities[selected] = 0.0
            probabilities /= probabilities.sum()
            chosen = int(rng.choice(len(direction), p=probabilities))
            selected.append(chosen)
            nearest = np.maximum(nearest, direction @ direction[chosen])
        centroids = direction[selected].copy()
        labels = np.zeros(len(direction), dtype=np.int64)
        for _ in range(iterations):
            new_labels = np.argmax(direction @ centroids.T, axis=1)
            new_centroids = []
            for cluster in range(clusters):
                members = direction[new_labels == cluster]
                if len(members) == 0:
                    # Re-seed with the worst represented row.
                    represented = np.max(direction @ centroids.T, axis=1)
                    replacement = direction[int(np.argmin(represented))]
                    new_centroids.append(replacement)
                else:
                    centroid = members.mean(axis=0)
                    centroid /= max(np.linalg.norm(centroid), EPS)
                    new_centroids.append(centroid)
            new_centroids = np.asarray(new_centroids)
            if np.array_equal(new_labels, labels) and np.allclose(
                new_centroids,
                centroids,
                atol=1e-8,
            ):
                labels = new_labels
                centroids = new_centroids
                break
            labels = new_labels
            centroids = new_centroids
        score = float(
            np.mean(np.max(direction @ centroids.T, axis=1))
        )
        if score > best_score:
            best_score = score
            best_labels = labels.copy()
    if best_labels is None:
        raise AssertionError('spherical k-means did not produce labels')
    # Preserve within-mode magnitude instead of returning unit vectors.
    prototypes = np.stack(
        [values[best_labels == cluster].mean(axis=0) for cluster in range(clusters)]
    )
    return prototypes, best_labels


def oracle_route(
    baseline: np.ndarray,
    oracle: np.ndarray,
    proposal_std: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # The zero residual is an explicit no-op / safety mode.
    codebook = np.concatenate(
        [np.zeros((1, prototypes.shape[1])), prototypes],
        axis=0,
    )
    candidates = baseline[:, None, :] + codebook[None]
    actual = candidates * proposal_std[:, None, :]
    target = oracle * proposal_std
    relative = (
        np.linalg.norm(actual - target[:, None, :], axis=2)
        / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), EPS)
    )
    selected = np.argmin(relative, axis=1)
    return candidates[np.arange(len(candidates)), selected], selected


def pca_fraction(values: np.ndarray, ranks: tuple[int, ...]) -> dict[str, float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    total = max(float(np.sum(energy)), EPS)
    cumulative = np.cumsum(energy) / total
    return {
        str(rank): float(cumulative[min(rank, len(cumulative)) - 1])
        for rank in ranks
    }


def analyze(
    source: Path,
    *,
    topk: int,
    cluster_counts: list[int],
    bootstrap: int,
    seed: int,
) -> dict:
    trace = load_trace(source, topk=topk)
    residual = (
        trace.oracle_update_normalized
        - trace.model_update_normalized
    )
    num_states = int(np.max(trace.state_ids)) + 1
    states = np.arange(num_states)
    baseline_cosine, baseline_relative = update_metrics(
        trace.model_update_normalized,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    baseline_state_cosine = state_aggregate(
        baseline_cosine,
        trace.state_ids,
        states,
    )
    baseline_state_relative = state_aggregate(
        baseline_relative,
        trace.state_ids,
        states,
    )
    analyses = []
    for clusters in cluster_counts:
        routed = np.empty_like(trace.model_update_normalized)
        selected_modes = np.empty(len(routed), dtype=np.int64)
        fold_rows = []
        pca_rows = []
        for fold in range(3):
            val_states = states[states % 3 == fold]
            train_states = states[states % 3 != fold]
            train_indices = np.flatnonzero(
                np.isin(trace.state_ids, train_states)
            )
            val_indices = np.flatnonzero(
                np.isin(trace.state_ids, val_states)
            )
            prototypes, labels = spherical_kmeans(
                residual[train_indices],
                clusters=clusters,
                seed=seed + 1000 * clusters + fold,
            )
            routed[val_indices], selected_modes[val_indices] = oracle_route(
                trace.model_update_normalized[val_indices],
                trace.oracle_update_normalized[val_indices],
                trace.proposal_std[val_indices],
                prototypes,
            )
            counts = np.bincount(labels, minlength=clusters)
            fold_rows.append(
                {
                    'fold': fold,
                    'train_states': train_states.tolist(),
                    'val_states': val_states.tolist(),
                    'cluster_sizes': counts.tolist(),
                    'no_op_fraction': float(
                        np.mean(selected_modes[val_indices] == 0)
                    ),
                }
            )
            pca_rows.append(
                pca_fraction(
                    residual[train_indices],
                    (4, 8, 16, 32),
                )
            )

        routed_cosine, routed_relative = update_metrics(
            routed,
            trace.oracle_update_normalized,
            trace.proposal_std,
        )
        routed_state_cosine = state_aggregate(
            routed_cosine,
            trace.state_ids,
            states,
        )
        routed_state_relative = state_aggregate(
            routed_relative,
            trace.state_ids,
            states,
        )
        cosine_delta = routed_state_cosine - baseline_state_cosine
        relative_delta = routed_state_relative - baseline_state_relative
        rng = np.random.default_rng(seed + clusters)
        analyses.append(
            {
                'clusters': clusters,
                'baseline_cosine': float(np.mean(baseline_state_cosine)),
                'routed_cosine': float(np.mean(routed_state_cosine)),
                'cosine_delta': float(np.mean(cosine_delta)),
                'cosine_delta_ci': list(
                    paired_bootstrap(
                        cosine_delta,
                        samples=bootstrap,
                        rng=rng,
                    )
                ),
                'baseline_relative_error': float(
                    np.mean(baseline_state_relative)
                ),
                'routed_relative_error': float(
                    np.mean(routed_state_relative)
                ),
                'relative_error_delta': float(np.mean(relative_delta)),
                'relative_error_delta_ci': list(
                    paired_bootstrap(
                        relative_delta,
                        samples=bootstrap,
                        rng=rng,
                    )
                ),
                'no_op_fraction': float(
                    np.mean(selected_modes == 0)
                ),
                'folds': fold_rows,
                'pca_fraction_mean': {
                    rank: float(
                        np.mean([row[rank] for row in pca_rows])
                    )
                    for rank in pca_rows[0]
                },
            }
        )
    return {
        'cell': trace.label,
        'source': str(source.resolve()),
        'num_states': num_states,
        'num_examples': int(len(residual)),
        'analyses': analyses,
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def write_outputs(out_dir: Path, results: list[dict], args) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'cluster_counts': args.cluster_counts,
        'results': results,
    }
    (out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    rows = []
    for result in results:
        for row in result['analyses']:
            rows.append(
                {
                    'cell': result['cell'],
                    'clusters': row['clusters'],
                    'baseline_cosine': row['baseline_cosine'],
                    'routed_cosine': row['routed_cosine'],
                    'cosine_delta': row['cosine_delta'],
                    'cosine_delta_ci_low': row['cosine_delta_ci'][0],
                    'cosine_delta_ci_high': row['cosine_delta_ci'][1],
                    'baseline_relative_error': row[
                        'baseline_relative_error'
                    ],
                    'routed_relative_error': row[
                        'routed_relative_error'
                    ],
                    'relative_error_delta': row['relative_error_delta'],
                    'relative_error_delta_ci_low': row[
                        'relative_error_delta_ci'
                    ][0],
                    'relative_error_delta_ci_high': row[
                        'relative_error_delta_ci'
                    ][1],
                    'no_op_fraction': row['no_op_fraction'],
                    **{
                        f'pca_fraction_{rank}': value
                        for rank, value in row[
                            'pca_fraction_mean'
                        ].items()
                    },
                }
            )
    with (out_dir / 'metrics.csv').open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Discrete OE update-mode codebook ceiling',
        '',
        'Codebooks use only outer-training states; held-out routing is oracle '
        'and therefore an upper bound, not a deployable result. The zero '
        'correction is always available.',
        '',
        '| cell | K | baseline cosine | oracle-routed cosine | Δ cosine '
        '| baseline rel. error | oracle-routed rel. error | Δ rel. error '
        '| no-op |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for result in results:
        for row in result['analyses']:
            lines.append(
                f'| {result["cell"]} | {row["clusters"]} '
                f'| {row["baseline_cosine"]:.3f} '
                f'| {row["routed_cosine"]:.3f} '
                f'| {row["cosine_delta"]:+.3f} '
                f'{format_ci(row["cosine_delta_ci"])} '
                f'| {row["baseline_relative_error"]:.3f} '
                f'| {row["routed_relative_error"]:.3f} '
                f'| {row["relative_error_delta"]:+.3f} '
                f'{format_ci(row["relative_error_delta_ci"])} '
                f'| {row["no_op_fraction"]:.3f} |'
            )
    lines.extend(
        [
            '',
            'A strong oracle-routed ceiling justifies building a deployable '
            'latent/planner router. A weak ceiling rejects the discrete-mode '
            'abstraction before router training.',
            '',
        ]
    )
    (out_dir / 'report.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument(
        '--clusters',
        default='2,4,8,16,32',
    )
    args = parser.parse_args()
    args.cluster_counts = [
        int(item)
        for item in args.clusters.split(',')
        if item.strip()
    ]
    results = [
        analyze(
            source,
            topk=args.topk,
            cluster_counts=args.cluster_counts,
            bootstrap=args.bootstrap,
            seed=args.seed + 100 * source_i,
        )
        for source_i, source in enumerate(args.sources)
    ]
    write_outputs(args.out_dir, results, args)
    for result in results:
        best = max(
            result['analyses'],
            key=lambda row: row['cosine_delta'],
        )
        print(
            f'{result["cell"]}: best K={best["clusters"]} '
            f'cos Δ={best["cosine_delta"]:+.3f}, '
            f'rel Δ={best["relative_error_delta"]:+.3f}, '
            f'no-op={best["no_op_fraction"]:.3f}',
            flush=True,
        )
    print(f'results -> {args.out_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
