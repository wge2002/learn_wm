"""Audit optimizer-basin topology at matched elite rates.

The audit asks a stricter question than rank correlation: on the same sampled
action population, does a world-model cost preserve the connected low-cost
basins induced by true dynamics?

Candidate actions are projected onto a two-dimensional, state-held-out basis
of true-minus-K3 CEM update directions.  A symmetric k-NN graph is fixed in
that geometry.  True, K3, and K10 costs then induce lower-star filtrations on
the *same* graph.  We compare persistent H0 basins and connected components
at identical elite rates, so results are invariant to each cost's scale.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from oe_update_mode_codebook_probe import spherical_kmeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--k3-outcome', type=Path, required=True)
    parser.add_argument('--k10-outcome', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--projection-dims', type=int, default=2)
    parser.add_argument(
        '--geometry',
        choices=('correction', 'full', 'random'),
        default='correction',
        help=(
            'Candidate graph geometry: a state-held-out correction-mode '
            'subspace, the full proposal-whitened action space, or a '
            'seeded random subspace of matched dimension.'
        ),
    )
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--elite', type=int, default=30)
    parser.add_argument('--codebook-modes', type=int, default=4)
    parser.add_argument('--neighbors', default='8,12,20')
    parser.add_argument('--rates', default='.05,.1,.2')
    parser.add_argument(
        '--persistence-thresholds',
        default='.05,.1,.2',
    )
    parser.add_argument('--bootstrap', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260720)
    return parser.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(',') if item]


def comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item]


def load_inputs(
    source_path: Path,
    k3_path: Path,
    k10_path: Path,
) -> dict[str, np.ndarray]:
    with np.load(source_path, allow_pickle=False) as archive:
        source = {
            key: np.asarray(archive[key])
            for key in (
                'rows',
                'candidates',
                'true',
                'success',
                'pred',
                'prev_mean',
                'prev_var',
                'steps',
            )
        }
    if source['candidates'].shape[1] != 1:
        raise ValueError('audit currently expects one generator')
    candidates = source['candidates'][:, 0].astype(np.float32)
    true_cost = source['true'][:, 0].astype(np.float64)
    k3_source = source['pred'][:, 0, :, 0].astype(np.float64)
    rows = source['rows']

    costs = {}
    for name, path in (('k3', k3_path), ('k10', k10_path)):
        with np.load(path, allow_pickle=False) as archive:
            outcome_rows = np.asarray(archive['rows'])
            recomputed = np.asarray(
                archive['recomputed_cost'],
                dtype=np.float64,
            )
        if not np.array_equal(rows, outcome_rows):
            raise ValueError(f'{name} outcome rows differ from source')
        if recomputed.shape != true_cost.shape:
            raise ValueError(
                f'{name} cost shape {recomputed.shape} != '
                f'{true_cost.shape}'
            )
        costs[name] = recomputed
    max_k3_mismatch = float(
        np.max(np.abs(costs['k3'] - k3_source))
    )
    return {
        'rows': rows,
        'candidates': candidates,
        'true': true_cost,
        'success': source['success'][:, 0].astype(bool),
        # Use the exact costs that generated the recorded CEM update.  The
        # recomputed cache sees float16-serialized candidates and is retained
        # only as a numerical audit.
        'k3': k3_source,
        'k10': costs['k10'],
        'prev_mean': source['prev_mean'][:, 0].astype(np.float32),
        'prev_std': np.maximum(
            source['prev_var'][:, 0].astype(np.float32),
            1e-4,
        ),
        'steps': source['steps'],
        'max_k3_recompute_mismatch': np.asarray(max_k3_mismatch),
    }


def fold_assignments(num_states: int, folds: int) -> np.ndarray:
    if folds < 2 or folds > num_states:
        raise ValueError('folds must be in [2, num_states]')
    # Contiguous blocks preserve the independent query-bank shard boundary
    # when the source is a concatenation of equal-sized collections.
    return np.minimum(
        np.arange(num_states) * folds // num_states,
        folds - 1,
    )


def correction_basis(
    data: dict[str, np.ndarray],
    *,
    training_states: np.ndarray,
    elite: int,
    dims: int,
    codebook_modes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrections = []
    for state_i in training_states:
        for round_i in range(data['candidates'].shape[1]):
            candidates = data['candidates'][state_i, round_i]
            mean = data['prev_mean'][state_i, round_i]
            std = data['prev_std'][state_i, round_i]
            normalized = (
                (candidates - mean[None]) / std[None]
            ).reshape(len(candidates), -1)
            true_order = np.argsort(
                data['true'][state_i, round_i],
                kind='stable',
            )[:elite]
            k3_order = np.argsort(
                data['k3'][state_i, round_i],
                kind='stable',
            )[:elite]
            corrections.append(
                normalized[true_order].mean(axis=0)
                - normalized[k3_order].mean(axis=0)
            )
    matrix = np.asarray(corrections, dtype=np.float64)
    prototypes, _ = spherical_kmeans(
        matrix,
        clusters=codebook_modes,
        seed=seed,
    )
    _, singular, vt = np.linalg.svd(
        prototypes,
        full_matrices=False,
    )
    energy = np.square(singular)
    explained = energy / max(float(energy.sum()), 1e-12)
    raw_singular = np.linalg.svd(
        matrix,
        full_matrices=False,
        compute_uv=False,
    )
    raw_energy = np.square(raw_singular)
    raw_explained = raw_energy / max(
        float(raw_energy.sum()),
        1e-12,
    )
    return vt[:dims], explained, raw_explained


def projected_population(
    data: dict[str, np.ndarray],
    *,
    state_i: int,
    round_i: int,
    basis: np.ndarray,
) -> np.ndarray:
    candidates = data['candidates'][state_i, round_i]
    normalized = (
        (
            candidates
            - data['prev_mean'][state_i, round_i][None]
        )
        / data['prev_std'][state_i, round_i][None]
    ).reshape(len(candidates), -1)
    projected = normalized @ basis.T
    scale = np.maximum(projected.std(axis=0), 1e-6)
    return projected / scale


def full_population(
    data: dict[str, np.ndarray],
    *,
    state_i: int,
    round_i: int,
) -> np.ndarray:
    candidates = data['candidates'][state_i, round_i]
    normalized = (
        (
            candidates
            - data['prev_mean'][state_i, round_i][None]
        )
        / data['prev_std'][state_i, round_i][None]
    ).reshape(len(candidates), -1)
    scale = np.maximum(normalized.std(axis=0), 1e-6)
    return normalized / scale


def random_basis(
    width: int,
    *,
    dims: int,
    seed: int,
) -> np.ndarray:
    if dims > width:
        raise ValueError(
            f'projection dims {dims} exceed action width {width}'
        )
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(width, dims))
    q, _ = np.linalg.qr(matrix)
    return q[:, :dims].T


def symmetric_knn(
    points: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    count = len(points)
    if neighbors < 1 or neighbors >= count:
        raise ValueError('neighbors must be in [1, N-1]')
    squared = np.sum(
        np.square(points[:, None] - points[None]),
        axis=-1,
    )
    np.fill_diagonal(squared, np.inf)
    nearest = np.argpartition(
        squared,
        kth=neighbors - 1,
        axis=1,
    )[:, :neighbors]
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[
        np.arange(count)[:, None],
        nearest,
    ] = True
    adjacency |= adjacency.T
    return adjacency


def component_labels(
    adjacency: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    count = len(active)
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left = find(left)
        right = find(right)
        if left != right:
            parent[right] = left

    active_indices = np.flatnonzero(active)
    for left in active_indices:
        for right in np.flatnonzero(
            adjacency[left] & active
        ):
            if right > left:
                union(int(left), int(right))
    labels = np.full(count, -1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    for index in active_indices:
        root = find(int(index))
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[index] = root_to_label[root]
    return labels


def persistent_h0(
    adjacency: np.ndarray,
    cost: np.ndarray,
) -> np.ndarray:
    count = len(cost)
    order = np.argsort(cost, kind='stable')
    parent = np.arange(count)
    birth = np.full(count, np.nan, dtype=np.float64)
    active = np.zeros(count, dtype=bool)
    persistence = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for rank_i, raw_index in enumerate(order):
        index = int(raw_index)
        level = rank_i / max(count - 1, 1)
        active[index] = True
        parent[index] = index
        birth[index] = level
        for raw_neighbor in np.flatnonzero(
            adjacency[index] & active
        ):
            neighbor = int(raw_neighbor)
            root_left = find(index)
            root_right = find(neighbor)
            if root_left == root_right:
                continue
            if birth[root_left] <= birth[root_right]:
                older, younger = root_left, root_right
            else:
                older, younger = root_right, root_left
            persistence.append(level - birth[younger])
            parent[younger] = older
            parent[index] = older

    roots = {
        find(int(index))
        for index in np.flatnonzero(active)
    }
    persistence.extend(1.0 - birth[root] for root in roots)
    values = np.sort(
        np.asarray(persistence, dtype=np.float64)
    )[::-1]
    # The oldest connected component is the unavoidable global H0 class.
    return values[1:] if len(values) else values


def persistence_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    width = max(len(left), len(right))
    padded_left = np.zeros(width, dtype=np.float64)
    padded_right = np.zeros(width, dtype=np.float64)
    padded_left[: len(left)] = left
    padded_right[: len(right)] = right
    return float(np.mean(np.abs(padded_left - padded_right)))


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def paired_bootstrap(
    values: np.ndarray,
    state_indices: np.ndarray,
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    states = np.unique(state_indices)
    state_means = np.asarray(
        [
            np.mean(values[state_indices == state])
            for state in states
        ],
        dtype=np.float64,
    )
    indices = rng.integers(
        0,
        len(states),
        size=(bootstrap, len(states)),
    )
    distribution = state_means[indices].mean(axis=1)
    return {
        'mean': float(state_means.mean()),
        'median_state_mean': float(np.median(state_means)),
        'ci95': np.quantile(
            distribution,
            [0.025, 0.975],
        ).tolist(),
        'states': int(len(states)),
    }


def main() -> None:
    args = parse_args()
    neighbors_grid = comma_ints(args.neighbors)
    rates = comma_floats(args.rates)
    persistence_thresholds = comma_floats(
        args.persistence_thresholds
    )
    data = load_inputs(
        args.source,
        args.k3_outcome,
        args.k10_outcome,
    )
    num_states, num_rounds, population_size = data['true'].shape
    folds = fold_assignments(num_states, args.folds)
    bases = {}
    explained = {}
    raw_explained = {}
    action_width = int(np.prod(data['candidates'].shape[-2:]))
    if args.geometry == 'correction':
        for fold in range(args.folds):
            basis, fold_explained, fold_raw_explained = (
                correction_basis(
                    data,
                    training_states=np.flatnonzero(folds != fold),
                    elite=args.elite,
                    dims=args.projection_dims,
                    codebook_modes=args.codebook_modes,
                    seed=args.seed + fold,
                )
            )
            bases[fold] = basis
            explained[fold] = fold_explained
            raw_explained[fold] = fold_raw_explained
    elif args.geometry == 'random':
        for fold in range(args.folds):
            bases[fold] = random_basis(
                action_width,
                dims=args.projection_dims,
                seed=args.seed + fold,
            )

    cost_names = (
        'true',
        'k3',
        'k10',
        'minrank',
        'consensus',
        'shuffle',
    )
    comparison_names = (
        'k3',
        'k10',
        'minrank',
        'consensus',
        'shuffle',
    )
    persistence_records = []
    component_records = []
    rng = np.random.default_rng(args.seed)
    for state_i in range(num_states):
        basis = bases.get(int(folds[state_i]))
        for round_i in range(num_rounds):
            if args.geometry == 'full':
                points = full_population(
                    data,
                    state_i=state_i,
                    round_i=round_i,
                )
            else:
                points = projected_population(
                    data,
                    state_i=state_i,
                    round_i=round_i,
                    basis=basis,
                )
            shuffled = data['true'][state_i, round_i][
                rng.permutation(population_size)
            ]
            k3_rank = rank_fraction(
                data['k3'][state_i, round_i]
            )
            k10_rank = rank_fraction(
                data['k10'][state_i, round_i]
            )
            consensus = 0.5 * (k3_rank + k10_rank)
            costs = {
                'true': data['true'][state_i, round_i],
                'k3': data['k3'][state_i, round_i],
                'k10': data['k10'][state_i, round_i],
                # The tiny consensus tie-break makes the min-rank frontier
                # deterministic without materially changing its union
                # semantics.
                'minrank': (
                    np.minimum(k3_rank, k10_rank)
                    + 1e-6 * consensus
                ),
                'consensus': consensus,
                'shuffle': shuffled,
            }
            for neighbors in neighbors_grid:
                adjacency = symmetric_knn(
                    points,
                    neighbors=neighbors,
                )
                persistence = {
                    name: persistent_h0(adjacency, costs[name])
                    for name in cost_names
                }
                for threshold in persistence_thresholds:
                    record = {
                        'state': state_i,
                        'round': round_i,
                        'neighbors': neighbors,
                        'threshold': threshold,
                    }
                    for name in cost_names:
                        record[f'{name}_count'] = int(
                            np.sum(persistence[name] >= threshold)
                        )
                    for name in comparison_names:
                        record[f'{name}_distance'] = (
                            persistence_distance(
                                persistence['true'],
                                persistence[name],
                            )
                        )
                    persistence_records.append(record)

                for rate in rates:
                    active_count = max(
                        2,
                        int(math.ceil(rate * population_size)),
                    )
                    active = {}
                    labels = {}
                    for name in cost_names:
                        mask = np.zeros(
                            population_size,
                            dtype=bool,
                        )
                        mask[
                            np.argsort(
                                costs[name],
                                kind='stable',
                            )[:active_count]
                        ] = True
                        active[name] = mask
                        labels[name] = component_labels(
                            adjacency,
                            mask,
                        )
                    true_labels = labels['true']
                    true_components = np.unique(
                        true_labels[true_labels >= 0]
                    )
                    candidate_success = data['success'][
                        state_i,
                        round_i,
                    ]
                    successful_true_components = [
                        component
                        for component in true_components
                        if np.any(
                            candidate_success
                            & (true_labels == component)
                        )
                    ]
                    record = {
                        'state': state_i,
                        'round': round_i,
                        'neighbors': neighbors,
                        'rate': rate,
                        'true_count': int(len(true_components)),
                        'successful_true_count': int(
                            len(successful_true_components)
                        ),
                    }
                    for name in comparison_names:
                        model_components = np.unique(
                            labels[name][labels[name] >= 0]
                        )
                        hits = []
                        masses = []
                        for component in true_components:
                            members = true_labels == component
                            hits.append(
                                bool(np.any(active[name] & members))
                            )
                            masses.append(int(np.sum(members)))
                        record[f'{name}_count'] = int(
                            len(model_components)
                        )
                        record[f'{name}_elite_recall'] = float(
                            np.mean(
                                active[name] & active['true']
                            )
                            / max(np.mean(active['true']), 1e-12)
                        )
                        record[f'{name}_component_coverage'] = (
                            float(np.mean(hits))
                            if hits
                            else 0.0
                        )
                        successful_hits = [
                            bool(
                                np.any(
                                    active[name]
                                    & (
                                        true_labels
                                        == component
                                    )
                                )
                            )
                            for component
                            in successful_true_components
                        ]
                        record[
                            f'{name}_successful_component_coverage'
                        ] = (
                            float(np.mean(successful_hits))
                            if successful_hits
                            else np.nan
                        )
                        success_count = int(
                            np.sum(candidate_success)
                        )
                        record[f'{name}_success_recall'] = (
                            float(
                                np.sum(
                                    active[name]
                                    & candidate_success
                                )
                                / success_count
                            )
                            if success_count
                            else np.nan
                        )
                        record[f'{name}_success_precision'] = float(
                            np.mean(candidate_success[active[name]])
                        )
                        record[f'{name}_mass_coverage'] = (
                            float(
                                np.average(
                                    np.asarray(hits, dtype=np.float64),
                                    weights=np.asarray(masses),
                                )
                            )
                            if masses
                            else 0.0
                        )
                    component_records.append(record)

            if (
                (state_i * num_rounds + round_i + 1)
                % max(1, num_rounds * 10)
                == 0
            ):
                print(
                    f'topology audit '
                    f'{state_i + 1}/{num_states} states',
                    flush=True,
                )

    args.out.mkdir(parents=True, exist_ok=True)
    persistence_keys = list(persistence_records[0])
    persistence_arrays = {
        key: np.asarray([row[key] for row in persistence_records])
        for key in persistence_keys
    }
    component_keys = list(component_records[0])
    component_arrays = {
        key: np.asarray([row[key] for row in component_records])
        for key in component_keys
    }
    np.savez_compressed(
        args.out / 'raw_metrics.npz',
        rows=data['rows'],
        steps=data['steps'],
        fold=folds,
        **{
            f'persistence_{key}': value
            for key, value in persistence_arrays.items()
        },
        **{
            f'component_{key}': value
            for key, value in component_arrays.items()
        },
    )

    report = {
        'version': 1,
        'source': str(args.source.resolve()),
        'num_states': num_states,
        'num_rounds': num_rounds,
        'population_size': population_size,
        'geometry': args.geometry,
        'folds': args.folds,
        'projection_dims': (
            action_width
            if args.geometry == 'full'
            else args.projection_dims
        ),
        'codebook_modes': args.codebook_modes,
        'elite_for_basis': args.elite,
        'neighbors': neighbors_grid,
        'rates': rates,
        'persistence_thresholds': persistence_thresholds,
        'max_k3_recompute_mismatch': float(
            data['max_k3_recompute_mismatch']
        ),
        'basis_energy': (
            {
                str(fold): {
                    'first_dim': float(explained[fold][0]),
                    'selected_dims': float(
                        explained[fold][
                            : args.projection_dims
                        ].sum()
                    ),
                    'raw_selected_dims': float(
                        raw_explained[fold][
                            : args.projection_dims
                        ].sum()
                    ),
                }
                for fold in range(args.folds)
            }
            if args.geometry == 'correction'
            else None
        ),
        'persistence': {},
        'components': {},
    }
    summary_rng = np.random.default_rng(args.seed + 1)
    for neighbors in neighbors_grid:
        for threshold in persistence_thresholds:
            mask = (
                (persistence_arrays['neighbors'] == neighbors)
                & (
                    np.abs(
                        persistence_arrays['threshold'] - threshold
                    )
                    < 1e-9
                )
            )
            key = f'k{neighbors}_p{threshold:g}'
            true_count = persistence_arrays['true_count'][mask]
            entry = {
                'true_count': paired_bootstrap(
                    true_count,
                    persistence_arrays['state'][mask],
                    bootstrap=args.bootstrap,
                    rng=summary_rng,
                )
            }
            shuffle_distance = persistence_arrays[
                'shuffle_distance'
            ][mask]
            for name in comparison_names:
                count = persistence_arrays[f'{name}_count'][mask]
                distance = persistence_arrays[
                    f'{name}_distance'
                ][mask]
                entry[name] = {
                    'count': paired_bootstrap(
                        count,
                        persistence_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'count_delta_vs_true': paired_bootstrap(
                        count - true_count,
                        persistence_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'persistence_l1': paired_bootstrap(
                        distance,
                        persistence_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'fidelity_vs_shuffle': float(
                        1.0
                        - np.mean(distance)
                        / max(np.mean(shuffle_distance), 1e-12)
                    ),
                }
            report['persistence'][key] = entry

        for rate in rates:
            mask = (
                (component_arrays['neighbors'] == neighbors)
                & (
                    np.abs(component_arrays['rate'] - rate)
                    < 1e-9
                )
            )
            key = f'k{neighbors}_r{rate:g}'
            true_count = component_arrays['true_count'][mask]
            entry = {
                'true_count': paired_bootstrap(
                    true_count,
                    component_arrays['state'][mask],
                    bootstrap=args.bootstrap,
                    rng=summary_rng,
                ),
                'successful_true_count': paired_bootstrap(
                    component_arrays['successful_true_count'][mask],
                    component_arrays['state'][mask],
                    bootstrap=args.bootstrap,
                    rng=summary_rng,
                ),
            }
            for name in comparison_names:
                count = component_arrays[f'{name}_count'][mask]
                entry[name] = {
                    'count_delta_vs_true': paired_bootstrap(
                        count - true_count,
                        component_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'elite_recall': paired_bootstrap(
                        component_arrays[
                            f'{name}_elite_recall'
                        ][mask],
                        component_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'component_coverage': paired_bootstrap(
                        component_arrays[
                            f'{name}_component_coverage'
                        ][mask],
                        component_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'successful_component_coverage': (
                        paired_bootstrap(
                            component_arrays[
                                f'{name}_successful_component_coverage'
                            ][mask][
                                ~np.isnan(
                                    component_arrays[
                                        f'{name}_successful_component_coverage'
                                    ][mask]
                                )
                            ],
                            component_arrays['state'][mask][
                                ~np.isnan(
                                    component_arrays[
                                        f'{name}_successful_component_coverage'
                                    ][mask]
                                )
                            ],
                            bootstrap=args.bootstrap,
                            rng=summary_rng,
                        )
                    ),
                    'success_recall': paired_bootstrap(
                        component_arrays[
                            f'{name}_success_recall'
                        ][mask][
                            ~np.isnan(
                                component_arrays[
                                    f'{name}_success_recall'
                                ][mask]
                            )
                        ],
                        component_arrays['state'][mask][
                            ~np.isnan(
                                component_arrays[
                                    f'{name}_success_recall'
                                ][mask]
                            )
                        ],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                    'success_precision': paired_bootstrap(
                        component_arrays[
                            f'{name}_success_precision'
                        ][mask],
                        component_arrays['state'][mask],
                        bootstrap=args.bootstrap,
                        rng=summary_rng,
                    ),
                }
            report['components'][key] = entry

    (args.out / 'report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
