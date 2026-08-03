"""Audit whether a CEM update retains the true low-cost basin set.

Static candidate ranking is not the causal object in adaptive planning: the
selected elite set is refit into the proposal that generates the next query
population.  This audit therefore holds the current population fixed, builds
the next diagonal-Gaussian proposal induced by each scorer, and asks whether
that proposal still supports every persistent true basin at an identical
top-k rate.

The graph and component definitions are shared with
``oe_basin_topology_audit.py``.  For each scorer, the next proposal is fit to
its top-k candidates.  We then rank the *current* population by likelihood
under that proposal and retain exactly top-k support witnesses.  This is a
scale-free, iso-rate diagnostic of basin transport; simulator outcomes remain
hidden except for identifying which true basins contain successful actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oe_basin_topology_audit import (
    component_labels,
    correction_basis,
    fold_assignments,
    full_population,
    load_inputs,
    paired_bootstrap,
    projected_population,
    random_basis,
    rank_fraction,
    symmetric_knn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--k3-outcome', type=Path, required=True)
    parser.add_argument('--k10-outcome', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--geometry',
        choices=('correction', 'full', 'random'),
        default='correction',
    )
    parser.add_argument('--projection-dims', type=int, default=2)
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--elite', type=int, default=30)
    parser.add_argument('--codebook-modes', type=int, default=4)
    parser.add_argument('--neighbors', default='8,12,20')
    parser.add_argument('--rates', default='.05,.1,.2')
    parser.add_argument('--support-rate', type=float, default=0.1)
    parser.add_argument('--std-floor', type=float, default=0.05)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260720)
    return parser.parse_args()


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(',') if item]


def comma_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(',') if item]


def fit_proposal(
    candidates: np.ndarray,
    cost: np.ndarray,
    *,
    elite: int,
    std_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(cost, kind='stable')[:elite]
    selected = candidates[indices].astype(np.float64)
    return (
        selected.mean(axis=0),
        np.maximum(selected.std(axis=0, ddof=1), std_floor),
    )


def proposal_support(
    candidates: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    count: int,
) -> np.ndarray:
    normalized = (
        (candidates.astype(np.float64) - mean[None])
        / std[None]
    )
    # The log-determinant is constant across candidates for one proposal.
    nll = np.square(normalized).reshape(len(candidates), -1).mean(axis=1)
    mask = np.zeros(len(candidates), dtype=bool)
    mask[np.argsort(nll, kind='stable')[:count]] = True
    return mask


def main() -> None:
    args = parse_args()
    neighbors_grid = comma_ints(args.neighbors)
    rates = comma_floats(args.rates)
    data = load_inputs(args.source, args.k3_outcome, args.k10_outcome)
    num_states, num_rounds, population_size = data['true'].shape
    folds = fold_assignments(num_states, args.folds)
    action_width = int(np.prod(data['candidates'].shape[-2:]))
    bases: dict[int, np.ndarray] = {}
    if args.geometry == 'correction':
        for fold in range(args.folds):
            basis, _, _ = correction_basis(
                data,
                training_states=np.flatnonzero(folds != fold),
                elite=args.elite,
                dims=args.projection_dims,
                codebook_modes=args.codebook_modes,
                seed=args.seed + fold,
            )
            bases[fold] = basis
    elif args.geometry == 'random':
        for fold in range(args.folds):
            bases[fold] = random_basis(
                action_width,
                dims=args.projection_dims,
                seed=args.seed + fold,
            )

    scorer_names = ('k3', 'k10', 'consensus')
    records = []
    support_count = max(
        2,
        int(np.ceil(args.support_rate * population_size)),
    )
    for state_i in range(num_states):
        basis = bases.get(int(folds[state_i]))
        for round_i in range(num_rounds):
            candidates = data['candidates'][state_i, round_i]
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

            k3_rank = rank_fraction(data['k3'][state_i, round_i])
            k10_rank = rank_fraction(data['k10'][state_i, round_i])
            costs = {
                'true': data['true'][state_i, round_i],
                'k3': data['k3'][state_i, round_i],
                'k10': data['k10'][state_i, round_i],
                'consensus': 0.5 * (k3_rank + k10_rank),
            }
            proposals = {
                name: fit_proposal(
                    candidates,
                    costs[name],
                    elite=args.elite,
                    std_floor=args.std_floor,
                )
                for name in costs
            }
            supports = {
                name: proposal_support(
                    candidates,
                    *proposals[name],
                    count=support_count,
                )
                for name in costs
            }
            oracle_mean, oracle_std = proposals['true']
            candidate_success = data['success'][state_i, round_i]

            for neighbors in neighbors_grid:
                adjacency = symmetric_knn(points, neighbors=neighbors)
                for rate in rates:
                    active_count = max(
                        2,
                        int(np.ceil(rate * population_size)),
                    )
                    true_active = np.zeros(population_size, dtype=bool)
                    true_active[
                        np.argsort(
                            costs['true'],
                            kind='stable',
                        )[:active_count]
                    ] = True
                    true_labels = component_labels(
                        adjacency,
                        true_active,
                    )
                    true_components = np.unique(
                        true_labels[true_labels >= 0]
                    )
                    successful_components = [
                        component
                        for component in true_components
                        if np.any(
                            candidate_success
                            & (true_labels == component)
                        )
                    ]
                    row: dict[str, float | int] = {
                        'state': state_i,
                        'round': round_i,
                        'neighbors': neighbors,
                        'rate': rate,
                        'true_count': int(len(true_components)),
                        'successful_true_count': int(
                            len(successful_components)
                        ),
                    }
                    for name in scorer_names:
                        support = supports[name]
                        hits = [
                            bool(
                                np.any(
                                    support
                                    & (true_labels == component)
                                )
                            )
                            for component in true_components
                        ]
                        successful_hits = [
                            bool(
                                np.any(
                                    support
                                    & (true_labels == component)
                                )
                            )
                            for component in successful_components
                        ]
                        proposal_mean, proposal_std = proposals[name]
                        row[f'{name}_component_coverage'] = (
                            float(np.mean(hits)) if hits else 0.0
                        )
                        row[
                            f'{name}_successful_component_coverage'
                        ] = (
                            float(np.mean(successful_hits))
                            if successful_hits
                            else np.nan
                        )
                        row[f'{name}_elite_recall'] = float(
                            np.mean(support & true_active)
                            / max(np.mean(true_active), 1e-12)
                        )
                        row[f'{name}_success_recall'] = (
                            float(
                                np.sum(support & candidate_success)
                                / np.sum(candidate_success)
                            )
                            if np.any(candidate_success)
                            else np.nan
                        )
                        row[f'{name}_mean_error'] = float(
                            np.sqrt(
                                np.mean(
                                    np.square(
                                        (
                                            proposal_mean
                                            - oracle_mean
                                        )
                                        / np.maximum(
                                            data['prev_std'][
                                                state_i,
                                                round_i,
                                            ],
                                            args.std_floor,
                                        )
                                    )
                                )
                            )
                        )
                        row[f'{name}_logstd_error'] = float(
                            np.mean(
                                np.abs(
                                    np.log(proposal_std)
                                    - np.log(oracle_std)
                                )
                            )
                        )
                    records.append(row)

        if (state_i + 1) % 20 == 0:
            print(
                f'transport audit {state_i + 1}/{num_states} states',
                flush=True,
            )

    arrays = {
        key: np.asarray([row[key] for row in records])
        for key in records[0]
    }
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / 'raw_metrics.npz',
        rows=data['rows'],
        steps=data['steps'],
        fold=folds,
        **arrays,
    )
    report: dict = {
        'version': 1,
        'source': str(args.source.resolve()),
        'num_states': num_states,
        'num_rounds': num_rounds,
        'population_size': population_size,
        'geometry': args.geometry,
        'projection_dims': (
            action_width
            if args.geometry == 'full'
            else args.projection_dims
        ),
        'elite': args.elite,
        'support_rate': args.support_rate,
        'support_count': support_count,
        'neighbors': neighbors_grid,
        'rates': rates,
        'components': {},
    }
    rng = np.random.default_rng(args.seed + 1)
    for neighbors in neighbors_grid:
        for rate in rates:
            mask = (
                (arrays['neighbors'] == neighbors)
                & (np.abs(arrays['rate'] - rate) < 1e-9)
            )
            key = f'k{neighbors}_r{rate:g}'
            entry = {
                'true_count': paired_bootstrap(
                    arrays['true_count'][mask],
                    arrays['state'][mask],
                    bootstrap=args.bootstrap,
                    rng=rng,
                ),
                'successful_true_count': paired_bootstrap(
                    arrays['successful_true_count'][mask],
                    arrays['state'][mask],
                    bootstrap=args.bootstrap,
                    rng=rng,
                ),
            }
            for name in scorer_names:
                scorer_entry = {}
                for metric in (
                    'component_coverage',
                    'successful_component_coverage',
                    'elite_recall',
                    'success_recall',
                    'mean_error',
                    'logstd_error',
                ):
                    values = arrays[f'{name}_{metric}'][mask]
                    states = arrays['state'][mask]
                    finite = np.isfinite(values)
                    scorer_entry[metric] = paired_bootstrap(
                        values[finite],
                        states[finite],
                        bootstrap=args.bootstrap,
                        rng=rng,
                    )
                entry[name] = scorer_entry
            report['components'][key] = entry

    (args.out / 'report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
