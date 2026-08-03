"""Audit scorer fidelity along paired, self-induced CEM query paths.

The paired query bank contains two CEM generators (K3 and K10), both model
scores on every sampled action, and simulator outcomes.  This audit separates
three effects that a frozen-population comparison confounds:

1. scorer fidelity on a fixed candidate population;
2. proposal support retained by each scorer's own recursive CEM path; and
3. conversion of successful elite witnesses into a successful returned mean.

It also measures connected components of the true top-rate set in each local
proposal-whitened action graph.  Those component counts are a sampled support
diagnostic, not an identification of the same basin across different rounds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oe_basin_topology_audit import component_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--elite', type=int, default=30)
    parser.add_argument('--neighbors', type=int, default=12)
    parser.add_argument('--bootstrap', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260721)
    return parser.parse_args()


def stable_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = stable_ranks(left)
    right_rank = stable_ranks(right)
    left_rank -= left_rank.mean()
    right_rank -= right_rank.mean()
    denominator = np.sqrt(
        np.sum(np.square(left_rank))
        * np.sum(np.square(right_rank))
    )
    if denominator <= 0:
        return 0.0
    return float(np.sum(left_rank * right_rank) / denominator)


def symmetric_knn_fast(
    points: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    count = len(points)
    if neighbors < 1 or neighbors >= count:
        raise ValueError('neighbors must be in [1, N-1]')
    squared_norm = np.sum(np.square(points), axis=1)
    squared = (
        squared_norm[:, None]
        + squared_norm[None]
        - 2.0 * (points @ points.T)
    )
    np.fill_diagonal(squared, np.inf)
    nearest = np.argpartition(
        squared,
        kth=neighbors - 1,
        axis=1,
    )[:, :neighbors]
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[np.arange(count)[:, None], nearest] = True
    adjacency |= adjacency.T
    return adjacency


def bootstrap_summary(
    values: np.ndarray,
    *,
    indices: np.ndarray,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {
            'mean': None,
            'ci95': [None, None],
            'n': 0,
        }
    # Resampling is by paired state.  NaNs remain missing within a replicate;
    # this is used only for metrics whose denominator does not exist in every
    # state (for example success recall when no successful candidate exists).
    sampled = values[indices]
    counts = np.sum(np.isfinite(sampled), axis=1)
    sums = np.nansum(sampled, axis=1)
    distribution = np.divide(
        sums,
        counts,
        out=np.full(len(sums), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return {
        'mean': float(np.nanmean(values)),
        'ci95': np.nanquantile(distribution, [0.025, 0.975]).tolist(),
        'n': int(np.sum(finite)),
    }


def load_source(path: Path) -> dict[str, np.ndarray]:
    required = (
        'rows',
        'candidates',
        'pred',
        'true',
        'success',
        'topk_indices',
        'mean',
        'prev_mean',
        'prev_var',
        'returned_true',
        'returned_success',
        'generators',
        'scorers',
        'steps',
    )
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in required}
    if data['candidates'].ndim != 6:
        raise ValueError('expected candidates [state,generator,round,N,H,A]')
    if data['pred'].shape[:3] != data['true'].shape[:3]:
        raise ValueError('pred and true leading dimensions differ')
    if data['pred'].shape[-1] != data['true'].shape[-1]:
        raise ValueError('pred and true population sizes differ')
    if len(data['generators']) != len(data['scorers']):
        raise ValueError('paired audit expects aligned generators/scorers')
    if not np.array_equal(data['generators'], data['scorers']):
        raise ValueError('generator and scorer names/order must align')
    return data


def markdown_report(report: dict) -> str:
    lines = [
        '# Paired basin-lineage audit',
        '',
        (
            f"Source: `{report['source']}`; "
            f"{report['num_states']} states, "
            f"steps {report['steps']}."
        ),
        '',
        '## Recursive outcomes',
        '',
        '| generator | step | returned success | population any-success | '
        'population success rate | best true cost |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for generator in report['generators']:
        for row in report['recursive_outcomes'][generator]:
            lines.append(
                f"| {generator} | {row['step']} | "
                f"{row['returned_success']['mean']:.3f} | "
                f"{row['population_any_success']['mean']:.3f} | "
                f"{row['population_success_rate']['mean']:.3f} | "
                f"{row['population_best_true']['mean']:.2f} |"
            )
    lines.extend(
        [
            '',
            '## Fixed-population scorer fidelity',
            '',
            '| generator path | step | scorer | true-elite recall | '
            'Spearman | selected success rate | selected any-success |',
            '|---|---:|---|---:|---:|---:|---:|',
        ]
    )
    for generator in report['generators']:
        for row in report['scorer_fidelity'][generator]:
            lines.append(
                f"| {generator} | {row['step']} | {row['scorer']} | "
                f"{row['elite_recall']['mean']:.3f} | "
                f"{row['spearman']['mean']:.3f} | "
                f"{row['selected_success_rate']['mean']:.3f} | "
                f"{row['selected_any_success']['mean']:.3f} |"
            )
    final = report['final_pair']
    interaction = report['scorer_by_generator_path_interaction'][-1]
    lines.extend(
        [
            '',
            '## Final paired comparison',
            '',
            (
                f"K10 − K3 returned-success delta: "
                f"{final['k10_minus_k3_returned_success']['mean']:+.3f} "
                f"(95% state bootstrap "
                f"[{final['k10_minus_k3_returned_success']['ci95'][0]:+.3f}, "
                f"{final['k10_minus_k3_returned_success']['ci95'][1]:+.3f}])."
            ),
            '',
            (
                'Discordant states: '
                f"K3-only {final['outcome_counts']['k3_only']}, "
                f"K10-only {final['outcome_counts']['k10_only']}; "
                f"both {final['outcome_counts']['both']}, "
                f"neither {final['outcome_counts']['neither']}."
            ),
            '',
            (
                'Final true-elite-recall path interaction '
                '[(K10−K3 scorer on K10 path) − '
                '(K10−K3 scorer on K3 path)]: '
                f"{interaction['elite_recall']['mean']:+.3f} "
                f"(95% state bootstrap "
                f"[{interaction['elite_recall']['ci95'][0]:+.3f}, "
                f"{interaction['elite_recall']['ci95'][1]:+.3f}])."
            ),
            '',
            '## Elite-to-mean conversion at the final step',
            '',
            '| generator | elite has success | returned success | '
            'witness but returned failure | conversion given witness |',
            '|---|---:|---:|---:|---:|',
        ]
    )
    for generator in report['generators']:
        row = report['elite_to_mean'][generator]
        lines.append(
            f"| {generator} | {row['elite_any_success']['mean']:.3f} | "
            f"{row['returned_success']['mean']:.3f} | "
            f"{row['witness_but_return_failure']['mean']:.3f} | "
            f"{row['conversion_given_witness']:.3f} |"
        )
    lines.extend(
        [
            '',
            '## Interpretation guardrail',
            '',
            (
                'Connected-component results are local sampled-support '
                'diagnostics. Components were not matched across rounds, so '
                'their count trajectory alone is not a basin-identity claim.'
            ),
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    data = load_source(args.source)
    true = data['true'].astype(np.float64)
    success = data['success'].astype(bool)
    pred = data['pred'].astype(np.float64)
    candidates = data['candidates'].astype(np.float32)
    prev_mean = data['prev_mean'].astype(np.float32)
    prev_std = np.maximum(data['prev_var'].astype(np.float32), 1e-4)
    returned_true = data['returned_true'].astype(np.float64)
    returned_success = data['returned_success'].astype(bool)
    num_states, num_generators, num_rounds, population_size = true.shape
    elite = args.elite
    if elite < 1 or elite >= population_size:
        raise ValueError('elite must be in [1, population_size-1]')

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        num_states,
        size=(args.bootstrap, num_states),
    )
    scorer_metrics = {
        name: np.full(
            (num_states, num_generators, num_rounds, num_generators),
            np.nan,
            dtype=np.float64,
        )
        for name in (
            'elite_recall',
            'spearman',
            'selected_true_mean',
            'true_elite_regret',
            'selected_success_rate',
            'selected_success_recall',
            'selected_any_success',
            'component_coverage',
            'successful_component_coverage',
        )
    }
    population_metrics = {
        name: np.full(
            (num_states, num_generators, num_rounds),
            np.nan,
            dtype=np.float64,
        )
        for name in (
            'any_success',
            'success_rate',
            'best_true',
            'true_elite_mean',
            'true_component_count',
            'successful_true_component_count',
        )
    }

    for state_i in range(num_states):
        for generator_i in range(num_generators):
            for round_i in range(num_rounds):
                state_true = true[state_i, generator_i, round_i]
                state_success = success[state_i, generator_i, round_i]
                true_order = np.argsort(state_true, kind='stable')[:elite]
                true_mask = np.zeros(population_size, dtype=bool)
                true_mask[true_order] = True
                population_metrics['any_success'][
                    state_i, generator_i, round_i
                ] = np.any(state_success)
                population_metrics['success_rate'][
                    state_i, generator_i, round_i
                ] = np.mean(state_success)
                population_metrics['best_true'][
                    state_i, generator_i, round_i
                ] = np.min(state_true)
                population_metrics['true_elite_mean'][
                    state_i, generator_i, round_i
                ] = np.mean(state_true[true_order])

                normalized = (
                    (
                        candidates[state_i, generator_i, round_i]
                        - prev_mean[state_i, generator_i, round_i][None]
                    )
                    / prev_std[state_i, generator_i, round_i][None]
                ).reshape(population_size, -1)
                scale = np.maximum(normalized.std(axis=0), 1e-6)
                points = normalized / scale
                adjacency = symmetric_knn_fast(
                    points,
                    neighbors=args.neighbors,
                )
                true_labels = component_labels(adjacency, true_mask)
                true_components = np.unique(
                    true_labels[true_labels >= 0]
                )
                successful_components = [
                    component
                    for component in true_components
                    if np.any(
                        state_success & (true_labels == component)
                    )
                ]
                population_metrics['true_component_count'][
                    state_i, generator_i, round_i
                ] = len(true_components)
                population_metrics['successful_true_component_count'][
                    state_i, generator_i, round_i
                ] = len(successful_components)

                for scorer_i in range(num_generators):
                    state_pred = pred[
                        state_i, generator_i, round_i, scorer_i
                    ]
                    selected = np.argsort(
                        state_pred,
                        kind='stable',
                    )[:elite]
                    selected_mask = np.zeros(population_size, dtype=bool)
                    selected_mask[selected] = True
                    overlap = np.sum(selected_mask & true_mask)
                    scorer_metrics['elite_recall'][
                        state_i, generator_i, round_i, scorer_i
                    ] = overlap / elite
                    scorer_metrics['spearman'][
                        state_i, generator_i, round_i, scorer_i
                    ] = spearman(state_pred, state_true)
                    selected_true_mean = np.mean(state_true[selected])
                    scorer_metrics['selected_true_mean'][
                        state_i, generator_i, round_i, scorer_i
                    ] = selected_true_mean
                    scorer_metrics['true_elite_regret'][
                        state_i, generator_i, round_i, scorer_i
                    ] = (
                        selected_true_mean - np.mean(state_true[true_order])
                    )
                    scorer_metrics['selected_success_rate'][
                        state_i, generator_i, round_i, scorer_i
                    ] = np.mean(state_success[selected])
                    scorer_metrics['selected_any_success'][
                        state_i, generator_i, round_i, scorer_i
                    ] = np.any(state_success[selected])
                    success_count = int(np.sum(state_success))
                    scorer_metrics['selected_success_recall'][
                        state_i, generator_i, round_i, scorer_i
                    ] = (
                        np.sum(selected_mask & state_success) / success_count
                        if success_count
                        else np.nan
                    )
                    hits = [
                        np.any(
                            selected_mask & (true_labels == component)
                        )
                        for component in true_components
                    ]
                    scorer_metrics['component_coverage'][
                        state_i, generator_i, round_i, scorer_i
                    ] = float(np.mean(hits)) if hits else 0.0
                    successful_hits = [
                        np.any(
                            selected_mask & (true_labels == component)
                        )
                        for component in successful_components
                    ]
                    scorer_metrics['successful_component_coverage'][
                        state_i, generator_i, round_i, scorer_i
                    ] = (
                        float(np.mean(successful_hits))
                        if successful_hits
                        else np.nan
                    )

        if (state_i + 1) % 10 == 0:
            print(
                f'paired lineage audit {state_i + 1}/{num_states} states',
                flush=True,
            )

    generators = [str(item) for item in data['generators']]
    scorers = [str(item) for item in data['scorers']]
    steps = [int(item) for item in data['steps']]
    report = {
        'version': 1,
        'source': str(args.source.resolve()),
        'num_states': num_states,
        'num_generators': num_generators,
        'num_rounds': num_rounds,
        'population_size': population_size,
        'elite': elite,
        'neighbors': args.neighbors,
        'generators': generators,
        'scorers': scorers,
        'steps': steps,
        'recursive_outcomes': {},
        'scorer_fidelity': {},
        'scorer_delta_k10_minus_k3': {},
        'sampled_support_topology': {},
        'elite_to_mean': {},
    }
    for generator_i, generator in enumerate(generators):
        report['recursive_outcomes'][generator] = []
        report['scorer_fidelity'][generator] = []
        report['scorer_delta_k10_minus_k3'][generator] = []
        report['sampled_support_topology'][generator] = []
        for round_i, step in enumerate(steps):
            report['recursive_outcomes'][generator].append(
                {
                    'step': step,
                    'returned_success': bootstrap_summary(
                        returned_success[:, generator_i, round_i],
                        indices=bootstrap_indices,
                    ),
                    'returned_true': bootstrap_summary(
                        returned_true[:, generator_i, round_i],
                        indices=bootstrap_indices,
                    ),
                    'population_any_success': bootstrap_summary(
                        population_metrics['any_success'][
                            :, generator_i, round_i
                        ],
                        indices=bootstrap_indices,
                    ),
                    'population_success_rate': bootstrap_summary(
                        population_metrics['success_rate'][
                            :, generator_i, round_i
                        ],
                        indices=bootstrap_indices,
                    ),
                    'population_best_true': bootstrap_summary(
                        population_metrics['best_true'][
                            :, generator_i, round_i
                        ],
                        indices=bootstrap_indices,
                    ),
                }
            )
            topology_row = {
                'step': step,
                'true_component_count': bootstrap_summary(
                    population_metrics['true_component_count'][
                        :, generator_i, round_i
                    ],
                    indices=bootstrap_indices,
                ),
                'successful_true_component_count': bootstrap_summary(
                    population_metrics['successful_true_component_count'][
                        :, generator_i, round_i
                    ],
                    indices=bootstrap_indices,
                ),
            }
            for scorer_i, scorer in enumerate(scorers):
                row = {'step': step, 'scorer': scorer}
                for metric in scorer_metrics:
                    row[metric] = bootstrap_summary(
                        scorer_metrics[metric][
                            :, generator_i, round_i, scorer_i
                        ],
                        indices=bootstrap_indices,
                    )
                report['scorer_fidelity'][generator].append(row)
                topology_row[scorer] = {
                    metric: row[metric]
                    for metric in (
                        'component_coverage',
                        'successful_component_coverage',
                    )
                }
            delta_row = {'step': step}
            for metric in scorer_metrics:
                delta = (
                    scorer_metrics[metric][
                        :, generator_i, round_i, 1
                    ]
                    - scorer_metrics[metric][
                        :, generator_i, round_i, 0
                    ]
                )
                delta_row[metric] = bootstrap_summary(
                    delta,
                    indices=bootstrap_indices,
                )
            report['scorer_delta_k10_minus_k3'][generator].append(
                delta_row
            )
            report['sampled_support_topology'][generator].append(
                topology_row
            )

        final_round = num_rounds - 1
        own_selected_any = scorer_metrics['selected_any_success'][
            :, generator_i, final_round, generator_i
        ].astype(bool)
        final_return = returned_success[:, generator_i, final_round]
        witness_failure = own_selected_any & ~final_return
        report['elite_to_mean'][generator] = {
            'elite_any_success': bootstrap_summary(
                own_selected_any,
                indices=bootstrap_indices,
            ),
            'returned_success': bootstrap_summary(
                final_return,
                indices=bootstrap_indices,
            ),
            'witness_but_return_failure': bootstrap_summary(
                witness_failure,
                indices=bootstrap_indices,
            ),
            'witness_but_return_failure_count': int(
                np.sum(witness_failure)
            ),
            'conversion_given_witness': float(
                np.sum(own_selected_any & final_return)
                / max(np.sum(own_selected_any), 1)
            ),
            'population_witness_but_return_failure_count': int(
                np.sum(
                    population_metrics['any_success'][
                        :, generator_i, final_round
                    ].astype(bool)
                    & ~final_return
                )
            ),
        }

    if num_generators == 2:
        final_k3 = returned_success[:, 0, -1]
        final_k10 = returned_success[:, 1, -1]
        report['final_pair'] = {
            'k10_minus_k3_returned_success': bootstrap_summary(
                final_k10.astype(np.float64)
                - final_k3.astype(np.float64),
                indices=bootstrap_indices,
            ),
            'k10_minus_k3_returned_true': bootstrap_summary(
                returned_true[:, 1, -1] - returned_true[:, 0, -1],
                indices=bootstrap_indices,
            ),
            'k10_minus_k3_population_success_rate': bootstrap_summary(
                population_metrics['success_rate'][:, 1, -1]
                - population_metrics['success_rate'][:, 0, -1],
                indices=bootstrap_indices,
            ),
            'k10_minus_k3_population_best_true': bootstrap_summary(
                population_metrics['best_true'][:, 1, -1]
                - population_metrics['best_true'][:, 0, -1],
                indices=bootstrap_indices,
            ),
            'outcome_counts': {
                'both': int(np.sum(final_k3 & final_k10)),
                'k3_only': int(np.sum(final_k3 & ~final_k10)),
                'k10_only': int(np.sum(~final_k3 & final_k10)),
                'neither': int(np.sum(~final_k3 & ~final_k10)),
            },
        }
        report['scorer_by_generator_path_interaction'] = []
        for round_i, step in enumerate(steps):
            interaction_row = {'step': step}
            for metric in scorer_metrics:
                # Difference in differences:
                #   (K10 - K3 scorer on the K10-generated population)
                # - (K10 - K3 scorer on the K3-generated population).
                # A negative tail-fidelity value means K10's relative
                # advantage is specifically lost on its own induced path.
                interaction = (
                    scorer_metrics[metric][:, 1, round_i, 1]
                    - scorer_metrics[metric][:, 1, round_i, 0]
                    - scorer_metrics[metric][:, 0, round_i, 1]
                    + scorer_metrics[metric][:, 0, round_i, 0]
                )
                interaction_row[metric] = bootstrap_summary(
                    interaction,
                    indices=bootstrap_indices,
                )
            report['scorer_by_generator_path_interaction'].append(
                interaction_row
            )

    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out / 'raw_metrics.npz',
        rows=data['rows'],
        generators=data['generators'],
        scorers=data['scorers'],
        steps=data['steps'],
        returned_true=returned_true,
        returned_success=returned_success,
        **{
            f'population_{name}': value
            for name, value in population_metrics.items()
        },
        **{
            f'scorer_{name}': value
            for name, value in scorer_metrics.items()
        },
    )
    (args.out / 'report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    (args.out / 'REPORT.md').write_text(
        markdown_report(report),
        encoding='utf-8',
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
