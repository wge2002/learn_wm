"""Summarize counterfactual global and component-wise CEM refits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--refit', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260721)
    return parser.parse_args()


def bootstrap_summary(
    values: np.ndarray,
    *,
    indices: np.ndarray,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {'mean': None, 'ci95': [None, None], 'n': 0}
    sampled = values[indices]
    counts = np.sum(np.isfinite(sampled), axis=1)
    distribution = np.divide(
        np.nansum(sampled, axis=1),
        counts,
        out=np.full(len(sampled), np.nan),
        where=counts > 0,
    )
    return {
        'mean': float(np.nanmean(values)),
        'ci95': np.nanquantile(distribution, [0.025, 0.975]).tolist(),
        'n': int(np.sum(finite)),
    }


def aligned_source(
    source_path: Path,
    rows: np.ndarray,
) -> dict[str, np.ndarray]:
    fields = ('rows', 'true', 'success', 'returned_true', 'returned_success')
    with np.load(source_path, allow_pickle=False) as archive:
        source = {field: np.asarray(archive[field]) for field in fields}
    index_by_row = {
        int(row): index for index, row in enumerate(source['rows'])
    }
    try:
        order = np.asarray([index_by_row[int(row)] for row in rows])
    except KeyError as error:
        raise ValueError(f'refit row absent from source: {error}') from error
    return {
        key: value[order] if key != 'rows' else value[order]
        for key, value in source.items()
    }


def component_choice(
    values: np.ndarray,
    count: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    width = values.shape[-1]
    valid = np.arange(width) < count[..., None]
    masked = np.where(valid, values, np.inf)
    indices = np.argmin(masked, axis=-1)
    selected = np.take_along_axis(
        values,
        indices[..., None],
        axis=-1,
    )[..., 0]
    return indices, selected


def gather_component(
    values: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.take_along_axis(
        values,
        indices[..., None],
        axis=-1,
    )[..., 0]


def fmt(summary: dict, *, percent: bool = True) -> str:
    value = summary['mean']
    if value is None:
        return 'NA'
    return f'{100 * value:.1f}%' if percent else f'{value:.2f}'


def markdown_report(report: dict) -> str:
    lines = [
        '# Counterfactual refit audit',
        '',
        (
            f"{report['num_states']} paired states; steps "
            f"{report['steps']}."
        ),
        '',
        '## Final global-mean refits',
        '',
        '| generator path | K3 elite mean | K10 elite mean | '
        'true elite mean | stored mean |',
        '|---|---:|---:|---:|---:|',
    ]
    for generator in report['generators']:
        final = report['global_refits'][generator][-1]
        by_selector = {
            item['selector']: item for item in final['selectors']
        }
        lines.append(
            f"| {generator} | "
            f"{fmt(by_selector['k3']['success'])} | "
            f"{fmt(by_selector['k10']['success'])} | "
            f"{fmt(by_selector['true']['success'])} | "
            f"{fmt(by_selector['stored']['success'])} |"
        )
    lines.extend(
        [
            '',
            '## Final component-wise refits',
            '',
            '| path | elite source | global mean | own/true-selected '
            'component mean | any component succeeds |',
            '|---|---|---:|---:|---:|',
        ]
    )
    for generator in report['generators']:
        final = report['component_refits'][generator][-1]
        global_final = report['global_refits'][generator][-1]
        global_by_selector = {
            item['selector']: item for item in global_final['selectors']
        }
        for item in final['selectors']:
            lines.append(
                f"| {generator} | {item['selector']} | "
                f"{fmt(global_by_selector[item['selector']]['success'])} | "
                f"{fmt(item['native_selected_success'])} | "
                f"{fmt(item['any_component_success'])} |"
            )
    mechanism = report['final_mechanism']
    lines.extend(
        [
            '',
            '## Final K10-path decomposition',
            '',
            (
                f"K3-global rescues K10-global failures: "
                f"{mechanism['k3_global_rescues_k10_global_count']} states."
            ),
            '',
            (
                f"True-global fails but a true-elite component mean succeeds: "
                f"{mechanism['true_component_rescues_true_global_count']} "
                'states.'
            ),
            '',
            (
                f"Candidate population contains no success: "
                f"{mechanism['proposal_support_failure_count']} states."
            ),
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    with np.load(args.refit, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    rows = result['rows'].astype(np.int64)
    source = aligned_source(args.source, rows)
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    steps = result['steps'].astype(int).tolist()
    global_selectors = result['global_selectors'].astype(str).tolist()
    component_selectors = result['component_selectors'].astype(str).tolist()
    num_states, num_generators, num_rounds = result['global_true'].shape[:3]
    if not np.array_equal(rows, source['rows']):
        raise ValueError('source/refit row alignment failed')

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        num_states,
        size=(args.bootstrap, num_states),
    )
    report = {
        'version': 1,
        'refit': str(args.refit.resolve()),
        'source': str(args.source.resolve()),
        'num_states': num_states,
        'generators': generators,
        'scorers': scorers,
        'steps': steps,
        'global_selectors': global_selectors,
        'component_selectors': component_selectors,
        'global_refits': {},
        'component_refits': {},
    }

    component_derived: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
    for generator_i, generator in enumerate(generators):
        report['global_refits'][generator] = []
        report['component_refits'][generator] = []
        for round_i, step in enumerate(steps):
            global_row = {'step': step, 'selectors': []}
            for selector_i, selector in enumerate(global_selectors):
                global_row['selectors'].append(
                    {
                        'selector': selector,
                        'success': bootstrap_summary(
                            result['global_success'][
                                :, generator_i, round_i, selector_i
                            ],
                            indices=bootstrap_indices,
                        ),
                        'true_cost': bootstrap_summary(
                            result['global_true'][
                                :, generator_i, round_i, selector_i
                            ],
                            indices=bootstrap_indices,
                        ),
                    }
                )
            report['global_refits'][generator].append(global_row)

            component_row = {'step': step, 'selectors': []}
            for selector_i, selector in enumerate(component_selectors):
                counts = result['component_count'][
                    :, generator_i, round_i, selector_i
                ]
                true_values = result['component_true'][
                    :, generator_i, round_i, selector_i
                ]
                success_values = result['component_success'][
                    :, generator_i, round_i, selector_i
                ]
                oracle_index, oracle_true = component_choice(
                    true_values,
                    counts,
                )
                oracle_success = gather_component(
                    success_values,
                    oracle_index,
                )
                valid = (
                    np.arange(true_values.shape[-1])[None, :]
                    < counts[:, None]
                )
                any_success = np.any(success_values & valid, axis=-1)
                scorer_entries = {}
                chosen_success = {}
                chosen_true = {}
                for scorer_i, scorer in enumerate(scorers):
                    predicted = result['component_pred'][
                        :, generator_i, round_i, selector_i, scorer_i
                    ]
                    model_index, _ = component_choice(
                        predicted,
                        counts,
                    )
                    model_success = gather_component(
                        success_values,
                        model_index,
                    )
                    model_true = gather_component(
                        true_values,
                        model_index,
                    )
                    chosen_success[scorer] = model_success
                    chosen_true[scorer] = model_true
                    scorer_entries[scorer] = {
                        'selected_success': bootstrap_summary(
                            model_success,
                            indices=bootstrap_indices,
                        ),
                        'selected_true_cost': bootstrap_summary(
                            model_true,
                            indices=bootstrap_indices,
                        ),
                    }
                native_scorer = (
                    scorers[0]
                    if selector == 'k3'
                    else scorers[1]
                    if selector == 'k10'
                    else None
                )
                native_success = (
                    chosen_success[native_scorer]
                    if native_scorer is not None
                    else oracle_success
                )
                native_true = (
                    chosen_true[native_scorer]
                    if native_scorer is not None
                    else oracle_true
                )
                component_row['selectors'].append(
                    {
                        'selector': selector,
                        'component_count': bootstrap_summary(
                            counts,
                            indices=bootstrap_indices,
                        ),
                        'native_selected_success': bootstrap_summary(
                            native_success,
                            indices=bootstrap_indices,
                        ),
                        'native_selected_true_cost': bootstrap_summary(
                            native_true,
                            indices=bootstrap_indices,
                        ),
                        'oracle_min_cost_success': bootstrap_summary(
                            oracle_success,
                            indices=bootstrap_indices,
                        ),
                        'oracle_min_true_cost': bootstrap_summary(
                            oracle_true,
                            indices=bootstrap_indices,
                        ),
                        'any_component_success': bootstrap_summary(
                            any_success,
                            indices=bootstrap_indices,
                        ),
                        'scorer_selection': scorer_entries,
                    }
                )
                component_derived[
                    (generator_i, round_i, selector_i)
                ] = {
                    'oracle_success': oracle_success,
                    'any_success': any_success,
                    'native_success': native_success,
                    **{
                        f'{scorer}_success': values
                        for scorer, values in chosen_success.items()
                    },
                }
            report['component_refits'][generator].append(component_row)

    final_round = num_rounds - 1
    k3_global_i = global_selectors.index('k3')
    k10_global_i = global_selectors.index('k10')
    true_global_i = global_selectors.index('true')
    stored_i = global_selectors.index('stored')
    k10_path_i = 1
    k3_global_success = result['global_success'][
        :, k10_path_i, final_round, k3_global_i
    ]
    k10_global_success = result['global_success'][
        :, k10_path_i, final_round, k10_global_i
    ]
    true_global_success = result['global_success'][
        :, k10_path_i, final_round, true_global_i
    ]
    stored_success = result['global_success'][
        :, k10_path_i, final_round, stored_i
    ]
    true_component_i = component_selectors.index('true')
    true_component = component_derived[
        (k10_path_i, final_round, true_component_i)
    ]
    candidate_support = source['success'][
        :, k10_path_i, final_round
    ].any(axis=-1)
    report['final_mechanism'] = {
        'k3_global_rescues_k10_global_count': int(
            np.sum(k3_global_success & ~k10_global_success)
        ),
        'k10_global_rescues_k3_global_count': int(
            np.sum(k10_global_success & ~k3_global_success)
        ),
        'k3_global_minus_k10_global_success': bootstrap_summary(
            k3_global_success.astype(np.float64)
            - k10_global_success.astype(np.float64),
            indices=bootstrap_indices,
        ),
        'true_global_rescues_stored_count': int(
            np.sum(true_global_success & ~stored_success)
        ),
        'true_component_rescues_true_global_count': int(
            np.sum(true_component['any_success'] & ~true_global_success)
        ),
        'true_component_oracle_rescues_true_global_count': int(
            np.sum(true_component['oracle_success'] & ~true_global_success)
        ),
        'proposal_support_failure_count': int(np.sum(~candidate_support)),
        'proposal_support_success': bootstrap_summary(
            candidate_support,
            indices=bootstrap_indices,
        ),
    }

    args.out.mkdir(parents=True, exist_ok=True)
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
