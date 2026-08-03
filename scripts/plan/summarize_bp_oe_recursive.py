"""Pool recursive BP-OE shards and report paired deployable comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=20261120)
    return parser.parse_args()


def paired_interval(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def main() -> None:
    args = parse_args()
    archives = []
    audits = []
    for path in args.shards:
        with np.load(path, allow_pickle=False) as archive:
            archives.append(
                {key: np.asarray(archive[key]) for key in archive.files}
            )
        audits.append(json.loads(str(archives[-1]['audit'].item())))

    methods = archives[0]['method_names'].astype(str).tolist()
    selectors = archives[0]['selector_names'].astype(str).tolist()
    for archive in archives[1:]:
        if archive['method_names'].astype(str).tolist() != methods:
            raise ValueError('method mismatch across shards')
        if archive['selector_names'].astype(str).tolist() != selectors:
            raise ValueError('selector mismatch across shards')

    rows = np.concatenate([archive['rows'] for archive in archives])
    if len(np.unique(rows)) != len(rows):
        raise ValueError('duplicate eval rows across shards')
    order = np.argsort(rows)

    state_keys = [
        'population_min_true',
        'population_success',
        'population_mean_true',
        'branch_count_history',
        'branch_distance_history',
        'final_branch_true',
        'final_branch_success',
        'selected_true',
        'selected_success',
        'selected_index',
        'selected_modes',
    ]
    pooled = {
        key: np.concatenate(
            [archive[key] for archive in archives],
            axis=0,
        )[order]
        for key in state_keys
    }
    rows = rows[order]

    method_index = {name: index for index, name in enumerate(methods)}
    selector_index = {
        name: index for index, name in enumerate(selectors)
    }
    rng = np.random.default_rng(args.seed)
    final = {}
    for method, method_i in method_index.items():
        final[method] = {}
        for selector, selector_i in selector_index.items():
            cost = pooled['selected_true'][:, method_i, selector_i]
            success = pooled['selected_success'][:, method_i, selector_i]
            final[method][selector] = {
                'mean_true_cost': float(cost.mean()),
                'median_true_cost': float(np.median(cost)),
                'success_rate': float(success.mean()),
                'oracle_branch_match_rate': float(
                    np.mean(
                        pooled['selected_index'][
                            :, method_i, selector_i
                        ]
                        == pooled['selected_index'][
                            :,
                            method_i,
                            selector_index['oracle_union'],
                        ]
                    )
                ),
            }

    baseline = pooled['selected_true'][
        :, method_index['k3_1x300'], selector_index['primary']
    ]
    baseline_success = pooled['selected_success'][
        :, method_index['k3_1x300'], selector_index['primary']
    ].astype(np.float64)
    multistart = pooled['selected_true'][
        :, method_index['k3_2x150'], selector_index['primary']
    ]
    multistart_success = pooled['selected_success'][
        :, method_index['k3_2x150'], selector_index['primary']
    ].astype(np.float64)
    comparisons = {}
    candidates = [
        ('k3_2x150', 'primary'),
        ('bp_matched', 'primary'),
        ('bp_matched', 'k3'),
        ('bp_matched', 'k10'),
        ('bp_matched', 'consensus'),
        ('bp_matched', 'oracle_union'),
        ('bp_full', 'primary'),
        ('bp_full', 'k3'),
        ('bp_full', 'k10'),
        ('bp_full', 'consensus'),
        ('bp_full', 'oracle_union'),
    ]
    if 'bp_sparse_matched' in method_index:
        candidates.extend(
            [
                ('bp_sparse_matched', 'primary'),
                ('bp_sparse_matched', 'k3'),
                ('bp_sparse_matched', 'k10'),
                ('bp_sparse_matched', 'consensus'),
                ('bp_sparse_matched', 'oracle_union'),
            ]
        )
    for method, selector in candidates:
        cost = pooled['selected_true'][
            :, method_index[method], selector_index[selector]
        ]
        success = pooled['selected_success'][
            :, method_index[method], selector_index[selector]
        ].astype(np.float64)
        delta = cost - baseline
        key = f'{method}/{selector}'
        comparisons[key] = {
            'delta_true_cost_vs_k3_1x300': float(delta.mean()),
            'median_delta_true_cost_vs_k3_1x300': float(
                np.median(delta)
            ),
            'trimmed_delta_true_cost_vs_k3_1x300': float(
                np.sort(delta)[1:-1].mean()
                if len(delta) >= 5
                else delta.mean()
            ),
            'delta_true_cost_ci95': paired_interval(
                delta,
                samples=args.bootstrap,
                rng=rng,
            ),
            'win_tie_loss': [
                int(np.sum(delta < -1e-9)),
                int(np.sum(np.abs(delta) <= 1e-9)),
                int(np.sum(delta > 1e-9)),
            ],
            'delta_success_vs_k3_1x300': float(
                (success - baseline_success).mean()
            ),
        }
        if method.startswith('bp_'):
            delta_multistart = cost - multistart
            comparisons[key].update(
                {
                    'delta_true_cost_vs_k3_2x150': float(
                        delta_multistart.mean()
                    ),
                    'delta_true_cost_vs_k3_2x150_ci95': paired_interval(
                        delta_multistart,
                        samples=args.bootstrap,
                        rng=rng,
                    ),
                    'win_tie_loss_vs_k3_2x150': [
                        int(np.sum(delta_multistart < -1e-9)),
                        int(np.sum(np.abs(delta_multistart) <= 1e-9)),
                        int(np.sum(delta_multistart > 1e-9)),
                    ],
                    'delta_success_vs_k3_2x150': float(
                        (success - multistart_success).mean()
                    ),
                }
            )

    rounds = {}
    for method, method_i in method_index.items():
        rounds[method] = {
            'mean_population_min_true': pooled[
                'population_min_true'
            ][:, method_i].mean(axis=0).tolist(),
            'population_success_coverage': pooled[
                'population_success'
            ][:, method_i].mean(axis=0).tolist(),
            'mean_branch_distance': np.nanmean(
                pooled['branch_distance_history'][:, method_i],
                axis=0,
            ).tolist(),
        }

    report = {
        'version': 1,
        'num_states': int(len(rows)),
        'rows': rows.tolist(),
        'source_audits': audits,
        'methods': methods,
        'selectors': selectors,
        'final': final,
        'paired_comparisons': comparisons,
        'rounds': rounds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(comparisons, indent=2, sort_keys=True))
    print(f'pooled recursive report -> {args.out}')


if __name__ == '__main__':
    main()
