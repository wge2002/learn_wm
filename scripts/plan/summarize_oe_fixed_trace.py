"""Summarize state-cross-fitted OE fixed-trace training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'runs',
        type=Path,
        nargs='+',
        help='Run directories containing audit.json and metrics.json.',
    )
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20_260_719)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'refusing to write empty CSV: {path}')
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    runs = []
    for run_dir in args.runs:
        audit = load_json(run_dir / 'audit.json')
        metrics = load_json(run_dir / 'metrics.json')['history']
        if not metrics:
            raise ValueError(f'empty metric history: {run_dir}')
        runs.append(
            {
                'path': str(run_dir.resolve()),
                'name': audit['run_name'],
                'audit': audit,
                'history': metrics,
                'epochs': {int(row['epoch']): row for row in metrics},
            }
        )

    reference = runs[0]['audit']
    invariant_keys = (
        'source_sha256',
        'base_policy',
        'source_generator',
        'selected_steps',
        'num_candidates',
        'topk',
        'learning_rate',
        'temperature',
        'boundary_weight',
        'mean_weight',
        'logstd_weight',
        'anchor_weight',
        'relative_update_weight',
        'calibrate_elite_mass',
        'replay_sha256',
        'replay_weight',
        'replay_batch_size',
        'trainable_modules',
    )
    for run in runs[1:]:
        for key in invariant_keys:
            if run['audit'].get(key) != reference.get(key):
                raise ValueError(
                    f'run mismatch for {key}: '
                    f'{run["name"]}={run["audit"].get(key)!r}, '
                    f'reference={reference.get(key)!r}'
                )

    state_owners: dict[int, str] = {}
    for run in runs:
        for state in run['audit']['val_states']:
            if state in state_owners:
                raise ValueError(
                    f'validation state {state} appears in both '
                    f'{state_owners[state]} and {run["name"]}'
                )
            state_owners[state] = run['name']

    common_epochs = sorted(
        set.intersection(*(set(run['epochs']) for run in runs))
    )
    if not common_epochs:
        raise ValueError('runs have no common evaluation epoch')
    metric_names = list(runs[0]['history'][0]['val'])
    has_state_metrics = all(
        'val_state_metrics' in run['epochs'][epoch]
        for run in runs
        for epoch in common_epochs
    )
    has_replay_metrics = all(
        run['epochs'][epoch].get('replay_validation_mse') is not None
        for run in runs
        for epoch in common_epochs
    )

    fold_rows = []
    aggregate_rows = []
    heldout_state_rows = []
    for epoch in common_epochs:
        aggregate: dict[str, float | int] = {
            'epoch': epoch,
            'n_folds': len(runs),
            'n_heldout_states': len(state_owners),
        }
        total_weight = sum(len(run['audit']['val_states']) for run in runs)
        for metric in metric_names:
            aggregate[metric] = (
                sum(
                    len(run['audit']['val_states'])
                    * float(run['epochs'][epoch]['val'][metric])
                    for run in runs
                )
                / total_weight
            )
        if has_replay_metrics:
            aggregate['replay_validation_mse'] = float(
                np.mean(
                    [
                        run['epochs'][epoch]['replay_validation_mse']
                        for run in runs
                    ]
                )
            )
        aggregate_rows.append(aggregate)

        for run in runs:
            row: dict[str, str | int | float] = {
                'run': run['name'],
                'epoch': epoch,
                'n_val_states': len(run['audit']['val_states']),
                'val_states': ','.join(
                    str(state) for state in run['audit']['val_states']
                ),
            }
            row.update(
                {
                    metric: float(run['epochs'][epoch]['val'][metric])
                    for metric in metric_names
                }
            )
            if has_replay_metrics:
                row['replay_validation_mse'] = float(
                    run['epochs'][epoch]['replay_validation_mse']
                )
            fold_rows.append(row)
            if has_state_metrics:
                for state_row in run['epochs'][epoch]['val_state_metrics']:
                    heldout_state_rows.append(
                        {
                            'run': run['name'],
                            'epoch': epoch,
                            **state_row,
                        }
                    )

    baseline = aggregate_rows[0]
    for row in aggregate_rows:
        for metric in metric_names:
            row[f'{metric}_delta_vs_epoch0'] = float(row[metric]) - float(
                baseline[metric]
            )
        if has_replay_metrics:
            row['replay_validation_mse_delta_vs_epoch0'] = float(
                row['replay_validation_mse']
            ) - float(baseline['replay_validation_mse'])

    if has_state_metrics:
        baseline_by_state = {
            int(row['state_index']): row
            for row in heldout_state_rows
            if int(row['epoch']) == common_epochs[0]
        }
        for row in heldout_state_rows:
            baseline_state = baseline_by_state[int(row['state_index'])]
            for metric in metric_names:
                row[f'{metric}_delta_vs_epoch0'] = float(row[metric]) - float(
                    baseline_state[metric]
                )

        if args.bootstrap < 1:
            raise ValueError('--bootstrap must be positive')
        ordered_states = sorted(baseline_by_state)
        rng = np.random.default_rng(args.seed)
        sample_indices = rng.integers(
            0,
            len(ordered_states),
            size=(args.bootstrap, len(ordered_states)),
        )
        state_epoch = {
            (int(row['state_index']), int(row['epoch'])): row
            for row in heldout_state_rows
        }
        for aggregate in aggregate_rows:
            epoch = int(aggregate['epoch'])
            for metric in metric_names:
                deltas = np.asarray(
                    [
                        float(state_epoch[(state, epoch)][metric])
                        - float(baseline_by_state[state][metric])
                        for state in ordered_states
                    ],
                    dtype=np.float64,
                )
                bootstrap_means = deltas[sample_indices].mean(axis=1)
                low, high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )
                aggregate[f'{metric}_delta_ci_low'] = float(low)
                aggregate[f'{metric}_delta_ci_high'] = float(high)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / 'crossfit_metrics.csv', aggregate_rows)
    write_csv(args.out_dir / 'fold_metrics.csv', fold_rows)
    if heldout_state_rows:
        write_csv(
            args.out_dir / 'heldout_state_metrics.csv',
            heldout_state_rows,
        )
    audit = {
        'runs': [run['path'] for run in runs],
        'run_names': [run['name'] for run in runs],
        'heldout_state_owners': {
            str(state): owner for state, owner in sorted(state_owners.items())
        },
        'common_epochs': common_epochs,
        'state_level_metrics_available': has_state_metrics,
        'replay_metrics_available': has_replay_metrics,
        'bootstrap': args.bootstrap if has_state_metrics else None,
        'bootstrap_seed': args.seed if has_state_metrics else None,
        'invariants': {key: reference.get(key) for key in invariant_keys},
    }
    (args.out_dir / 'audit.json').write_text(
        json.dumps(audit, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    lines = [
        '# OE fixed-trace state cross-fit',
        '',
        f'- Folds: {len(runs)}',
        f'- Unique held-out states: {len(state_owners)}',
        f'- Trainable modules: `{", ".join(reference["trainable_modules"])}`',
        f'- Selected source steps: `{reference["selected_steps"]}`',
        *(
            [
                f'- Dynamics replay weight: '
                f'`{reference.get("replay_weight", 0.0)}`',
                f'- Dynamics replay cache: '
                f'`{reference.get("replay_sha256")}`',
            ]
            if has_replay_metrics
            else []
        ),
        '',
        'Each row pools predictions made by a model that did not train on '
        'that row’s held-out states. This is a feasibility diagnostic, not a '
        'deployable single-checkpoint or closed-loop MPC result.',
        '',
        (
            '| epoch | update cosine | Δ cosine | relative error '
            '| Δ rel. error | elite overlap | Δ overlap '
            '| selected-elite true cost | Δ true cost | replay MSE '
            '| Δ replay MSE |'
            if has_replay_metrics
            else '| epoch | update cosine | Δ cosine | relative error '
            '| Δ rel. error | elite overlap | Δ overlap '
            '| selected-elite true cost | Δ true cost |'
        ),
        (
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
            if has_replay_metrics
            else '|---:|---:|---:|---:|---:|---:|---:|---:|---:|'
        ),
    ]
    for row in aggregate_rows:
        if has_state_metrics:
            cosine_delta = (
                f'{row["update_cosine_delta_vs_epoch0"]:+.3f} '
                f'[{row["update_cosine_delta_ci_low"]:+.3f}, '
                f'{row["update_cosine_delta_ci_high"]:+.3f}]'
            )
            relative_delta = (
                f'{row["relative_update_error_delta_vs_epoch0"]:+.3f} '
                f'[{row["relative_update_error_delta_ci_low"]:+.3f}, '
                f'{row["relative_update_error_delta_ci_high"]:+.3f}]'
            )
        else:
            cosine_delta = f'{row["update_cosine_delta_vs_epoch0"]:+.3f}'
            relative_delta = (
                f'{row["relative_update_error_delta_vs_epoch0"]:+.3f}'
            )
        replay_columns = (
            f'| {row["replay_validation_mse"]:.5f} '
            f'| {row["replay_validation_mse_delta_vs_epoch0"]:+.5f} '
            if has_replay_metrics
            else ''
        )
        lines.append(
            f'| {row["epoch"]} '
            f'| {row["update_cosine"]:.3f} '
            f'| {cosine_delta} '
            f'| {row["relative_update_error"]:.3f} '
            f'| {relative_delta} '
            f'| {row["elite_overlap"]:.3f} '
            f'| {row["elite_overlap_delta_vs_epoch0"]:+.3f} '
            f'| {row["selected_elite_true_cost"]:.2f} '
            f'| {row["selected_elite_true_cost_delta_vs_epoch0"]:+.2f} '
            f'{replay_columns}|'
        )
    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print(f'summary -> {args.out_dir}')


if __name__ == '__main__':
    main()
