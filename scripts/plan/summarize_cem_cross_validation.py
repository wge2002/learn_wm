"""Summarize paired end-to-end CEM cross-validation experiments.

The evaluator samples the same ordered episode starts for runs sharing a
seed, horizon, and goal offset.  This script preserves that pairing, reports
the exact success-vector flips relative to the ordinary CEM baseline, and
uses an episode-level paired bootstrap for exploratory confidence intervals.

Expected filenames are of the form::

    cemcv_<variant>_h<horizon>_off<offset>_5090.txt

For example, ``cemcv_k3_to_k10_means_h5_off40_5090.txt`` is compared with
``cemcv_baseline_k3_h5_off40_5090.txt``.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import random
import re
from statistics import mean


RESULT_RE = re.compile(
    r'^cemcv_(?P<variant>.+)_h(?P<horizon>\d+)'
    r'_off(?P<offset>\d+)(?:_seed(?P<seed>\d+))?_5090\.txt$'
)
RATE_RE = re.compile(r"success_rate':\s*(?P<value>[0-9.]+)")
VECTOR_RE = re.compile(
    r"episode_successes': array\(\[(?P<values>.*?)\]\), 'seeds'",
    re.DOTALL,
)
TIME_RE = re.compile(r'evaluation_time:\s*(?P<value>[0-9.]+)')
SAMPLES_RE = re.compile(r'^\s*num_samples:\s*(?P<value>\d+)', re.MULTILINE)
SEED_RE = re.compile(r'^\s*seed:\s*(?P<value>\d+)', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--baseline-variant', default='baseline_k3')
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20_260_718)
    parser.add_argument('--source-label')
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--goal-offset', type=int)
    return parser.parse_args()


def required_match(
    pattern: re.Pattern[str],
    text: str,
    source: Path,
) -> re.Match[str]:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f'{source}: missing pattern {pattern.pattern!r}')
    return match


def latest_run(text: str, source: Path) -> str:
    """Return the last append-only evaluator record."""
    records = text.split('==== CONFIG ====')
    if len(records) < 2:
        raise ValueError(f'{source}: missing CONFIG record')
    record = records[-1]
    if '==== RESULTS ====' not in record:
        raise ValueError(f'{source}: last record has no RESULTS section')
    return record


def extract_braced_mapping(text: str, marker: str) -> dict | None:
    marker_index = text.find(marker)
    if marker_index < 0:
        return None
    start = text.find('{', marker_index + len(marker))
    if start < 0:
        raise ValueError(f'Found {marker!r} without a mapping')
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                value = ast.literal_eval(text[start : index + 1])
                if not isinstance(value, dict):
                    raise TypeError(f'{marker!r} did not contain a dict')
                return value
    raise ValueError(f'Unterminated mapping after {marker!r}')


def parse_result(path: Path) -> dict:
    name = RESULT_RE.match(path.name)
    if name is None:
        raise ValueError(f'Unexpected result filename: {path.name}')
    record = latest_run(path.read_text(encoding='utf-8'), path)
    vector_text = required_match(VECTOR_RE, record, path).group('values')
    successes = tuple(
        value == 'True' for value in re.findall(r'True|False', vector_text)
    )
    if not successes:
        raise ValueError(f'{path}: empty episode success vector')
    measured_rate = 100.0 * sum(successes) / len(successes)
    reported_rate = float(required_match(RATE_RE, record, path).group('value'))
    if abs(measured_rate - reported_rate) > 1e-8:
        raise ValueError(
            f'{path}: success_rate={reported_rate} disagrees with '
            f'vector rate={measured_rate}'
        )

    selection = extract_braced_mapping(record, "'cross_validation':")
    seed = int(required_match(SEED_RE, record, path).group('value'))
    filename_seed = name.group('seed')
    if filename_seed is not None and int(filename_seed) != seed:
        raise ValueError(
            f'{path}: filename seed {filename_seed} != config seed {seed}'
        )
    return {
        'variant': name.group('variant'),
        'seed': seed,
        'horizon': int(name.group('horizon')),
        'goal_offset': int(name.group('offset')),
        'num_samples': int(
            required_match(SAMPLES_RE, record, path).group('value')
        ),
        'n_episodes': len(successes),
        'n_success': sum(successes),
        'success_rate': measured_rate,
        'evaluation_time_seconds': float(
            required_match(TIME_RE, record, path).group('value')
        ),
        'num_plans': None if selection is None else selection['num_plans'],
        'mean_selected_step': (
            None if selection is None else selection.get('mean_selected_step')
        ),
        'step_counts': (
            None
            if selection is None
            else {
                int(step): int(count)
                for step, count in selection.get('step_counts', {}).items()
            }
        ),
        'successes': successes,
        'source': path.name,
    }


def quantile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap(
    deltas: list[float],
    *,
    bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    samples = [
        mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
        for _ in range(bootstrap)
    ]
    samples.sort()
    return quantile(samples, 0.025), quantile(samples, 0.975)


def exact_mcnemar(wins: int, losses: int) -> float:
    """Two-sided exact sign test on discordant pairs."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Refusing to write empty CSV: {path}')
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
    if args.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')

    result_dir = args.results.resolve()
    runs = [
        parse_result(path)
        for path in sorted(result_dir.glob('cemcv_*_h*_off*_5090.txt'))
        if RESULT_RE.match(path.name)
    ]
    runs = [
        run
        for run in runs
        if (args.horizon is None or run['horizon'] == args.horizon)
        and (
            args.goal_offset is None or run['goal_offset'] == args.goal_offset
        )
    ]
    if not runs:
        raise ValueError(f'No CEM cross-validation results in {result_dir}')

    by_cell: dict[tuple[int, int, int], list[dict]] = {}
    for run in runs:
        by_cell.setdefault(
            (run['seed'], run['horizon'], run['goal_offset']),
            [],
        ).append(run)

    comparisons = []
    comparison_deltas: dict[tuple[int, int, int, str], list[float]] = {}
    rng = random.Random(args.seed)
    for cell, cell_runs in sorted(by_cell.items()):
        baselines = [
            run for run in cell_runs if run['variant'] == args.baseline_variant
        ]
        if len(baselines) != 1:
            raise ValueError(
                f'Cell {cell} has {len(baselines)} '
                f'{args.baseline_variant!r} baselines'
            )
        baseline = baselines[0]
        for run in sorted(cell_runs, key=lambda item: item['variant']):
            if run is baseline:
                continue
            if run['n_episodes'] != baseline['n_episodes']:
                raise ValueError(
                    f'{run["source"]} and baseline have different lengths'
                )
            deltas = [
                100.0 * (int(value) - int(reference))
                for value, reference in zip(
                    run['successes'],
                    baseline['successes'],
                    strict=True,
                )
            ]
            wins = sum(delta > 0 for delta in deltas)
            losses = sum(delta < 0 for delta in deltas)
            ci_low, ci_high = paired_bootstrap(
                deltas,
                bootstrap=args.bootstrap,
                rng=rng,
            )
            comparison_deltas[
                (
                    run['seed'],
                    run['horizon'],
                    run['goal_offset'],
                    run['variant'],
                )
            ] = deltas
            comparisons.append(
                {
                    'horizon': run['horizon'],
                    'goal_offset': run['goal_offset'],
                    'seed': run['seed'],
                    'variant': run['variant'],
                    'baseline': baseline['variant'],
                    'variant_success_rate': run['success_rate'],
                    'baseline_success_rate': baseline['success_rate'],
                    'delta_success_pp': mean(deltas),
                    'delta_ci_low_pp': ci_low,
                    'delta_ci_high_pp': ci_high,
                    'wins': wins,
                    'losses': losses,
                    'ties': len(deltas) - wins - losses,
                    'mcnemar_exact_p': exact_mcnemar(wins, losses),
                    'changed_episode_indices': ','.join(
                        str(index)
                        for index, delta in enumerate(deltas)
                        if delta
                    ),
                }
            )

    aggregate_groups: dict[tuple[int, int, str, str], list[dict]] = {}
    for row in comparisons:
        aggregate_groups.setdefault(
            (
                row['horizon'],
                row['goal_offset'],
                row['variant'],
                row['baseline'],
            ),
            [],
        ).append(row)
    aggregate_rows = []
    for key, rows in sorted(aggregate_groups.items()):
        horizon, goal_offset, variant, baseline = key
        pooled_deltas = []
        for row in rows:
            pooled_deltas.extend(
                comparison_deltas[
                    (
                        row['seed'],
                        horizon,
                        goal_offset,
                        variant,
                    )
                ]
            )
        wins = sum(delta > 0 for delta in pooled_deltas)
        losses = sum(delta < 0 for delta in pooled_deltas)
        ci_low, ci_high = paired_bootstrap(
            pooled_deltas,
            bootstrap=args.bootstrap,
            rng=rng,
        )
        seed_deltas = [row['delta_success_pp'] for row in rows]
        aggregate_rows.append(
            {
                'horizon': horizon,
                'goal_offset': goal_offset,
                'variant': variant,
                'baseline': baseline,
                'n_seeds': len(rows),
                'seeds': ','.join(str(row['seed']) for row in rows),
                'n_episodes': len(pooled_deltas),
                'variant_success_rate': mean(
                    row['variant_success_rate'] for row in rows
                ),
                'baseline_success_rate': mean(
                    row['baseline_success_rate'] for row in rows
                ),
                'delta_success_pp': mean(pooled_deltas),
                'delta_ci_low_pp': ci_low,
                'delta_ci_high_pp': ci_high,
                'min_seed_delta_pp': min(seed_deltas),
                'max_seed_delta_pp': max(seed_deltas),
                'wins': wins,
                'losses': losses,
                'ties': len(pooled_deltas) - wins - losses,
                'mcnemar_exact_p': exact_mcnemar(wins, losses),
            }
        )

    out_dir = (args.out_dir or result_dir / 'cemcv_summary').resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for run in runs:
        row = {
            key: value
            for key, value in run.items()
            if key not in {'successes', 'step_counts'}
        }
        row['step_counts'] = (
            ''
            if run['step_counts'] is None
            else json.dumps(run['step_counts'], sort_keys=True)
        )
        row['success_bits'] = ''.join(
            '1' if value else '0' for value in run['successes']
        )
        run_rows.append(row)
    write_csv(out_dir / 'runs.csv', run_rows)
    if comparisons:
        write_csv(out_dir / 'paired_comparisons.csv', comparisons)
    if aggregate_rows:
        write_csv(out_dir / 'aggregate_comparisons.csv', aggregate_rows)

    lines = [
        '# End-to-end CEM cross-validation audit',
        '',
        (f'- Source: `{args.source_label or result_dir}`'),
        f'- Paired bootstrap: {args.bootstrap:,} resamples',
        '',
        '| seed | H | offset | variant | success | baseline | delta (pp) | '
        '95% CI | wins/losses |',
        '|---:|---:|---:|---|---:|---:|---:|---:|---:|',
    ]
    for row in comparisons:
        lines.append(
            f'| {row["seed"]} | {row["horizon"]} | '
            f'{row["goal_offset"]} | '
            f'`{row["variant"]}` | {row["variant_success_rate"]:.1f}% | '
            f'{row["baseline_success_rate"]:.1f}% | '
            f'{row["delta_success_pp"]:+.1f} | '
            f'[{row["delta_ci_low_pp"]:+.1f}, '
            f'{row["delta_ci_high_pp"]:+.1f}] | '
            f'{row["wins"]}/{row["losses"]} |'
        )
    multiseed_rows = [row for row in aggregate_rows if row['n_seeds'] > 1]
    if multiseed_rows:
        lines.extend(
            [
                '',
                '## Pooled multi-seed comparisons',
                '',
                '| H | offset | variant | seeds | success | baseline | '
                'delta (pp) | 95% CI | seed range | wins/losses |',
                '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|',
            ]
        )
        for row in multiseed_rows:
            lines.append(
                f'| {row["horizon"]} | {row["goal_offset"]} | '
                f'`{row["variant"]}` | {row["n_seeds"]} | '
                f'{row["variant_success_rate"]:.1f}% | '
                f'{row["baseline_success_rate"]:.1f}% | '
                f'{row["delta_success_pp"]:+.1f} | '
                f'[{row["delta_ci_low_pp"]:+.1f}, '
                f'{row["delta_ci_high_pp"]:+.1f}] | '
                f'[{row["min_seed_delta_pp"]:+.1f}, '
                f'{row["max_seed_delta_pp"]:+.1f}] | '
                f'{row["wins"]}/{row["losses"]} |'
            )
    lines.extend(
        [
            '',
            'Wins/losses count paired episodes that flip relative to the '
            'configured baseline. Pooled confidence intervals resample '
            'episodes within the available equal-sized seed cells; the seed '
            'range is also shown because three seeds do not support a stable '
            'between-seed variance estimate. These are exploratory '
            'comparisons and were not adjusted for multiple testing.',
            '',
        ]
    )
    (out_dir / 'report.md').write_text(
        '\n'.join(lines),
        encoding='utf-8',
    )

    metadata = {
        'source': args.source_label or str(result_dir),
        'baseline_variant': args.baseline_variant,
        'bootstrap': args.bootstrap,
        'bootstrap_seed': args.seed,
        'horizon_filter': args.horizon,
        'goal_offset_filter': args.goal_offset,
        'cells': [
            {
                'seed': seed,
                'horizon': horizon,
                'goal_offset': offset,
            }
            for seed, horizon, offset in sorted(by_cell)
        ],
        'n_runs': len(runs),
    }
    (out_dir / 'audit.json').write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'Analyzed {len(runs)} runs -> {out_dir}')
    print((out_dir / 'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
