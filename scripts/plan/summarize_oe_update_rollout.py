"""Summarize recursive oracle-interpolated CEM update interventions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20_260_718)
    return parser.parse_args()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def bootstrap_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    sampled = np.asarray(values, dtype=np.float64)[indices]
    means = sampled.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


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
    if args.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')
    result = load(args.result)
    required = {
        'generator',
        'state_indices',
        'start_step',
        'num_rounds',
        'alphas',
        'num_samples',
        'population_true',
        'population_success',
        'mean_true',
        'mean_success',
        'final_mean_true',
        'final_mean_success',
        'max_state_mismatch',
        'max_goal_mismatch',
        'max_roundtrip_error',
        'elapsed_seconds',
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f'missing result keys: {missing}')

    alphas = result['alphas'].astype(np.float64)
    if not np.all(np.diff(alphas) > 0):
        raise ValueError('alphas must be strictly increasing')
    baseline_matches = np.flatnonzero(np.isclose(alphas, 0.0))
    if len(baseline_matches) != 1:
        raise ValueError('alphas must contain exactly one zero control')
    baseline_i = int(baseline_matches[0])

    true = result['population_true'].astype(np.float64)
    success = result['population_success'].astype(bool)
    n_states = len(result['state_indices'])
    expected = (
        n_states,
        len(alphas),
        int(result['num_rounds']),
        int(result['num_samples']),
    )
    if true.shape != expected or success.shape != expected:
        raise ValueError(
            f'population arrays have {true.shape}/{success.shape}, '
            f'expected {expected}'
        )
    for key in ('mean_true', 'mean_success'):
        if result[key].shape != expected[:3]:
            raise ValueError(f'{key} shape does not match {expected[:3]}')
    for key in ('final_mean_true', 'final_mean_success'):
        if result[key].shape != expected[:2]:
            raise ValueError(f'{key} shape does not match {expected[:2]}')

    coverage = np.any(success, axis=-1).astype(float)
    min_true = np.min(true, axis=-1)
    state_metrics = {
        'average_coverage': coverage.mean(axis=2),
        'last_coverage': coverage[:, :, -1],
        'last_min_true': min_true[:, :, -1],
        'average_min_true': min_true.mean(axis=2),
        'mean_success': result['mean_success'].astype(float).mean(axis=2),
        'final_mean_true': result['final_mean_true'].astype(np.float64),
        'final_mean_success': result['final_mean_success'].astype(float),
    }

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        n_states,
        size=(args.bootstrap, n_states),
    )
    aggregate_rows = []
    for alpha_i, alpha in enumerate(alphas):
        row: dict[str, float | int] = {
            'alpha': float(alpha),
            'n_states': n_states,
        }
        for name, values in state_metrics.items():
            current = values[:, alpha_i]
            delta = current - values[:, baseline_i]
            low, high = bootstrap_ci(delta, bootstrap_indices)
            row[name] = float(current.mean())
            row[f'{name}_delta_vs_alpha0'] = float(delta.mean())
            row[f'{name}_delta_ci_low'] = low
            row[f'{name}_delta_ci_high'] = high
        aggregate_rows.append(row)

    round_rows = []
    for alpha_i, alpha in enumerate(alphas):
        for round_i in range(int(result['num_rounds'])):
            round_rows.append(
                {
                    'alpha': float(alpha),
                    'branch_round': round_i + 1,
                    'coverage': float(coverage[:, alpha_i, round_i].mean()),
                    'min_true': float(min_true[:, alpha_i, round_i].mean()),
                    'mean_true': float(
                        result['mean_true'][:, alpha_i, round_i].mean()
                    ),
                    'mean_success': float(
                        result['mean_success'][:, alpha_i, round_i].mean()
                    ),
                }
            )

    centered_alpha = alphas - alphas.mean()
    denominator = float(np.sum(centered_alpha**2))
    slope_rows = []
    for name, values in state_metrics.items():
        slopes = np.sum(values * centered_alpha[None], axis=1) / denominator
        low, high = bootstrap_ci(slopes, bootstrap_indices)
        slope_rows.append(
            {
                'metric': name,
                'mean_slope_per_alpha': float(slopes.mean()),
                'ci_low': low,
                'ci_high': high,
            }
        )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else args.result.with_name(f'{args.result.stem}_summary')
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'aggregate_metrics.csv', aggregate_rows)
    write_csv(out_dir / 'round_metrics.csv', round_rows)
    write_csv(out_dir / 'dose_slopes.csv', slope_rows)

    audit = {
        'source': str(args.result.resolve()),
        'generator': str(result['generator']),
        'scorer': str(result.get('scorer', result['generator'])),
        'n_states': n_states,
        'start_step': int(result['start_step']),
        'num_rounds': int(result['num_rounds']),
        'alphas': alphas.tolist(),
        'num_samples': int(result['num_samples']),
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'max_state_mismatch': float(result['max_state_mismatch']),
        'max_goal_mismatch': float(result['max_goal_mismatch']),
        'candidate_storage_dtype': str(
            result.get('candidate_storage_dtype', 'legacy-mixed')
        ),
        'max_candidate_quantization_error': float(
            result.get('max_candidate_quantization_error', float('nan'))
        ),
        'max_roundtrip_error': float(result['max_roundtrip_error']),
        'elapsed_seconds': float(result['elapsed_seconds']),
    }
    (out_dir / 'audit.json').write_text(
        json.dumps(audit, indent=2) + '\n',
        encoding='utf-8',
    )

    lines = [
        '# Recursive OE update intervention',
        '',
        f'- Generator: `{audit["generator"]}`',
        f'- Scorer: `{audit["scorer"]}`',
        f'- Paired states: {n_states}',
        f'- Start after source CEM step: {audit["start_step"]}',
        f'- Counterfactual rounds: {audit["num_rounds"]}',
        f'- Candidates per branch round: {audit["num_samples"]}',
        '',
        '| alpha | avg coverage | last coverage | last min true '
        '| final mean true | Δ final true | final mean success '
        '| Δ final success |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in aggregate_rows:
        lines.append(
            f'| {row["alpha"]:.2f} '
            f'| {row["average_coverage"]:.3f} '
            f'| {row["last_coverage"]:.3f} '
            f'| {row["last_min_true"]:.2f} '
            f'| {row["final_mean_true"]:.2f} '
            f'| {row["final_mean_true_delta_vs_alpha0"]:+.2f} '
            f'[{row["final_mean_true_delta_ci_low"]:+.2f}, '
            f'{row["final_mean_true_delta_ci_high"]:+.2f}] '
            f'| {row["final_mean_success"]:.3f} '
            f'| {row["final_mean_success_delta_vs_alpha0"]:+.3f} '
            f'[{row["final_mean_success_delta_ci_low"]:+.3f}, '
            f'{row["final_mean_success_delta_ci_high"]:+.3f}] |'
        )
    lines.extend(
        [
            '',
            '## Linear dose slopes',
            '',
            '| metric | slope per alpha | 95% paired bootstrap CI |',
            '|---|---:|---:|',
        ]
    )
    for row in slope_rows:
        lines.append(
            f'| `{row["metric"]}` '
            f'| {row["mean_slope_per_alpha"]:+.4f} '
            f'| [{row["ci_low"]:+.4f}, {row["ci_high"]:+.4f}] |'
        )
    lines.extend(
        [
            '',
            'Negative slopes are favorable for true-cost metrics; positive '
            'slopes are favorable for success metrics.',
            '',
            '## Integrity',
            '',
            f'- State mismatch: `{audit["max_state_mismatch"]:.3e}`',
            f'- Goal mismatch: `{audit["max_goal_mismatch"]:.3e}`',
            (
                '- Candidate quantization: '
                f'`{audit["candidate_storage_dtype"]}`, max error '
                f'`{audit["max_candidate_quantization_error"]:.3e}`'
            ),
            f'- Action roundtrip error: `{audit["max_roundtrip_error"]:.3e}`',
        ]
    )
    (out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print(f'summary -> {out_dir}')


if __name__ == '__main__':
    main()
