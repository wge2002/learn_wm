"""Summarize the paired causal gate from ``oe_update_resample.py``."""

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


def validate(result: dict[str, np.ndarray]) -> None:
    required = {
        'version',
        'source',
        'generator',
        'state_indices',
        'steps',
        'alphas',
        'num_samples',
        'sample_true',
        'sample_success',
        'mean_success',
        'model_refit_true',
        'model_refit_success',
        'oracle_refit_true',
        'oracle_refit_success',
        'max_state_mismatch',
        'max_goal_mismatch',
        'max_roundtrip_error',
        'elapsed_seconds',
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f'missing result keys: {missing}')
    shape = result['sample_true'].shape
    expected = (
        len(result['state_indices']),
        len(result['steps']),
        len(result['alphas']),
        int(result['num_samples']),
    )
    if shape != expected:
        raise ValueError(f'sample_true has shape {shape}, expected {expected}')
    if result['sample_success'].shape != expected:
        raise ValueError('sample_success shape does not match sample_true')
    for key in (
        'mean_success',
        'model_refit_true',
        'model_refit_success',
        'oracle_refit_true',
        'oracle_refit_success',
    ):
        if result[key].shape != expected[:3]:
            raise ValueError(
                f'{key} has shape {result[key].shape}, expected {expected[:3]}'
            )
    alphas = result['alphas'].astype(np.float64)
    if not np.all(np.diff(alphas) > 0):
        raise ValueError('alphas must be strictly increasing')
    if not np.any(np.isclose(alphas, 0.0)):
        raise ValueError('alphas must include the learned-update control 0')


def bootstrap_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def state_metrics(
    result: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    true = result['sample_true'].astype(np.float64)
    success = result['sample_success'].astype(bool)
    return {
        'coverage': np.any(success, axis=-1).mean(axis=1),
        'min_true': np.min(true, axis=-1).mean(axis=1),
        'mean_success': result['mean_success'].astype(float).mean(axis=1),
        'model_refit_true': result['model_refit_true']
        .astype(np.float64)
        .mean(axis=1),
        'model_refit_success': result['model_refit_success']
        .astype(float)
        .mean(axis=1),
        'oracle_refit_true': result['oracle_refit_true']
        .astype(np.float64)
        .mean(axis=1),
        'oracle_refit_success': result['oracle_refit_success']
        .astype(float)
        .mean(axis=1),
    }


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
    validate(result)
    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else args.result.with_name(f'{args.result.stem}_summary')
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = result['alphas'].astype(np.float64)
    steps = result['steps'].astype(np.int64)
    baseline_i = int(np.flatnonzero(np.isclose(alphas, 0.0))[0])
    metrics = state_metrics(result)
    n_states = len(result['state_indices'])
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
            'n_rounds': len(steps),
        }
        for name, values in metrics.items():
            current = values[:, alpha_i]
            delta = current - values[:, baseline_i]
            low, high = bootstrap_ci(delta, bootstrap_indices)
            row[name] = float(current.mean())
            row[f'{name}_delta_vs_alpha0'] = float(delta.mean())
            row[f'{name}_delta_ci_low'] = low
            row[f'{name}_delta_ci_high'] = high
        aggregate_rows.append(row)

    round_rows = []
    sample_true = result['sample_true'].astype(np.float64)
    sample_success = result['sample_success'].astype(bool)
    for round_i, step in enumerate(steps):
        for alpha_i, alpha in enumerate(alphas):
            true = sample_true[:, round_i, alpha_i]
            success = sample_success[:, round_i, alpha_i]
            round_rows.append(
                {
                    'step': int(step),
                    'alpha': float(alpha),
                    'coverage': float(np.any(success, axis=-1).mean()),
                    'min_true': float(np.min(true, axis=-1).mean()),
                    'mean_success': float(
                        result['mean_success'][:, round_i, alpha_i].mean()
                    ),
                    'model_refit_true': float(
                        result['model_refit_true'][
                            :,
                            round_i,
                            alpha_i,
                        ].mean()
                    ),
                    'model_refit_success': float(
                        result['model_refit_success'][
                            :,
                            round_i,
                            alpha_i,
                        ].mean()
                    ),
                    'oracle_refit_true': float(
                        result['oracle_refit_true'][
                            :,
                            round_i,
                            alpha_i,
                        ].mean()
                    ),
                    'oracle_refit_success': float(
                        result['oracle_refit_success'][
                            :,
                            round_i,
                            alpha_i,
                        ].mean()
                    ),
                }
            )

    centered_alpha = alphas - alphas.mean()
    denominator = float(np.sum(centered_alpha**2))
    slope_rows = []
    for name, values in metrics.items():
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

    min_true_by_round = np.min(sample_true, axis=-1)
    coverage_by_round = np.any(sample_success, axis=-1).astype(float)
    adjacent_rows = []
    for alpha_i in range(len(alphas) - 1):
        next_i = alpha_i + 1
        adjacent_rows.append(
            {
                'alpha_from': float(alphas[alpha_i]),
                'alpha_to': float(alphas[next_i]),
                'fraction_min_true_nonincreasing': float(
                    np.mean(
                        min_true_by_round[:, :, next_i]
                        <= min_true_by_round[:, :, alpha_i]
                    )
                ),
                'fraction_coverage_nondecreasing': float(
                    np.mean(
                        coverage_by_round[:, :, next_i]
                        >= coverage_by_round[:, :, alpha_i]
                    )
                ),
                'fraction_model_refit_true_nonincreasing': float(
                    np.mean(
                        result['model_refit_true'][:, :, next_i]
                        <= result['model_refit_true'][:, :, alpha_i]
                    )
                ),
            }
        )

    write_csv(out_dir / 'aggregate_metrics.csv', aggregate_rows)
    write_csv(out_dir / 'round_metrics.csv', round_rows)
    write_csv(out_dir / 'dose_slopes.csv', slope_rows)
    write_csv(out_dir / 'adjacent_monotonicity.csv', adjacent_rows)

    audit = {
        'source': str(args.result.resolve()),
        'generator': str(result['generator']),
        'scorer': str(result.get('scorer', result['generator'])),
        'rescored_source': bool(result.get('rescored_source', False)),
        'n_states': n_states,
        'steps': steps.tolist(),
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
        '# OE update one-step causal gate',
        '',
        f'- Generator: `{audit["generator"]}`',
        f'- Scorer: `{audit["scorer"]}`',
        f'- Re-scored source populations: `{audit["rescored_source"]}`',
        f'- Paired states: {n_states}',
        f'- Source CEM rounds: {steps.tolist()}',
        f'- Next-population samples per intervention: {audit["num_samples"]}',
        '',
        '## Aggregate over states and source rounds',
        '',
        '| alpha | coverage | Δ coverage | min true | Δ min true '
        '| model-refit success | Δ model success | oracle-refit success |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in aggregate_rows:
        lines.append(
            f'| {row["alpha"]:.2f} '
            f'| {row["coverage"]:.3f} '
            f'| {row["coverage_delta_vs_alpha0"]:+.3f} '
            f'| {row["min_true"]:.2f} '
            f'| {row["min_true_delta_vs_alpha0"]:+.2f} '
            f'[{row["min_true_delta_ci_low"]:+.2f}, '
            f'{row["min_true_delta_ci_high"]:+.2f}] '
            f'| {row["model_refit_success"]:.3f} '
            f'| {row["model_refit_success_delta_vs_alpha0"]:+.3f} '
            f'[{row["model_refit_success_delta_ci_low"]:+.3f}, '
            f'{row["model_refit_success_delta_ci_high"]:+.3f}] '
            f'| {row["oracle_refit_success"]:.3f} |'
        )

    lines.extend(
        [
            '',
            'Coverage asks whether the resampled next population contains any '
            'successful candidate. Model-refit uses the unchanged world '
            'model to select its top-k mean; oracle-refit is a ceiling.',
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
