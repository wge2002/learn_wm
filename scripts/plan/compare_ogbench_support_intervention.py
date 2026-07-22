"""Paired comparison of zero-mean and expert-seeded OGBench CEM paths.

Both conditions must contain exactly the same dataset rows, models, CEM
settings, and candidate counts.  The only intended difference is
``metadata.init_prior``.  Dataset state remains the bootstrap unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import summarize_ogbench_candidate_fidelity as ogb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', nargs='+', required=True, type=Path)
    parser.add_argument('--prior', nargs='+', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=20260722)
    return parser.parse_args()


def assert_paired(
    base_protocol: dict,
    base_data: dict,
    prior_protocol: dict,
    prior_data: dict,
) -> None:
    for field in ogb.PROTOCOL_FIELDS:
        if not np.array_equal(base_protocol[field], prior_protocol[field]):
            raise ValueError(f'Protocol field {field!r} differs by condition')
    for field in ('rows', 'episodes', 'starts', 'sampled_initial_distance'):
        if not np.array_equal(base_data[field], prior_data[field]):
            raise ValueError(f'State pairing field {field!r} differs by condition')

    base_prior = str(base_data['metadata'].get('init_prior', 'zero'))
    intervention_prior = str(prior_data['metadata'].get('init_prior', 'zero'))
    if base_prior != 'zero':
        raise ValueError(f'Baseline init_prior must be zero, got {base_prior!r}')
    if intervention_prior == 'zero':
        raise ValueError('Intervention init_prior must not be zero')

    base_audit = ogb.audit_protocol(base_protocol, base_data)
    prior_audit = ogb.audit_protocol(prior_protocol, prior_data)
    for field in ('num_states', 'natural_candidates', 'anchors'):
        if base_audit[field] != prior_audit[field]:
            raise ValueError(f'Audit field {field!r} differs by condition')


def ci_dict(values: np.ndarray, rng: np.random.Generator) -> dict:
    mean, lo, hi, n = ogb.bootstrap_mean_ci(values, rng=rng)
    return {'mean': mean, 'ci95': [lo, hi], 'states': n}


def round_summary(
    values: np.ndarray,
    steps: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    return {
        str(int(step)): ci_dict(values[:, round_i], rng)
        for round_i, step in enumerate(steps)
    }


def generator_round_summary(
    values: np.ndarray,
    generators: list[str],
    steps: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    return {
        str(int(step)): {
            generator: ci_dict(values[:, generator_i, round_i], rng)
            for generator_i, generator in enumerate(generators)
        }
        for round_i, step in enumerate(steps)
    }


def condition_metric_arrays(data: dict, metrics: dict) -> dict[str, np.ndarray]:
    return {
        'natural_support': metrics['proposal']['coverage'],
        'natural_success_fraction': metrics['proposal']['success_fraction'],
        'natural_oracle_min': metrics['proposal']['oracle_min'],
        'actual_mean_success': data['recorded_mean_success'].astype(np.float64),
        'actual_mean_distance': data['recorded_mean_true'].astype(np.float64),
        # Average scorer quality is useful as a condition-level diagnostic;
        # generator x scorer interaction is reported separately below.
        'natural_elite_success': np.nanmean(
            metrics['natural']['elite_success'], axis=-1
        ),
        'supported_elite_success': np.nanmean(
            metrics['augmented']['elite_success'], axis=-1
        ),
        'supported_refit_success': np.nanmean(
            data['refit_success'].astype(np.float64), axis=-1
        ),
        'supported_refit_distance': np.nanmean(data['refit_true'], axis=-1),
    }


def scorer_interactions(metrics: dict) -> dict[str, np.ndarray]:
    return {
        'natural_elite_success': ogb.scorer_path_interaction(
            metrics['natural']['elite_success']
        ),
        'supported_elite_success': ogb.scorer_path_interaction(
            metrics['augmented']['elite_success']
        ),
    }


def generate_report(
    base_protocol: dict,
    base_data: dict,
    prior_data: dict,
    *,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    steps = np.asarray(base_protocol['steps'])
    generators = [
        Path(str(value)).parent.name
        for value in base_protocol['generators'].tolist()
    ]
    base_metrics = ogb.compute_metrics(base_protocol, base_data)
    prior_metrics = ogb.compute_metrics(base_protocol, prior_data)
    base_arrays = condition_metric_arrays(base_data, base_metrics)
    prior_arrays = condition_metric_arrays(prior_data, prior_metrics)

    conditions = {'zero': {}, 'prior': {}}
    changes = {}
    for name in base_arrays:
        conditions['zero'][name] = generator_round_summary(
            base_arrays[name], generators, steps, rng
        )
        conditions['prior'][name] = generator_round_summary(
            prior_arrays[name], generators, steps, rng
        )
        changes[f'{name}_prior_minus_zero'] = generator_round_summary(
            prior_arrays[name] - base_arrays[name], generators, steps, rng
        )

    base_interactions = scorer_interactions(base_metrics)
    prior_interactions = scorer_interactions(prior_metrics)
    interactions = {}
    for name in base_interactions:
        interactions[name] = {
            'zero': round_summary(base_interactions[name], steps, rng),
            'prior': round_summary(prior_interactions[name], steps, rng),
            'prior_minus_zero': round_summary(
                prior_interactions[name] - base_interactions[name], steps, rng
            ),
        }

    # Generator gap asks whether the K5-induced path benefits more than K1.
    generator_gaps = {}
    for name in ('natural_support', 'actual_mean_success', 'actual_mean_distance'):
        base_gap = base_arrays[name][:, 1] - base_arrays[name][:, 0]
        prior_gap = prior_arrays[name][:, 1] - prior_arrays[name][:, 0]
        generator_gaps[name] = {
            'zero_k5_minus_k1': round_summary(base_gap, steps, rng),
            'prior_k5_minus_k1': round_summary(prior_gap, steps, rng),
            'gap_change': round_summary(prior_gap - base_gap, steps, rng),
        }

    final_i = -1
    print(
        'Paired support intervention: '
        f'N={len(base_data["rows"])}, '
        f'prior={prior_data["metadata"].get("init_prior")}, '
        f'steps={steps.tolist()}'
    )
    print(
        'step generator support(zero->prior) actual_success(zero->prior) '
        'actual_distance(zero->prior)'
    )
    for round_i, step in enumerate(steps):
        for generator_i, generator in enumerate(generators):
            print(
                f'{int(step):>4} {generator:<42} '
                f'{base_arrays["natural_support"][:, generator_i, round_i].mean():.3f}'
                f'->{prior_arrays["natural_support"][:, generator_i, round_i].mean():.3f} '
                f'{base_arrays["actual_mean_success"][:, generator_i, round_i].mean():.3f}'
                f'->{prior_arrays["actual_mean_success"][:, generator_i, round_i].mean():.3f} '
                f'{base_arrays["actual_mean_distance"][:, generator_i, round_i].mean():.4f}'
                f'->{prior_arrays["actual_mean_distance"][:, generator_i, round_i].mean():.4f}'
            )

    print('\nFinal paired prior-zero changes (state bootstrap 95% CI)')
    final = str(int(steps[final_i]))
    for name in ('natural_support', 'actual_mean_success', 'actual_mean_distance'):
        for generator in generators:
            item = changes[f'{name}_prior_minus_zero'][final][generator]
            print(
                f'{name} {generator}: {item["mean"]:.4f} '
                f'[{item["ci95"][0]:.4f}, {item["ci95"][1]:.4f}]'
            )
    item = interactions['supported_elite_success']['prior_minus_zero'][final]
    print(
        'supported path-interaction change: '
        f'{item["mean"]:.4f} [{item["ci95"][0]:.4f}, {item["ci95"][1]:.4f}]'
    )

    return {
        'version': 1,
        'protocol': {
            'steps': steps.astype(int).tolist(),
            'generators': generators,
            'labels': [str(value) for value in base_protocol['labels'].tolist()],
            'rows': base_data['rows'].astype(int).tolist(),
            'baseline_prior': base_data['metadata'].get('init_prior', 'zero'),
            'intervention_prior': prior_data['metadata'].get('init_prior'),
            'num_states': int(len(base_data['rows'])),
        },
        'conditions': conditions,
        'paired_changes': changes,
        'scorer_path_interactions': interactions,
        'generator_gaps': generator_gaps,
    }


def main() -> None:
    args = parse_args()
    base_protocol, base_data = ogb.load_shards(args.baseline)
    prior_protocol, prior_data = ogb.load_shards(args.prior)
    assert_paired(base_protocol, base_data, prior_protocol, prior_data)
    result = generate_report(
        base_protocol, base_data, prior_data, seed=args.seed
    )
    result['inputs'] = {
        'baseline': [
            {'path': str(path.resolve()), 'sha256': ogb.sha256(path)}
            for path in args.baseline
        ],
        'prior': [
            {'path': str(path.resolve()), 'sha256': ogb.sha256(path)}
            for path in args.prior
        ],
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(f'\nreport -> {args.out}')


if __name__ == '__main__':
    main()
