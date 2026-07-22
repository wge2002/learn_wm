"""Summarize paired OGBench CEM candidate-fidelity audit shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.stats import spearmanr


PROTOCOL_FIELDS = (
    'generators',
    'scorers',
    'labels',
    'anchor_names',
    'steps',
    'horizon',
    'goal_offset',
    'action_block',
    'topk',
)

STATE_FIELDS = (
    'rows',
    'episodes',
    'starts',
    'sampled_initial_distance',
    'candidate_indices',
    'candidates',
    'pred',
    'true',
    'success',
    'refit_true',
    'refit_success',
    'oracle_refit_true',
    'oracle_refit_success',
    'update_cosine',
    'update_norm_ratio',
    'recorded_mean_true',
    'recorded_mean_success',
)


def scalar(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_shards(paths: list[Path]) -> tuple[dict, dict]:
    archives = [np.load(path, allow_pickle=False) for path in paths]
    try:
        reference = archives[0]
        protocol = {field: np.asarray(reference[field]) for field in PROTOCOL_FIELDS}
        for path, archive in zip(paths[1:], archives[1:], strict=True):
            for field, expected in protocol.items():
                actual = np.asarray(archive[field])
                if not np.array_equal(actual, expected):
                    raise ValueError(f'{path}: protocol field {field} differs')
        data = {
            field: np.concatenate(
                [np.asarray(archive[field]) for archive in archives], axis=0
            )
            for field in STATE_FIELDS
        }
        data['max_reset_error'] = max(
            float(scalar(archive['max_reset_error'])) for archive in archives
        )
        data['max_roundtrip_error'] = max(
            float(scalar(archive['max_roundtrip_error'])) for archive in archives
        )
        data['elapsed_seconds'] = [
            float(scalar(archive['elapsed_seconds'])) for archive in archives
        ]
        metadata = [json.loads(str(scalar(archive['metadata']))) for archive in archives]
        if any(item != metadata[0] for item in metadata[1:]):
            raise ValueError('Shard metadata differs')
        data['metadata'] = metadata[0]
        return protocol, data
    finally:
        for archive in archives:
            archive.close()


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int = 20_000,
) -> tuple[float, float, float, int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return float('nan'), float('nan'), float('nan'), 0
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))]
    means = draws.mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi), int(len(values))


def format_ci(values: np.ndarray, rng: np.random.Generator, scale=1.0) -> str:
    mean, lo, hi, n = bootstrap_mean_ci(values, rng=rng)
    if not n:
        return 'nan'
    return f'{mean * scale:.3f} [{lo * scale:.3f}, {hi * scale:.3f}]'


def rank_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    success: np.ndarray,
    *,
    topk: int,
) -> dict[str, float]:
    true_span = float(np.ptp(true))
    rho = (
        float(spearmanr(pred, true).statistic)
        if true_span > 1e-8
        else float('nan')
    )
    k = min(int(topk), len(true))
    pred_top = np.argsort(pred, kind='stable')[:k]
    true_top = np.argsort(true, kind='stable')[:k]
    chosen = int(np.argmin(pred))
    return {
        'rho': rho,
        'topk_overlap': len(np.intersect1d(pred_top, true_top)) / k,
        'top1_true': float(true[chosen]),
        'top1_success': float(success[chosen]),
        'elite_true': float(np.mean(true[pred_top])),
        'elite_success': float(np.mean(success[pred_top])),
    }


def compute_metrics(protocol: dict, data: dict) -> dict:
    indices = data['candidate_indices']
    pred = data['pred']
    true = data['true']
    success = data['success']
    n_states, n_generators, n_rounds, n_scorers, _ = pred.shape
    topk = int(scalar(protocol['topk']))

    metric_names = (
        'rho',
        'topk_overlap',
        'top1_true',
        'top1_success',
        'elite_true',
        'elite_success',
    )
    natural = {
        name: np.full(
            (n_states, n_generators, n_rounds, n_scorers),
            np.nan,
            dtype=np.float64,
        )
        for name in metric_names
    }
    augmented = {name: np.full_like(value, np.nan) for name, value in natural.items()}
    proposal = {
        name: np.full(
            (n_states, n_generators, n_rounds), np.nan, dtype=np.float64
        )
        for name in (
            'coverage',
            'success_fraction',
            'oracle_min',
            'nondegenerate',
        )
    }

    anchor_names = [str(value) for value in protocol['anchor_names'].tolist()]
    n_anchors = len(anchor_names)
    anchor_rank = np.full(
        (n_states, n_generators, n_rounds, n_scorers, n_anchors),
        np.nan,
        dtype=np.float64,
    )
    anchor_true = np.full(
        (n_states, n_generators, n_rounds, n_anchors),
        np.nan,
        dtype=np.float64,
    )
    anchor_success = np.full_like(anchor_true, np.nan)

    for state_i in range(n_states):
        for generator_i in range(n_generators):
            for round_i in range(n_rounds):
                ids = indices[state_i, generator_i, round_i]
                natural_mask = ids >= 0
                natural_true = true[state_i, generator_i, round_i, natural_mask]
                natural_success = success[
                    state_i, generator_i, round_i, natural_mask
                ]
                proposal['coverage'][state_i, generator_i, round_i] = float(
                    np.any(natural_success)
                )
                proposal['success_fraction'][state_i, generator_i, round_i] = float(
                    np.mean(natural_success)
                )
                proposal['oracle_min'][state_i, generator_i, round_i] = float(
                    np.min(natural_true)
                )
                proposal['nondegenerate'][state_i, generator_i, round_i] = float(
                    np.ptp(natural_true) > 1e-8
                )

                for scorer_i in range(n_scorers):
                    natural_pred = pred[
                        state_i, generator_i, round_i, scorer_i, natural_mask
                    ]
                    values = rank_metrics(
                        natural_pred,
                        natural_true,
                        natural_success,
                        topk=topk,
                    )
                    for name, value in values.items():
                        natural[name][
                            state_i, generator_i, round_i, scorer_i
                        ] = value

                    values = rank_metrics(
                        pred[state_i, generator_i, round_i, scorer_i],
                        true[state_i, generator_i, round_i],
                        success[state_i, generator_i, round_i],
                        topk=topk,
                    )
                    for name, value in values.items():
                        augmented[name][
                            state_i, generator_i, round_i, scorer_i
                        ] = value

                    order = np.argsort(
                        pred[state_i, generator_i, round_i, scorer_i],
                        kind='stable',
                    )
                    ranks = np.empty(len(order), dtype=np.float64)
                    ranks[order] = np.arange(len(order), dtype=np.float64)
                    denom = max(1, len(order) - 1)
                    for anchor_i in range(n_anchors):
                        positions = np.flatnonzero(ids == -(anchor_i + 1))
                        if len(positions) != 1:
                            raise ValueError(
                                f'Expected one candidate for anchor {anchor_i}, '
                                f'got {len(positions)}'
                            )
                        anchor_rank[
                            state_i,
                            generator_i,
                            round_i,
                            scorer_i,
                            anchor_i,
                        ] = ranks[positions[0]] / denom
                for anchor_i in range(n_anchors):
                    position = int(np.flatnonzero(ids == -(anchor_i + 1))[0])
                    anchor_true[
                        state_i, generator_i, round_i, anchor_i
                    ] = true[state_i, generator_i, round_i, position]
                    anchor_success[
                        state_i, generator_i, round_i, anchor_i
                    ] = success[state_i, generator_i, round_i, position]

    return {
        'natural': natural,
        'augmented': augmented,
        'proposal': proposal,
        'anchor_rank': anchor_rank,
        'anchor_true': anchor_true,
        'anchor_success': anchor_success,
    }


def state_balance(values: np.ndarray) -> np.ndarray:
    """Average generator banks while preserving state as the sampling unit."""
    return np.nanmean(np.asarray(values, dtype=np.float64), axis=1)


def scorer_delta(values: np.ndarray) -> np.ndarray:
    balanced = state_balance(values)
    if balanced.shape[-1] != 2:
        raise ValueError('Paired K1/K5 summary requires exactly two scorers')
    return balanced[..., 1] - balanced[..., 0]


def scorer_path_interaction(values: np.ndarray) -> np.ndarray:
    """(K5-K1 scorer gap on K5 path) - gap on K1 path."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape[1] != 2 or values.shape[-1] != 2:
        raise ValueError('Path interaction requires two generators and scorers')
    scorer_gap = values[..., 1] - values[..., 0]
    return scorer_gap[:, 1] - scorer_gap[:, 0]


def audit_protocol(protocol: dict, data: dict) -> dict:
    rows = data['rows']
    if len(np.unique(rows)) != len(rows):
        raise ValueError('Formal shards contain duplicate sampled rows')
    if not np.all(data['sampled_initial_distance'] > 0.04):
        raise ValueError('Formal shards contain initially successful rows')

    ids = data['candidate_indices']
    natural0 = ids[:, 0, 0] >= 0
    if not np.all(natural0 == (ids[:, 1, 0] >= 0)):
        raise ValueError('Round-zero natural masks differ across generators')
    round0_candidate_error = float(
        np.max(
            np.abs(
                data['candidates'][:, 0, 0] - data['candidates'][:, 1, 0]
            )
        )
    )
    round0_true_error = float(
        np.max(np.abs(data['true'][:, 0, 0] - data['true'][:, 1, 0]))
    )
    round0_pred_error = float(
        np.max(np.abs(data['pred'][:, 0, 0] - data['pred'][:, 1, 0]))
    )
    if round0_candidate_error != 0 or round0_true_error > 1e-10:
        raise ValueError(
            'Common-random round-zero pairing failed: '
            f'candidate={round0_candidate_error}, true={round0_true_error}'
        )
    return {
        'num_states': int(len(rows)),
        'num_unique_rows': int(len(np.unique(rows))),
        'initial_distance_mean': float(np.mean(data['sampled_initial_distance'])),
        'initial_distance_min': float(np.min(data['sampled_initial_distance'])),
        'initial_distance_max': float(np.max(data['sampled_initial_distance'])),
        'round0_candidate_max_error': round0_candidate_error,
        'round0_true_max_error': round0_true_error,
        'round0_pred_max_error': round0_pred_error,
        'max_reset_error': data['max_reset_error'],
        'max_roundtrip_error': data['max_roundtrip_error'],
        'natural_candidates': int(np.sum(ids[0, 0, 0] >= 0)),
        'anchors': int(np.sum(ids[0, 0, 0] < 0)),
    }


def summarize_metric_delta(
    values: np.ndarray,
    steps: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    return summarize_round_values(scorer_delta(values), steps, rng)


def summarize_round_values(
    values: np.ndarray,
    steps: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    return {
        str(int(step)): {
            'mean': mean,
            'ci95': [lo, hi],
            'states': n,
        }
        for step, column in zip(steps, values.T, strict=True)
        for mean, lo, hi, n in [bootstrap_mean_ci(column, rng=rng)]
    }


def report(protocol: dict, data: dict, metrics: dict, *, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    steps = np.asarray(protocol['steps'])
    labels = [str(value) for value in protocol['labels'].tolist()]
    generators = [Path(str(value)).parent.name for value in protocol['generators']]
    audit = audit_protocol(protocol, data)

    print(
        f'Protocol: N={audit["num_states"]}, natural={audit["natural_candidates"]}, '
        f'anchors={audit["anchors"]}, steps={steps.tolist()}, labels={labels}'
    )
    print(f'Initial prior: {data["metadata"].get("init_prior", "zero")}')
    print(
        f'Pairing: round0 candidates={audit["round0_candidate_max_error"]:.1e}, '
        f'true={audit["round0_true_max_error"]:.1e}, '
        f'pred={audit["round0_pred_max_error"]:.1e}; '
        f'reset={audit["max_reset_error"]:.1e}, '
        f'roundtrip={audit["max_roundtrip_error"]:.1e}'
    )

    print('\nNatural CEM proposal / actual refit')
    print('step  generator  coverage  nondegenerate  oracle_min  actual_success  actual_dist')
    proposal_summary = {}
    for round_i, step in enumerate(steps):
        step_summary = {}
        for generator_i, generator in enumerate(generators):
            values = {
                'coverage': float(
                    np.mean(
                        metrics['proposal']['coverage'][
                            :, generator_i, round_i
                        ]
                    )
                ),
                'nondegenerate': float(
                    np.mean(
                        metrics['proposal']['nondegenerate'][
                            :, generator_i, round_i
                        ]
                    )
                ),
                'oracle_min': float(
                    np.mean(
                        metrics['proposal']['oracle_min'][
                            :, generator_i, round_i
                        ]
                    )
                ),
                'actual_success': float(
                    np.mean(
                        data['recorded_mean_success'][
                            :, generator_i, round_i
                        ]
                    )
                ),
                'actual_distance': float(
                    np.mean(
                        data['recorded_mean_true'][:, generator_i, round_i]
                    )
                ),
            }
            step_summary[generator] = values
            print(
                f'{int(step):>4}  {generator:<10}  '
                f'{values["coverage"]:.3f}     '
                f'{values["nondegenerate"]:.3f}          '
                f'{values["oracle_min"]:.4f}      '
                f'{values["actual_success"]:.3f}          '
                f'{values["actual_distance"]:.4f}'
            )
        proposal_summary[str(int(step))] = step_summary

    print('\nNatural population scorer fidelity (balanced over generator banks)')
    print('step  scorer  rho_defined  rho  top30  top1_success  elite_success')
    natural_scorer_summary = {}
    for round_i, step in enumerate(steps):
        step_summary = {}
        for scorer_i, label in enumerate(labels):
            rho = metrics['natural']['rho'][:, :, round_i, scorer_i]
            values = {
                'rho_defined': float(
                    np.mean(np.any(np.isfinite(rho), axis=1))
                ),
                'rho': float(np.nanmean(rho)),
                'top30_overlap': float(
                    np.mean(
                        metrics['natural']['topk_overlap'][
                            :, :, round_i, scorer_i
                        ]
                    )
                ),
                'top1_success': float(
                    np.mean(
                        metrics['natural']['top1_success'][
                            :, :, round_i, scorer_i
                        ]
                    )
                ),
                'elite_success': float(
                    np.mean(
                        metrics['natural']['elite_success'][
                            :, :, round_i, scorer_i
                        ]
                    )
                ),
            }
            step_summary[label] = values
            print(
                f'{int(step):>4}  {label:<6}  '
                f'{values["rho_defined"]:.3f}       '
                f'{values["rho"]:.3f}  '
                f'{values["top30_overlap"]:.3f}  '
                f'{values["top1_success"]:.3f}         '
                f'{values["elite_success"]:.3f}'
            )
        natural_scorer_summary[str(int(step))] = step_summary

    anchor_names = [str(value) for value in protocol['anchor_names'].tolist()]
    exact_ids = [
        anchor_names.index(name)
        for name in ('expert_next', 'expert_same')
        if name in anchor_names
    ]
    exact_success = metrics['anchor_success'][..., exact_ids]
    noise_ids = [i for i, name in enumerate(anchor_names) if 'noise' in name]
    print('\nExpert support audit')
    print(
        f'exact anchor success={np.mean(exact_success):.3f}, '
        f'noise anchor success={np.mean(metrics["anchor_success"][..., noise_ids]):.3f}'
    )
    print('step  scorer  exact_rank_pct  top30_success  refit_success  refit_dist  update_cos')
    supported_summary = {}
    for round_i, step in enumerate(steps):
        step_summary = {}
        for scorer_i, label in enumerate(labels):
            exact_rank = metrics['anchor_rank'][
                :, :, round_i, scorer_i, exact_ids
            ].min(axis=-1)
            values = {
                'exact_rank_percentile': float(np.mean(exact_rank)),
                'top30_success': float(
                    np.mean(
                        metrics['augmented']['elite_success'][
                            :, :, round_i, scorer_i
                        ]
                    )
                ),
                'refit_success': float(
                    np.mean(data['refit_success'][:, :, round_i, scorer_i])
                ),
                'refit_distance': float(
                    np.mean(data['refit_true'][:, :, round_i, scorer_i])
                ),
                'update_cosine': float(
                    np.nanmean(
                        data['update_cosine'][:, :, round_i, scorer_i]
                    )
                ),
            }
            step_summary[label] = values
            print(
                f'{int(step):>4}  {label:<6}  '
                f'{values["exact_rank_percentile"]:.3f}           '
                f'{values["top30_success"]:.3f}          '
                f'{values["refit_success"]:.3f}         '
                f'{values["refit_distance"]:.4f}     '
                f'{values["update_cosine"]:.3f}'
            )
        supported_summary[str(int(step))] = step_summary

    deltas = {
        'natural_proposal_coverage_k5gen_minus_k1gen': summarize_round_values(
            metrics['proposal']['coverage'][:, 1]
            - metrics['proposal']['coverage'][:, 0],
            steps,
            rng,
        ),
        'natural_oracle_min_k5gen_minus_k1gen': summarize_round_values(
            metrics['proposal']['oracle_min'][:, 1]
            - metrics['proposal']['oracle_min'][:, 0],
            steps,
            rng,
        ),
        'actual_mean_success_k5gen_minus_k1gen': summarize_round_values(
            data['recorded_mean_success'][:, 1].astype(np.float64)
            - data['recorded_mean_success'][:, 0].astype(np.float64),
            steps,
            rng,
        ),
        'actual_mean_distance_k5gen_minus_k1gen': summarize_round_values(
            data['recorded_mean_true'][:, 1]
            - data['recorded_mean_true'][:, 0],
            steps,
            rng,
        ),
        'natural_rho_k5_minus_k1': summarize_metric_delta(
            metrics['natural']['rho'], steps, rng
        ),
        'natural_top30_overlap_k5_minus_k1': summarize_metric_delta(
            metrics['natural']['topk_overlap'], steps, rng
        ),
        'augmented_top30_success_k5_minus_k1': summarize_metric_delta(
            metrics['augmented']['elite_success'], steps, rng
        ),
        'augmented_top30_success_path_interaction': summarize_round_values(
            scorer_path_interaction(
                metrics['augmented']['elite_success']
            ),
            steps,
            rng,
        ),
        'augmented_refit_success_k5_minus_k1': summarize_metric_delta(
            data['refit_success'], steps, rng
        ),
        'augmented_refit_success_path_interaction': summarize_round_values(
            scorer_path_interaction(data['refit_success']),
            steps,
            rng,
        ),
        'augmented_refit_distance_k5_minus_k1': summarize_metric_delta(
            data['refit_true'], steps, rng
        ),
        'update_cosine_k5_minus_k1': summarize_metric_delta(
            data['update_cosine'], steps, rng
        ),
    }

    exact_rank_values = metrics['anchor_rank'][..., exact_ids].min(axis=-1)
    deltas['exact_anchor_rank_pct_k5_minus_k1'] = summarize_metric_delta(
        exact_rank_values, steps, rng
    )

    print('\nPaired K5-K1 deltas at final CEM round (state bootstrap 95% CI)')
    final = str(int(steps[-1]))
    for name, values in deltas.items():
        item = values[final]
        print(
            f'{name}: {item["mean"]:.4f} '
            f'[{item["ci95"][0]:.4f}, {item["ci95"][1]:.4f}]'
        )

    return {
        'version': 1,
        'protocol': {
            'steps': steps.tolist(),
            'labels': labels,
            'generators': generators,
            'rows': data['rows'].astype(int).tolist(),
            'episodes': data['episodes'].astype(int).tolist(),
            'starts': data['starts'].astype(int).tolist(),
            'metadata': data['metadata'],
            'shard_elapsed_seconds': data['elapsed_seconds'],
            **audit,
        },
        'expert_support': {
            'exact_success_rate': float(np.mean(exact_success)),
            'noise_success_rate': float(
                np.mean(metrics['anchor_success'][..., noise_ids])
            ),
        },
        'natural_proposal': proposal_summary,
        'natural_scorer': natural_scorer_summary,
        'supported_scorer': supported_summary,
        'paired_deltas': deltas,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol, data = load_shards(args.paths)
    metrics = compute_metrics(protocol, data)
    result = report(protocol, data, metrics, seed=args.seed)
    result['inputs'] = [
        {'path': str(path.resolve()), 'sha256': sha256(path)}
        for path in args.paths
    ]
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(f'\nreport -> {args.out}')


if __name__ == '__main__':
    main()
