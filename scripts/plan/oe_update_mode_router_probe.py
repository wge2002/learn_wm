"""Nested state-cross-fitted router for discrete OE correction modes.

This is the deployability bridge after ``oe_update_mode_codebook_probe.py``.
Each outer fold learns a correction codebook only from training states,
labels those examples with the best codebook/no-op mode, and learns a small
ridge or RBF router from planner, frozen-latent, or oracle-state features.
Router type and regularization are selected by an inner state-held-out loop
using update quality rather than classification accuracy.

No held-out state participates in codebook fitting, mode labeling, router
selection, or router fitting.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from oe_update_corrector_probe import (
    HeadConfig,
    feature_matrix,
    fit_predict,
    load_trace,
    paired_bootstrap,
    selection_score,
    state_aggregate,
    update_metrics,
)
from oe_update_mode_codebook_probe import (
    oracle_route,
    spherical_kmeans,
)


def classifier_configs() -> list[tuple[HeadConfig, float]]:
    heads = [
        HeadConfig(kind='ridge', alpha=alpha, blend=blend)
        for alpha in (0.1, 1.0, 10.0, 100.0)
        for blend in (0.25, 0.5, 1.0)
    ]
    heads.extend(
        HeadConfig(
            kind='rbf',
            alpha=alpha,
            blend=blend,
            gamma_factor=gamma,
        )
        for alpha in (0.1, 1.0, 10.0)
        for gamma in (0.25, 1.0, 4.0)
        for blend in (0.25, 0.5, 1.0)
    )
    return [
        (head, gate_quantile)
        for head in heads
        for gate_quantile in (0.0, 0.25, 0.5, 0.75)
    ]


def route_labels(
    trace,
    indices: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    _, labels = oracle_route(
        trace.model_update_normalized[indices],
        trace.oracle_update_normalized[indices],
        trace.proposal_std[indices],
        prototypes,
    )
    return labels


def one_hot(labels: np.ndarray, classes: int) -> np.ndarray:
    return np.eye(classes, dtype=np.float64)[labels]


def routed_updates(
    trace,
    indices: np.ndarray,
    prototypes: np.ndarray,
    labels: np.ndarray,
    *,
    blend: float = 1.0,
) -> np.ndarray:
    codebook = np.concatenate(
        [np.zeros((1, prototypes.shape[1])), prototypes],
        axis=0,
    )
    return (
        trace.model_update_normalized[indices]
        + blend * codebook[labels]
    )


def fit_router_predict(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
    config: HeadConfig,
    *,
    classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = fit_predict(
        train_x,
        one_hot(train_labels, classes),
        test_x,
        config,
    )
    labels = np.argmax(scores, axis=1)
    confidence = np.max(scores, axis=1) - np.partition(
        scores,
        -2,
        axis=1,
    )[:, -2]
    return labels, confidence


def state_folds(states: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = np.asarray(sorted(int(state) for state in states))
    folds = []
    for fold in range(3):
        validation = ordered[np.arange(len(ordered)) % 3 == fold]
        training = ordered[~np.isin(ordered, validation)]
        folds.append((training, validation))
    return folds


def select_router(
    trace,
    features: np.ndarray,
    outer_train_states: np.ndarray,
    *,
    clusters: int,
    seed: int,
) -> tuple[HeadConfig, float, list[dict]]:
    configs = classifier_configs()
    config_scores = [[] for _ in configs]
    for inner_fold, (train_states, val_states) in enumerate(
        state_folds(outer_train_states)
    ):
        train_indices = np.flatnonzero(
            np.isin(trace.state_ids, train_states)
        )
        val_indices = np.flatnonzero(
            np.isin(trace.state_ids, val_states)
        )
        residual = (
            trace.oracle_update_normalized[train_indices]
            - trace.model_update_normalized[train_indices]
        )
        prototypes, _ = spherical_kmeans(
            residual,
            clusters=clusters,
            seed=seed + 1000 * clusters + inner_fold,
        )
        labels = route_labels(trace, train_indices, prototypes)
        prediction_cache = {}
        for config_i, (config, gate_quantile) in enumerate(configs):
            fit_key = (config.kind, config.alpha, config.gamma_factor)
            if fit_key not in prediction_cache:
                train_prediction = fit_router_predict(
                    features[train_indices],
                    labels,
                    features[train_indices],
                    config,
                    classes=clusters + 1,
                )
                val_prediction = fit_router_predict(
                    features[train_indices],
                    labels,
                    features[val_indices],
                    config,
                    classes=clusters + 1,
                )
                prediction_cache[fit_key] = (
                    train_prediction,
                    val_prediction,
                )
            (
                (_, train_confidence),
                (predicted_labels, val_confidence),
            ) = prediction_cache[fit_key]
            threshold = (
                float(np.quantile(train_confidence, gate_quantile))
                if gate_quantile > 0
                else -np.inf
            )
            predicted_labels = predicted_labels.copy()
            predicted_labels[val_confidence < threshold] = 0
            corrected = routed_updates(
                trace,
                val_indices,
                prototypes,
                predicted_labels,
                blend=config.blend,
            )
            config_scores[config_i].append(
                selection_score(
                    corrected,
                    trace,
                    val_indices,
                    val_states,
                )
            )
    rows = [
        {
            **asdict(config),
            'gate_quantile': gate_quantile,
            'inner_score': float(np.mean(scores)),
            'inner_score_min': float(np.min(scores)),
        }
        for (config, gate_quantile), scores in zip(
            configs,
            config_scores,
            strict=True,
        )
    ]
    best_index = max(
        range(len(configs)),
        key=lambda index: (
            rows[index]['inner_score'],
            rows[index]['inner_score_min'],
            configs[index][0].alpha,
            configs[index][1],
        ),
    )
    return configs[best_index][0], configs[best_index][1], rows


def analyze_family(
    trace,
    family: str,
    *,
    clusters: int,
    bootstrap: int,
    seed: int,
) -> dict:
    features = feature_matrix(trace, family)
    residual = (
        trace.oracle_update_normalized
        - trace.model_update_normalized
    )
    num_states = int(np.max(trace.state_ids)) + 1
    states = np.arange(num_states)
    corrected = np.empty_like(trace.model_update_normalized)
    routed_labels = np.empty(len(corrected), dtype=np.int64)
    oracle_corrected = np.empty_like(corrected)
    fold_rows = []
    for fold in range(3):
        val_states = states[states % 3 == fold]
        train_states = states[states % 3 != fold]
        train_indices = np.flatnonzero(
            np.isin(trace.state_ids, train_states)
        )
        val_indices = np.flatnonzero(
            np.isin(trace.state_ids, val_states)
        )
        prototypes, _ = spherical_kmeans(
            residual[train_indices],
            clusters=clusters,
            seed=seed + 10000 * clusters + fold,
        )
        train_labels = route_labels(trace, train_indices, prototypes)
        config, gate_quantile, selection_rows = select_router(
            trace,
            features,
            train_states,
            clusters=clusters,
            seed=seed + 100000 * fold,
        )
        _, train_confidence = fit_router_predict(
            features[train_indices],
            train_labels,
            features[train_indices],
            config,
            classes=clusters + 1,
        )
        val_labels, confidence = fit_router_predict(
            features[train_indices],
            train_labels,
            features[val_indices],
            config,
            classes=clusters + 1,
        )
        threshold = (
            float(np.quantile(train_confidence, gate_quantile))
            if gate_quantile > 0
            else -np.inf
        )
        val_labels[confidence < threshold] = 0
        routed_labels[val_indices] = val_labels
        corrected[val_indices] = routed_updates(
            trace,
            val_indices,
            prototypes,
            val_labels,
            blend=config.blend,
        )
        oracle_corrected[val_indices], oracle_labels = oracle_route(
            trace.model_update_normalized[val_indices],
            trace.oracle_update_normalized[val_indices],
            trace.proposal_std[val_indices],
            prototypes,
        )
        fold_rows.append(
            {
                'fold': fold,
                'train_states': train_states.tolist(),
                'val_states': val_states.tolist(),
                'selected': {
                    **asdict(config),
                    'gate_quantile': gate_quantile,
                    'confidence_threshold': threshold,
                },
                'selected_inner_score': max(
                    row['inner_score']
                    for row in selection_rows
                    if all(
                        row[key] == value
                        for key, value in asdict(config).items()
                    )
                    and row['gate_quantile'] == gate_quantile
                ),
                'train_no_op_fraction': float(
                    np.mean(train_labels == 0)
                ),
                'router_no_op_fraction': float(
                    np.mean(val_labels == 0)
                ),
                'oracle_no_op_fraction': float(
                    np.mean(oracle_labels == 0)
                ),
                'mean_confidence': float(np.mean(confidence)),
                'selection_rows': selection_rows,
            }
        )

    baseline_cosine, baseline_relative = update_metrics(
        trace.model_update_normalized,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    corrected_cosine, corrected_relative = update_metrics(
        corrected,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    oracle_cosine, oracle_relative = update_metrics(
        oracle_corrected,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    baseline_state_cosine = state_aggregate(
        baseline_cosine,
        trace.state_ids,
        states,
    )
    corrected_state_cosine = state_aggregate(
        corrected_cosine,
        trace.state_ids,
        states,
    )
    oracle_state_cosine = state_aggregate(
        oracle_cosine,
        trace.state_ids,
        states,
    )
    baseline_state_relative = state_aggregate(
        baseline_relative,
        trace.state_ids,
        states,
    )
    corrected_state_relative = state_aggregate(
        corrected_relative,
        trace.state_ids,
        states,
    )
    oracle_state_relative = state_aggregate(
        oracle_relative,
        trace.state_ids,
        states,
    )
    cosine_delta = corrected_state_cosine - baseline_state_cosine
    relative_delta = corrected_state_relative - baseline_state_relative
    rng = np.random.default_rng(seed + clusters)
    return {
        'family': family,
        'clusters': clusters,
        'feature_width': int(features.shape[1]),
        'baseline': {
            'update_cosine': float(np.mean(baseline_state_cosine)),
            'relative_update_error': float(
                np.mean(baseline_state_relative)
            ),
        },
        'router': {
            'update_cosine': float(np.mean(corrected_state_cosine)),
            'relative_update_error': float(
                np.mean(corrected_state_relative)
            ),
            'no_op_fraction': float(np.mean(routed_labels == 0)),
        },
        'oracle_router': {
            'update_cosine': float(np.mean(oracle_state_cosine)),
            'relative_update_error': float(
                np.mean(oracle_state_relative)
            ),
        },
        'delta': {
            'update_cosine': float(np.mean(cosine_delta)),
            'update_cosine_ci': list(
                paired_bootstrap(
                    cosine_delta,
                    samples=bootstrap,
                    rng=rng,
                )
            ),
            'relative_update_error': float(np.mean(relative_delta)),
            'relative_update_error_ci': list(
                paired_bootstrap(
                    relative_delta,
                    samples=bootstrap,
                    rng=rng,
                )
            ),
        },
        'folds': fold_rows,
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--latent-caches', nargs='*', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260724)
    parser.add_argument('--clusters', default='4,8')
    parser.add_argument(
        '--families',
        default='planner,planner_latent,planner_state_oracle',
    )
    args = parser.parse_args()
    if args.latent_caches and len(args.latent_caches) != len(args.sources):
        raise ValueError('provide one latent cache per source')
    clusters = [
        int(item)
        for item in args.clusters.split(',')
        if item.strip()
    ]
    families = [
        item.strip()
        for item in args.families.split(',')
        if item.strip()
    ]
    analyses = []
    for source_i, source in enumerate(args.sources):
        trace = load_trace(
            source,
            topk=args.topk,
            latent_cache=(
                args.latent_caches[source_i]
                if args.latent_caches
                else None
            ),
        )
        for cluster_count in clusters:
            for family_i, family in enumerate(families):
                result = analyze_family(
                    trace,
                    family,
                    clusters=cluster_count,
                    bootstrap=args.bootstrap,
                    seed=(
                        args.seed
                        + source_i * 1000
                        + cluster_count * 10
                        + family_i
                    ),
                )
                result['cell'] = trace.label
                result['source'] = str(trace.source)
                analyses.append(result)
                print(
                    f'{trace.label} K={cluster_count} {family}: '
                    f'cos {result["baseline"]["update_cosine"]:.3f}'
                    f'->{result["router"]["update_cosine"]:.3f} '
                    f'({result["delta"]["update_cosine"]:+.3f}), '
                    f'rel '
                    f'{result["baseline"]["relative_update_error"]:.3f}'
                    f'->{result["router"]["relative_update_error"]:.3f} '
                    f'({result["delta"]["relative_update_error"]:+.3f}); '
                    f'oracle cos='
                    f'{result["oracle_router"]["update_cosine"]:.3f}',
                    flush=True,
                )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'clusters': clusters,
        'families': families,
        'analyses': analyses,
    }
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    rows = []
    for result in analyses:
        rows.append(
            {
                'cell': result['cell'],
                'clusters': result['clusters'],
                'family': result['family'],
                'baseline_cosine': result['baseline']['update_cosine'],
                'router_cosine': result['router']['update_cosine'],
                'oracle_router_cosine': result['oracle_router'][
                    'update_cosine'
                ],
                'cosine_delta': result['delta']['update_cosine'],
                'cosine_delta_ci_low': result['delta'][
                    'update_cosine_ci'
                ][0],
                'cosine_delta_ci_high': result['delta'][
                    'update_cosine_ci'
                ][1],
                'baseline_relative_error': result['baseline'][
                    'relative_update_error'
                ],
                'router_relative_error': result['router'][
                    'relative_update_error'
                ],
                'oracle_router_relative_error': result['oracle_router'][
                    'relative_update_error'
                ],
                'relative_error_delta': result['delta'][
                    'relative_update_error'
                ],
                'relative_error_delta_ci_low': result['delta'][
                    'relative_update_error_ci'
                ][0],
                'relative_error_delta_ci_high': result['delta'][
                    'relative_update_error_ci'
                ][1],
                'router_no_op_fraction': result['router'][
                    'no_op_fraction'
                ],
            }
        )
    with (args.out_dir / 'metrics.csv').open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# Discrete OE update-mode router probe',
        '',
        'Every row is nested state-cross-fitted. Oracle-router columns are '
        'codebook ceilings; deployable router columns use only the named '
        'feature family.',
        '',
        '| cell | K | features | baseline cosine | router cosine | oracle '
        'cosine | Δ cosine | baseline rel. | router rel. | oracle rel. '
        '| Δ rel. |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for result in analyses:
        lines.append(
            f'| {result["cell"]} | {result["clusters"]} '
            f'| `{result["family"]}` '
            f'| {result["baseline"]["update_cosine"]:.3f} '
            f'| {result["router"]["update_cosine"]:.3f} '
            f'| {result["oracle_router"]["update_cosine"]:.3f} '
            f'| {result["delta"]["update_cosine"]:+.3f} '
            f'{format_ci(result["delta"]["update_cosine_ci"])} '
            f'| {result["baseline"]["relative_update_error"]:.3f} '
            f'| {result["router"]["relative_update_error"]:.3f} '
            f'| {result["oracle_router"]["relative_update_error"]:.3f} '
            f'| {result["delta"]["relative_update_error"]:+.3f} '
            f'{format_ci(result["delta"]["relative_update_error_ci"])} |'
        )
    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print(f'results -> {args.out_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
