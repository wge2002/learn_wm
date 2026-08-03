"""State-cross-fitted probe for a frozen-WM CEM update corrector.

The OE fine-tuning experiments establish a causal target but fail to make the
large dynamics backbone generalize that target.  This probe asks a narrower
question before collecting substantially more labels:

    Can a small module predict the simulator-induced CEM *mean update* while
    the world model remains frozen?

Two feature families form a learnability ladder:

``planner``
    Deployable statistics already available inside CEM: the learned mean and
    log-standard-deviation update, the current proposal, round identity, and
    predicted-cost quantiles.

``planner_state_oracle``
    The same features plus the true PushT initial/goal state.  This is not a
    deployable method.  It is an upper-bound diagnostic: if even these features
    fail under held-out-state evaluation, a frozen-latent head is unlikely to
    be the immediate bottleneck.

The corrector predicts the oracle update in proposal-standard-deviation units.
All model class, regularization, kernel bandwidth, and residual blending
choices are selected only by nested state-held-out cross-validation inside
each outer training fold.  The outer 3-fold predictions therefore cover every
state exactly once without checkpoint or hyperparameter selection on it.

This is a fixed-trace feasibility test, not a deployable planning result.
Promotion still requires recursive re-sampling and closed-loop MPC.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


EPS = 1e-8


@dataclass(frozen=True)
class HeadConfig:
    kind: str
    alpha: float
    blend: float
    gamma_factor: float = 0.0


@dataclass
class TraceData:
    label: str
    source: Path
    horizon: int
    goal_offset: int
    steps: np.ndarray
    state_features: np.ndarray
    latent_features: np.ndarray | None
    planner_features: np.ndarray
    planner_history_features: np.ndarray
    ensemble_features: np.ndarray | None
    ensemble_history_features: np.ndarray | None
    model_update_normalized: np.ndarray
    oracle_update_normalized: np.ndarray
    proposal_std: np.ndarray
    state_ids: np.ndarray
    round_ids: np.ndarray


def elite_moments(
    candidates: np.ndarray,
    costs: np.ndarray,
    *,
    topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(costs, kind='stable')[:topk]
    elite = candidates[indices].astype(np.float64)
    mean = elite.mean(axis=0)
    std = elite.std(axis=0, ddof=1)
    return mean, np.maximum(std, 1e-6)


def cost_features(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    quantiles = np.quantile(
        cost,
        [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0],
    )
    median = quantiles[5]
    robust_scale = max((quantiles[6] - quantiles[4]) / 1.349, 1e-6)
    normalized = (quantiles - median) / robust_scale
    return np.concatenate(
        [
            normalized,
            np.asarray(
                [
                    np.mean(cost),
                    np.std(cost),
                    median,
                    robust_scale,
                ],
                dtype=np.float64,
            ),
        ]
    )


def load_latent_cache(
    path: Path,
    *,
    source_rows: np.ndarray,
) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {'features', 'rows'}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{path} is missing latent fields {missing}')
        features = np.asarray(archive['features']).astype(np.float64)
        rows = np.asarray(archive['rows']).astype(np.int64)
    if not np.array_equal(rows, source_rows):
        raise ValueError(
            f'latent-cache/source row mismatch for {path}: '
            f'{np.sum(rows != source_rows)} differing rows'
        )
    if features.ndim != 2 or len(features) != len(rows):
        raise ValueError(
            f'latent features must be (states,D), got {features.shape}'
        )
    return features


def causal_history_features(
    features: np.ndarray,
    *,
    state_ids: np.ndarray,
    round_ids: np.ndarray,
    num_states: int,
    num_rounds: int,
) -> np.ndarray:
    width = features.shape[1]
    shaped = np.empty((num_states, num_rounds, width), dtype=np.float64)
    shaped[state_ids, round_ids] = features
    rows = []
    for state_i, round_i in zip(state_ids, round_ids, strict=True):
        history = np.zeros((num_rounds, width), dtype=np.float64)
        history[: round_i + 1] = shaped[state_i, : round_i + 1]
        mask = np.zeros(num_rounds, dtype=np.float64)
        mask[: round_i + 1] = 1.0
        rows.append(np.concatenate([history.reshape(-1), mask]))
    return np.asarray(rows)


def load_trace(
    path: Path,
    *,
    topk: int,
    latent_cache: Path | None = None,
) -> TraceData:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            'candidates',
            'pred',
            'true',
            'prev_mean',
            'prev_var',
            'initial_state',
            'goal_state',
            'steps',
            'horizon',
            'goal_offset',
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{path} is missing fields {missing}')
        candidates = np.asarray(archive['candidates'])[:, 0]
        predicted_all = np.asarray(archive['pred'])[:, 0]
        predicted = predicted_all[:, :, 0]
        true_cost = np.asarray(archive['true'])[:, 0]
        prev_mean = np.asarray(archive['prev_mean'])[:, 0].astype(np.float64)
        prev_var = np.asarray(archive['prev_var'])[:, 0].astype(np.float64)
        initial = np.asarray(archive['initial_state']).astype(np.float64)
        goal = np.asarray(archive['goal_state']).astype(np.float64)
        source_rows = np.asarray(archive['rows']).astype(np.int64)
        steps = np.asarray(archive['steps']).astype(np.int64)
        horizon = int(np.asarray(archive['horizon']).item())
        goal_offset = int(np.asarray(archive['goal_offset']).item())

    if candidates.shape[:3] != true_cost.shape:
        raise ValueError(
            'candidate/true shape mismatch: '
            f'{candidates.shape} vs {true_cost.shape}'
        )
    num_states, num_rounds, num_candidates = true_cost.shape
    if topk < 2 or topk >= num_candidates:
        raise ValueError(f'topk={topk} is invalid for N={num_candidates}')
    if len(steps) != num_rounds:
        raise ValueError('step count does not match recorded rounds')

    state_width = initial.shape[-1]
    one_hot_round = np.eye(num_rounds, dtype=np.float64)
    planner_rows = []
    ensemble_rows = []
    state_rows = []
    model_updates = []
    oracle_updates = []
    proposal_stds = []
    state_ids = []
    round_ids = []
    for state_i in range(num_states):
        physical = np.concatenate(
            [
                initial[state_i],
                goal[state_i],
                goal[state_i] - initial[state_i],
                np.abs(goal[state_i] - initial[state_i]),
            ]
        )
        if len(physical) != 4 * state_width:
            raise AssertionError('unexpected physical feature width')
        for round_i in range(num_rounds):
            population = candidates[state_i, round_i].astype(np.float64)
            proposal_mean = prev_mean[state_i, round_i]
            # CEM keeps the historical ``var`` name, but the solver updates
            # this tensor with ``topk_candidates.std(dim=1)`` and samples as
            # ``noise * batch_var + batch_mean``.  The trace therefore stores
            # a standard deviation already; taking another square root would
            # put every correction in the wrong proposal coordinates.
            proposal_std = np.maximum(
                prev_var[state_i, round_i],
                1e-8,
            )
            learned_mean, learned_std = elite_moments(
                population,
                predicted[state_i, round_i],
                topk=topk,
            )
            oracle_mean, _ = elite_moments(
                population,
                true_cost[state_i, round_i],
                topk=topk,
            )
            model_update = (
                (learned_mean - proposal_mean) / proposal_std
            ).reshape(-1)
            oracle_update = (
                (oracle_mean - proposal_mean) / proposal_std
            ).reshape(-1)
            model_logstd = np.log(
                np.maximum(learned_std, 1e-6) / proposal_std
            ).reshape(-1)
            planner = np.concatenate(
                [
                    model_update,
                    model_logstd,
                    proposal_mean.reshape(-1),
                    np.log(proposal_std).reshape(-1),
                    one_hot_round[round_i],
                    cost_features(predicted[state_i, round_i]),
                ]
            )
            planner_rows.append(planner)
            if predicted_all.shape[2] > 1:
                extra_scorers = []
                for scorer_i in range(1, predicted_all.shape[2]):
                    scorer_cost = predicted_all[
                        state_i,
                        round_i,
                        scorer_i,
                    ]
                    scorer_mean, scorer_std = elite_moments(
                        population,
                        scorer_cost,
                        topk=topk,
                    )
                    scorer_update = (
                        (scorer_mean - proposal_mean) / proposal_std
                    ).reshape(-1)
                    scorer_logstd = np.log(
                        np.maximum(scorer_std, 1e-6) / proposal_std
                    ).reshape(-1)
                    extra_scorers.extend(
                        [
                            scorer_update,
                            scorer_update - model_update,
                            scorer_logstd,
                            cost_features(scorer_cost),
                        ]
                    )
                ensemble_rows.append(
                    np.concatenate([planner, *extra_scorers])
                )
            state_rows.append(physical)
            model_updates.append(model_update)
            oracle_updates.append(oracle_update)
            proposal_stds.append(proposal_std.reshape(-1))
            state_ids.append(state_i)
            round_ids.append(round_i)

    label = f'h{horizon}_off{goal_offset}'
    latent = (
        load_latent_cache(latent_cache, source_rows=source_rows)
        if latent_cache is not None
        else None
    )
    planner_array = np.asarray(planner_rows)
    ensemble_array = (
        np.asarray(ensemble_rows)
        if ensemble_rows
        else None
    )
    state_id_array = np.asarray(state_ids, dtype=np.int64)
    round_id_array = np.asarray(round_ids, dtype=np.int64)
    return TraceData(
        label=label,
        source=path.resolve(),
        horizon=horizon,
        goal_offset=goal_offset,
        steps=steps,
        state_features=np.asarray(state_rows),
        latent_features=latent,
        planner_features=planner_array,
        planner_history_features=causal_history_features(
            planner_array,
            state_ids=state_id_array,
            round_ids=round_id_array,
            num_states=num_states,
            num_rounds=num_rounds,
        ),
        ensemble_features=ensemble_array,
        ensemble_history_features=(
            causal_history_features(
                ensemble_array,
                state_ids=state_id_array,
                round_ids=round_id_array,
                num_states=num_states,
                num_rounds=num_rounds,
            )
            if ensemble_array is not None
            else None
        ),
        model_update_normalized=np.asarray(model_updates),
        oracle_update_normalized=np.asarray(oracle_updates),
        proposal_std=np.asarray(proposal_stds),
        state_ids=state_id_array,
        round_ids=round_id_array,
    )


def feature_matrix(trace: TraceData, family: str) -> np.ndarray:
    if family == 'planner':
        return trace.planner_features
    if family == 'planner_state_oracle':
        return np.concatenate(
            [trace.planner_features, trace.state_features],
            axis=1,
        )
    if family == 'planner_latent':
        if trace.latent_features is None:
            raise ValueError(
                f'{trace.label} has no latent cache for planner_latent'
            )
        repeated = trace.latent_features[trace.state_ids]
        return np.concatenate(
            [trace.planner_features, repeated],
            axis=1,
        )
    if family == 'planner_history':
        return trace.planner_history_features
    if family == 'planner_history_latent':
        if trace.latent_features is None:
            raise ValueError(
                f'{trace.label} has no latent cache'
            )
        repeated = trace.latent_features[trace.state_ids]
        return np.concatenate(
            [trace.planner_history_features, repeated],
            axis=1,
        )
    if family == 'planner_ensemble':
        if trace.ensemble_features is None:
            raise ValueError(
                f'{trace.label} has no additional scorer predictions'
            )
        return trace.ensemble_features
    if family == 'planner_ensemble_latent':
        if trace.ensemble_features is None or trace.latent_features is None:
            raise ValueError(
                f'{trace.label} requires ensemble and latent features'
            )
        repeated = trace.latent_features[trace.state_ids]
        return np.concatenate(
            [trace.ensemble_features, repeated],
            axis=1,
        )
    if family == 'planner_ensemble_history':
        if trace.ensemble_history_features is None:
            raise ValueError(
                f'{trace.label} has no ensemble history'
            )
        return trace.ensemble_history_features
    if family == 'planner_ensemble_history_latent':
        if (
            trace.ensemble_history_features is None
            or trace.latent_features is None
        ):
            raise ValueError(
                f'{trace.label} requires ensemble history and latent'
            )
        repeated = trace.latent_features[trace.state_ids]
        return np.concatenate(
            [trace.ensemble_history_features, repeated],
            axis=1,
        )
    raise ValueError(f'unknown feature family {family!r}')


def standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return mean, scale


def squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = (
        np.sum(left * left, axis=1, keepdims=True)
        + np.sum(right * right, axis=1)[None]
        - 2.0 * left @ right.T
    )
    return np.maximum(distance, 0.0)


def fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: HeadConfig,
) -> np.ndarray:
    x_mean, x_scale = standardize_fit(train_x)
    train_xn = (train_x - x_mean) / x_scale
    test_xn = (test_x - x_mean) / x_scale
    y_mean, y_scale = standardize_fit(train_y)
    train_yn = (train_y - y_mean) / y_scale

    if config.kind == 'ridge':
        num_examples, num_features = train_xn.shape
        if num_features <= num_examples:
            gram = train_xn.T @ train_xn
            gram.flat[:: num_features + 1] += config.alpha
            weights = np.linalg.solve(gram, train_xn.T @ train_yn)
        else:
            gram = train_xn @ train_xn.T
            gram.flat[:: num_examples + 1] += config.alpha
            dual = np.linalg.solve(gram, train_yn)
            weights = train_xn.T @ dual
        predicted = test_xn @ weights
    elif config.kind == 'rbf':
        train_distance = squared_distances(train_xn, train_xn)
        positive = train_distance[train_distance > 1e-12]
        median = float(np.median(positive)) if len(positive) else 1.0
        gamma = config.gamma_factor / max(median, 1e-8)
        kernel = np.exp(-gamma * train_distance)
        kernel.flat[:: len(kernel) + 1] += config.alpha
        dual = np.linalg.solve(kernel, train_yn)
        test_kernel = np.exp(
            -gamma * squared_distances(test_xn, train_xn)
        )
        predicted = test_kernel @ dual
    else:
        raise ValueError(f'unknown head kind {config.kind!r}')
    return predicted * y_scale + y_mean


def apply_blend(
    baseline: np.ndarray,
    predicted_target: np.ndarray,
    blend: float,
) -> np.ndarray:
    return baseline + blend * (predicted_target - baseline)


def update_metrics(
    predicted_normalized: np.ndarray,
    oracle_normalized: np.ndarray,
    proposal_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = predicted_normalized * proposal_std
    oracle = oracle_normalized * proposal_std
    numerator = np.sum(predicted * oracle, axis=1)
    predicted_norm = np.linalg.norm(predicted, axis=1)
    oracle_norm = np.linalg.norm(oracle, axis=1)
    cosine = numerator / np.maximum(predicted_norm * oracle_norm, EPS)
    relative_error = (
        np.linalg.norm(predicted - oracle, axis=1)
        / np.maximum(oracle_norm, EPS)
    )
    return cosine, relative_error


def state_aggregate(
    values: np.ndarray,
    state_ids: np.ndarray,
    states: Iterable[int],
) -> np.ndarray:
    return np.asarray(
        [np.mean(values[state_ids == state]) for state in states],
        dtype=np.float64,
    )


def selection_score(
    predicted: np.ndarray,
    trace: TraceData,
    example_indices: np.ndarray,
    states: np.ndarray,
) -> float:
    cosine, relative = update_metrics(
        predicted,
        trace.oracle_update_normalized[example_indices],
        trace.proposal_std[example_indices],
    )
    example_states = trace.state_ids[example_indices]
    state_cosine = state_aggregate(cosine, example_states, states)
    state_relative = state_aggregate(relative, example_states, states)
    return float(np.mean(state_cosine - 0.5 * state_relative))


def candidate_configs() -> list[HeadConfig]:
    blends = (0.25, 0.5, 1.0)
    configs = [
        HeadConfig(kind='ridge', alpha=alpha, blend=blend)
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0)
        for blend in blends
    ]
    configs.extend(
        HeadConfig(
            kind='rbf',
            alpha=alpha,
            blend=blend,
            gamma_factor=gamma,
        )
        for alpha in (0.01, 0.1, 1.0, 10.0)
        for gamma in (0.25, 1.0, 4.0)
        for blend in blends
    )
    return configs


def inner_state_folds(states: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    ordered = np.asarray(sorted(int(state) for state in states))
    folds = []
    for fold in range(3):
        validation = ordered[np.arange(len(ordered)) % 3 == fold]
        training = ordered[~np.isin(ordered, validation)]
        folds.append((training, validation))
    return folds


def nested_select(
    trace: TraceData,
    features: np.ndarray,
    outer_train_states: np.ndarray,
) -> tuple[HeadConfig, list[dict]]:
    configs = candidate_configs()
    scores = [[] for _ in configs]
    for inner_train_states, inner_val_states in inner_state_folds(
        outer_train_states
    ):
        train_indices = np.flatnonzero(
            np.isin(trace.state_ids, inner_train_states)
        )
        val_indices = np.flatnonzero(
            np.isin(trace.state_ids, inner_val_states)
        )
        prediction_cache: dict[tuple, np.ndarray] = {}
        for config_i, config in enumerate(configs):
            fit_key = (config.kind, config.alpha, config.gamma_factor)
            if fit_key not in prediction_cache:
                prediction_cache[fit_key] = fit_predict(
                    features[train_indices],
                    trace.oracle_update_normalized[train_indices],
                    features[val_indices],
                    config,
                )
            corrected = apply_blend(
                trace.model_update_normalized[val_indices],
                prediction_cache[fit_key],
                config.blend,
            )
            scores[config_i].append(
                selection_score(
                    corrected,
                    trace,
                    val_indices,
                    inner_val_states,
                )
            )
    rows = []
    for config, fold_scores in zip(configs, scores, strict=True):
        rows.append(
            {
                **asdict(config),
                'inner_score': float(np.mean(fold_scores)),
                'inner_score_min': float(np.min(fold_scores)),
            }
        )
    # Mean inner score is primary.  Worst-fold score and stronger
    # regularization provide deterministic conservative tie breaks.
    best_i = max(
        range(len(configs)),
        key=lambda index: (
            rows[index]['inner_score'],
            rows[index]['inner_score_min'],
            configs[index].alpha,
            -configs[index].blend,
        ),
    )
    return configs[best_i], rows


def paired_bootstrap(
    delta: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if samples < 1:
        return float('nan'), float('nan')
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = np.mean(delta[draws], axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def analyze_family(
    trace: TraceData,
    family: str,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    features = feature_matrix(trace, family)
    num_states = int(np.max(trace.state_ids)) + 1
    corrected = np.empty_like(trace.oracle_update_normalized)
    outer_rows = []
    all_states = np.arange(num_states, dtype=np.int64)
    for fold in range(3):
        val_states = all_states[all_states % 3 == fold]
        train_states = all_states[all_states % 3 != fold]
        config, selection_rows = nested_select(
            trace,
            features,
            train_states,
        )
        train_indices = np.flatnonzero(
            np.isin(trace.state_ids, train_states)
        )
        val_indices = np.flatnonzero(np.isin(trace.state_ids, val_states))
        target = fit_predict(
            features[train_indices],
            trace.oracle_update_normalized[train_indices],
            features[val_indices],
            config,
        )
        corrected[val_indices] = apply_blend(
            trace.model_update_normalized[val_indices],
            target,
            config.blend,
        )
        outer_rows.append(
            {
                'fold': fold,
                'train_states': train_states.tolist(),
                'val_states': val_states.tolist(),
                'selected': asdict(config),
                'selected_inner_score': max(
                    row['inner_score']
                    for row in selection_rows
                    if all(
                        row[key] == value
                        for key, value in asdict(config).items()
                    )
                ),
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
    states = np.arange(num_states)
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
    cosine_delta = corrected_state_cosine - baseline_state_cosine
    relative_delta = corrected_state_relative - baseline_state_relative
    rng = np.random.default_rng(seed)
    cosine_ci = paired_bootstrap(
        cosine_delta,
        samples=bootstrap,
        rng=rng,
    )
    relative_ci = paired_bootstrap(
        relative_delta,
        samples=bootstrap,
        rng=rng,
    )
    state_rows = [
        {
            'state_index': int(state),
            'baseline_cosine': float(baseline_state_cosine[state]),
            'corrected_cosine': float(corrected_state_cosine[state]),
            'cosine_delta': float(cosine_delta[state]),
            'baseline_relative_error': float(
                baseline_state_relative[state]
            ),
            'corrected_relative_error': float(
                corrected_state_relative[state]
            ),
            'relative_error_delta': float(relative_delta[state]),
        }
        for state in states
    ]
    return {
        'family': family,
        'feature_width': int(features.shape[1]),
        'num_states': num_states,
        'num_examples': int(len(features)),
        'baseline': {
            'update_cosine': float(np.mean(baseline_state_cosine)),
            'relative_update_error': float(
                np.mean(baseline_state_relative)
            ),
        },
        'corrected': {
            'update_cosine': float(np.mean(corrected_state_cosine)),
            'relative_update_error': float(
                np.mean(corrected_state_relative)
            ),
        },
        'delta': {
            'update_cosine': float(np.mean(cosine_delta)),
            'update_cosine_ci': list(cosine_ci),
            'relative_update_error': float(np.mean(relative_delta)),
            'relative_update_error_ci': list(relative_ci),
        },
        'outer_folds': outer_rows,
        'state_rows': state_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def write_report(path: Path, analyses: list[dict]) -> None:
    lines = [
        '# Frozen-WM CEM update-corrector learnability probe',
        '',
        'All corrector choices are selected by nested state-held-out CV inside '
        'each outer training fold. The reported rows pool exactly one '
        'out-of-fold prediction per state.',
        '',
        '| cell | features | baseline cosine | corrected cosine | Δ cosine '
        '| baseline rel. error | corrected rel. error | Δ rel. error |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for analysis in analyses:
        delta = analysis['delta']
        lines.append(
            f'| {analysis["cell"]} | `{analysis["family"]}` '
            f'| {analysis["baseline"]["update_cosine"]:.3f} '
            f'| {analysis["corrected"]["update_cosine"]:.3f} '
            f'| {delta["update_cosine"]:+.3f} '
            f'{format_ci(delta["update_cosine_ci"])} '
            f'| {analysis["baseline"]["relative_update_error"]:.3f} '
            f'| {analysis["corrected"]["relative_update_error"]:.3f} '
            f'| {delta["relative_update_error"]:+.3f} '
            f'{format_ci(delta["relative_update_error_ci"])} |'
        )
    lines.extend(
        [
            '',
            'The exploratory promotion gate is Δ cosine >= +0.10 and '
            'Δ relative error <= -0.10 on deployable planner features, with '
            'the second horizon directionally consistent. '
            '`planner_state_oracle` is only a learnability upper bound.',
            '',
        ]
    )
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260722)
    parser.add_argument(
        '--families',
        default='planner,planner_latent,planner_state_oracle',
    )
    parser.add_argument(
        '--latent-caches',
        nargs='*',
        type=Path,
        default=None,
        help='Optional latent caches in the same order as sources.',
    )
    args = parser.parse_args()

    families = [
        item.strip()
        for item in args.families.split(',')
        if item.strip()
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.latent_caches is not None and len(args.latent_caches) not in (
        0,
        len(args.sources),
    ):
        raise ValueError(
            '--latent-caches must be omitted or contain one path per source'
        )
    analyses = []
    for source_i, source in enumerate(args.sources):
        latent_cache = (
            args.latent_caches[source_i]
            if args.latent_caches
            else None
        )
        trace = load_trace(
            source,
            topk=args.topk,
            latent_cache=latent_cache,
        )
        for family_i, family in enumerate(families):
            analysis = analyze_family(
                trace,
                family,
                bootstrap=args.bootstrap,
                seed=args.seed + 100 * source_i + family_i,
            )
            analysis['cell'] = trace.label
            analysis['source'] = str(trace.source)
            analysis['steps'] = trace.steps.tolist()
            analyses.append(analysis)
            print(
                f'{trace.label} {family}: '
                f'cos {analysis["baseline"]["update_cosine"]:.3f}'
                f'->{analysis["corrected"]["update_cosine"]:.3f} '
                f'({analysis["delta"]["update_cosine"]:+.3f}), '
                f'rel '
                f'{analysis["baseline"]["relative_update_error"]:.3f}'
                f'->{analysis["corrected"]["relative_update_error"]:.3f} '
                f'({analysis["delta"]["relative_update_error"]:+.3f})',
                flush=True,
            )

    payload = {
        'version': 1,
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'families': families,
        'analyses': analyses,
    }
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    flat_rows = []
    for analysis in analyses:
        for state_row in analysis['state_rows']:
            flat_rows.append(
                {
                    'cell': analysis['cell'],
                    'family': analysis['family'],
                    **state_row,
                }
            )
    write_csv(args.out_dir / 'state_metrics.csv', flat_rows)
    write_report(args.out_dir / 'report.md', analyses)
    print(f'results -> {args.out_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
