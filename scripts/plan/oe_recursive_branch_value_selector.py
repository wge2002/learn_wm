"""Fit an outcome-aligned selector for recursively retained BP-OE branches.

The set-valued operator is trained to cover oracle CEM update modes, so its
router probability is not a calibrated estimate of final planning value.
This probe keeps branch generation fixed and learns only the final choice
between the two retained branches.  Hyperparameters and feature families are
selected with shard-held-out cross-validation on an OOF branch bank; the
fresh evaluation directory is read only after that selection is complete.

The selector is deliberately small: an antisymmetric ridge head predicts
``true_cost(branch 0) - true_cost(branch 1)`` from branch-feature differences.
Swapping branch order therefore flips the decision by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', type=Path, required=True)
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--method',
        default='bp_sparse_matched',
    )
    parser.add_argument(
        '--baseline-method',
        default='k3_1x300',
    )
    parser.add_argument('--bootstrap', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=20260720)
    return parser.parse_args()


def load_shards(directory: Path) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob('shard_*.npz'))
    if not paths:
        raise FileNotFoundError(f'no shard_*.npz files in {directory}')
    shards = []
    groups = []
    for group_i, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            shard = {
                key: np.asarray(archive[key])
                for key in archive.files
                if key != 'audit'
            }
        state_count = len(shard['rows'])
        shards.append(shard)
        groups.append(
            np.full(state_count, group_i, dtype=np.int16)
        )
    shared = {
        'method_names',
        'selector_names',
    }
    result: dict[str, np.ndarray] = {}
    for key in shards[0]:
        if key in shared:
            reference = shards[0][key]
            for shard in shards[1:]:
                if not np.array_equal(reference, shard[key]):
                    raise ValueError(
                        f'{key} differs across shards in {directory}'
                    )
            result[key] = reference
            continue
        if (
            shards[0][key].ndim
            and shards[0][key].shape[0] == len(shards[0]['rows'])
        ):
            result[key] = np.concatenate(
                [shard[key] for shard in shards],
                axis=0,
            )
    result['groups'] = np.concatenate(groups)
    order = np.argsort(result['rows'], kind='stable')
    for key, value in list(result.items()):
        if (
            key not in shared
            and value.ndim
            and value.shape[0] == len(order)
        ):
            result[key] = value[order]
    if len(np.unique(result['rows'])) != len(result['rows']):
        raise ValueError(f'duplicate rows in {directory}')
    return result


def method_index(data: dict[str, np.ndarray], name: str) -> int:
    names = data['method_names'].astype(str).tolist()
    if name not in names:
        raise ValueError(f'method {name!r} absent from {names}')
    return names.index(name)


def selector_index(data: dict[str, np.ndarray], name: str) -> int:
    names = data['selector_names'].astype(str).tolist()
    if name not in names:
        raise ValueError(f'selector {name!r} absent from {names}')
    return names.index(name)


def branch_feature_groups(
    data: dict[str, np.ndarray],
    *,
    method: str,
) -> dict[str, np.ndarray]:
    method_i = method_index(data, method)
    required = {
        'final_model_cost',
        'final_model_relative',
        'proposal_mean_history',
        'proposal_std_history',
        'branch_log_score_history',
        'selected_modes',
    }
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f'branch archive misses {missing}')

    model_cost = data['final_model_cost'][:, method_i]
    relative = data['final_model_relative'][:, method_i]
    means = data['proposal_mean_history'][:, method_i, 1:]
    stds = data['proposal_std_history'][:, method_i, 1:]
    log_score = data['branch_log_score_history'][:, method_i, 1:]
    modes = data['selected_modes'][:, method_i]
    if (
        model_cost.shape[1] != 2
        or relative.shape[1] != 2
        or means.shape[2] != 2
    ):
        raise ValueError('selector expects exactly two retained branches')

    num_states, num_rounds, _, horizon, action_dim = means.shape
    mode_count = max(5, int(np.max(modes)) + 1)
    mode_one_hot = np.zeros(
        (num_states, 2, num_rounds, mode_count),
        dtype=np.float64,
    )
    for state_i in range(num_states):
        for round_i in range(num_rounds):
            for branch_i in range(2):
                mode = int(modes[state_i, round_i, branch_i])
                if mode >= 0:
                    mode_one_hot[
                        state_i, branch_i, round_i, mode
                    ] = 1.0

    # Compact per-step summaries preserve the temporal shape without making
    # the scalar-only control depend on hundreds of raw coordinates.
    action_rms = np.sqrt(np.mean(np.square(means), axis=-1))
    std_mean = np.mean(stds, axis=-1)
    movement = np.zeros_like(action_rms)
    movement[:, 1:] = np.sqrt(
        np.mean(
            np.square(means[:, 1:] - means[:, :-1]),
            axis=-1,
        )
    )
    disagreement = relative[:, :, 1] - relative[:, :, 0]

    scalar = np.concatenate(
        [
            model_cost,
            np.swapaxes(log_score, 1, 2),
            mode_one_hot.reshape(num_states, 2, -1),
            np.swapaxes(action_rms, 1, 2).reshape(
                num_states,
                2,
                num_rounds * horizon,
            ),
            np.swapaxes(std_mean, 1, 2).reshape(
                num_states,
                2,
                num_rounds * horizon,
            ),
            np.swapaxes(movement, 1, 2).reshape(
                num_states,
                2,
                num_rounds * horizon,
            ),
        ],
        axis=-1,
    )
    outcome = np.concatenate(
        [
            relative.reshape(num_states, 2, -1),
            disagreement,
        ],
        axis=-1,
    )
    action = np.concatenate(
        [
            np.swapaxes(means, 1, 2).reshape(
                num_states,
                2,
                num_rounds * horizon * action_dim,
            ),
            np.log(
                np.maximum(
                    np.swapaxes(stds, 1, 2),
                    1e-6,
                )
            ).reshape(
                num_states,
                2,
                num_rounds * horizon * action_dim,
            ),
        ],
        axis=-1,
    )
    groups = {
        'scalar': scalar,
        'outcome': np.concatenate([scalar, outcome], axis=-1),
        'action': np.concatenate([scalar, action], axis=-1),
        'full': np.concatenate([scalar, outcome, action], axis=-1),
    }
    for name, values in groups.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f'non-finite {name} branch features')
    return groups


def pair_differences(
    branch_features: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        name: values[:, 0] - values[:, 1]
        for name, values in branch_features.items()
    }


def fit_ridge(
    features: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    active = scale > 1e-6
    normalized = (
        features[:, active] - mean[active]
    ) / scale[active]
    target_mean = float(target.mean())
    centered = target - target_mean
    u, singular, vt = np.linalg.svd(
        normalized,
        full_matrices=False,
    )
    shrink = singular / (
        np.square(singular) + alpha * len(features)
    )
    weight = vt.T @ (shrink * (u.T @ centered))
    return {
        'mean': mean,
        'scale': scale,
        'active': active,
        'weight': weight,
        'target_mean': target_mean,
    }


def predict_ridge(
    model: dict[str, np.ndarray | float],
    features: np.ndarray,
) -> np.ndarray:
    active = np.asarray(model['active'], dtype=bool)
    normalized = (
        features[:, active] - np.asarray(model['mean'])[active]
    ) / np.asarray(model['scale'])[active]
    return (
        normalized @ np.asarray(model['weight'])
        + float(model['target_mean'])
    )


def target_values(
    cost_difference: np.ndarray,
    *,
    kind: str,
    reference: np.ndarray,
) -> np.ndarray:
    if kind == 'sign':
        return np.sign(cost_difference)
    if kind == 'clipped_delta':
        scale = max(
            float(np.median(np.abs(reference))),
            1e-6,
        )
        return np.clip(cost_difference / scale, -3.0, 3.0)
    raise ValueError(kind)


def choose_cost(
    prediction: np.ndarray,
    pair_cost: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = (prediction >= 0).astype(np.int16)
    cost = pair_cost[np.arange(len(pair_cost)), chosen]
    return chosen, cost


def cross_validate(
    features: dict[str, np.ndarray],
    pair_cost: np.ndarray,
    groups: np.ndarray,
) -> tuple[dict, list[dict], np.ndarray]:
    cost_difference = pair_cost[:, 0] - pair_cost[:, 1]
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        unique_groups = np.arange(5)
        groups = np.arange(len(pair_cost)) % 5
    cap = float(np.quantile(np.abs(cost_difference), 0.9))
    candidates = []
    prediction_by_candidate = []
    for family in ('scalar', 'outcome', 'action', 'full'):
        for target_kind in ('sign', 'clipped_delta'):
            for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                oof = np.full(len(pair_cost), np.nan, dtype=np.float64)
                for group in unique_groups:
                    validation = groups == group
                    training = ~validation
                    target = target_values(
                        cost_difference[training],
                        kind=target_kind,
                        reference=cost_difference[training],
                    )
                    model = fit_ridge(
                        features[family][training],
                        target,
                        alpha=alpha,
                    )
                    oof[validation] = predict_ridge(
                        model,
                        features[family][validation],
                    )
                chosen, selected = choose_cost(oof, pair_cost)
                wrong = (
                    chosen
                    != np.argmin(pair_cost, axis=1)
                )
                capped_regret = np.minimum(
                    np.abs(cost_difference),
                    cap,
                ) * wrong
                candidates.append(
                    {
                        'family': family,
                        'target': target_kind,
                        'alpha': alpha,
                        'capped_regret': float(capped_regret.mean()),
                        'mean_selected_cost': float(selected.mean()),
                        'median_selected_cost': float(
                            np.median(selected)
                        ),
                        'accuracy': float(1.0 - wrong.mean()),
                    }
                )
                prediction_by_candidate.append(oof)
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            candidates[index]['capped_regret'],
            candidates[index]['mean_selected_cost'],
            -candidates[index]['accuracy'],
        ),
    )
    best_i = order[0]
    ranked = [candidates[index] for index in order]
    return candidates[best_i], ranked, prediction_by_candidate[best_i]


def paired_summary(
    values: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    delta = values - baseline
    indices = rng.integers(
        0,
        len(delta),
        size=(bootstrap, len(delta)),
    )
    boot = delta[indices].mean(axis=1)
    ordered = np.sort(delta)
    trimmed = (
        ordered[1:-1].mean()
        if len(ordered) > 2
        else ordered.mean()
    )
    return {
        'mean_cost': float(values.mean()),
        'success_rate_cost_lt_20': float(np.mean(values < 20.0)),
        'mean_delta': float(delta.mean()),
        'median_delta': float(np.median(delta)),
        'trimmed_delta': float(trimmed),
        'wins': int(np.sum(delta < 0)),
        'ties': int(np.sum(delta == 0)),
        'num_states': int(len(delta)),
        'bootstrap_ci95': np.quantile(
            boot,
            [0.025, 0.975],
        ).tolist(),
    }


def prefix_divergence(
    data: dict[str, np.ndarray],
    *,
    method: str,
) -> dict:
    method_i = method_index(data, method)
    means = data['proposal_mean_history'][:, method_i, -1]
    stds = data['proposal_std_history'][:, method_i, -1]
    difference = means[:, 0] - means[:, 1]
    scale = np.maximum(
        0.5 * (stds[:, 0] + stds[:, 1]),
        1e-4,
    )
    absolute = np.sqrt(
        np.mean(np.square(difference), axis=(0, 2))
    )
    normalized = np.sqrt(
        np.mean(np.square(difference / scale), axis=(0, 2))
    )
    first = float(np.mean(normalized[:2]))
    last = float(np.mean(normalized[-2:]))
    return {
        'absolute_rms_by_horizon_step': absolute.tolist(),
        'proposal_std_normalized_rms_by_horizon_step': (
            normalized.tolist()
        ),
        'first_two_over_last_two': first / max(last, 1e-8),
    }


def static_selection_cost(
    data: dict[str, np.ndarray],
    *,
    method: str,
    selector: str,
) -> np.ndarray:
    method_i = method_index(data, method)
    selector_i = selector_index(data, selector)
    return data['selected_true'][:, method_i, selector_i]


def main() -> None:
    args = parse_args()
    train = load_shards(args.train_dir)
    train_features = pair_differences(
        branch_feature_groups(train, method=args.method)
    )
    train_method_i = method_index(train, args.method)
    train_pair_cost = train['final_branch_true'][:, train_method_i]
    if not np.all(np.isfinite(train_pair_cost)):
        raise ValueError('training branch labels are incomplete')

    # Lock model selection before opening the fresh evaluation arrays.
    selected_config, ranked, train_oof_prediction = cross_validate(
        train_features,
        train_pair_cost,
        train['groups'],
    )
    train_difference = (
        train_pair_cost[:, 0] - train_pair_cost[:, 1]
    )
    final_target = target_values(
        train_difference,
        kind=selected_config['target'],
        reference=train_difference,
    )
    model = fit_ridge(
        train_features[selected_config['family']],
        final_target,
        alpha=float(selected_config['alpha']),
    )

    evaluation = load_shards(args.eval_dir)
    eval_features = pair_differences(
        branch_feature_groups(evaluation, method=args.method)
    )
    prediction = predict_ridge(
        model,
        eval_features[selected_config['family']],
    )
    eval_method_i = method_index(evaluation, args.method)
    eval_pair_cost = evaluation['final_branch_true'][:, eval_method_i]
    if not np.all(np.isfinite(eval_pair_cost)):
        raise ValueError('evaluation branch labels are incomplete')
    chosen, learned_cost = choose_cost(prediction, eval_pair_cost)

    baseline = static_selection_cost(
        evaluation,
        method=args.baseline_method,
        selector='primary',
    )
    rng = np.random.default_rng(args.seed)
    methods = {
        'learned_outcome_selector': learned_cost,
        'bp_primary': static_selection_cost(
            evaluation,
            method=args.method,
            selector='primary',
        ),
        'bp_k3': static_selection_cost(
            evaluation,
            method=args.method,
            selector='k3',
        ),
        'bp_k10': static_selection_cost(
            evaluation,
            method=args.method,
            selector='k10',
        ),
        'bp_consensus': static_selection_cost(
            evaluation,
            method=args.method,
            selector='consensus',
        ),
        'bp_oracle_union': np.min(eval_pair_cost, axis=1),
    }
    eval_report = {
        name: paired_summary(
            values,
            baseline,
            bootstrap=args.bootstrap,
            rng=rng,
        )
        for name, values in methods.items()
    }

    train_chosen, train_oof_cost = choose_cost(
        train_oof_prediction,
        train_pair_cost,
    )
    train_oracle = np.min(train_pair_cost, axis=1)
    train_primary = static_selection_cost(
        train,
        method=args.method,
        selector='primary',
    )
    report = {
        'version': 1,
        'method': args.method,
        'baseline_method': args.baseline_method,
        'train_rows': int(len(train['rows'])),
        'eval_rows': int(len(evaluation['rows'])),
        'row_overlap': int(
            len(np.intersect1d(train['rows'], evaluation['rows']))
        ),
        'selection_locked_from_train_only': True,
        'selected_config': selected_config,
        'cv_top10': ranked[:10],
        'train_oof': {
            'accuracy': float(
                np.mean(
                    train_chosen
                    == np.argmin(train_pair_cost, axis=1)
                )
            ),
            'mean_cost': float(train_oof_cost.mean()),
            'mean_primary_cost': float(train_primary.mean()),
            'mean_oracle_cost': float(train_oracle.mean()),
            'delta_vs_primary': float(
                np.mean(train_oof_cost - train_primary)
            ),
            'oracle_gap': float(
                np.mean(train_oof_cost - train_oracle)
            ),
        },
        'evaluation': {
            'baseline_mean_cost': float(baseline.mean()),
            'methods': eval_report,
        },
        'prefix_divergence': {
            'train': prefix_divergence(
                train,
                method=args.method,
            ),
            'evaluation': prefix_divergence(
                evaluation,
                method=args.method,
            ),
        },
    }
    if report['row_overlap']:
        raise ValueError('training and fresh evaluation rows overlap')

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    np.savez_compressed(
        args.out / 'predictions.npz',
        train_rows=train['rows'],
        train_oof_prediction=train_oof_prediction,
        train_pair_cost=train_pair_cost,
        eval_rows=evaluation['rows'],
        eval_prediction=prediction,
        eval_chosen_index=chosen,
        eval_pair_cost=eval_pair_cost,
        eval_baseline=baseline,
        eval_learned_cost=learned_cost,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
