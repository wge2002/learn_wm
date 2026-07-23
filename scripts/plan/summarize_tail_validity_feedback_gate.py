"""Summarize the closed-loop tail-validity feedback-channel gate.

All inference is paired at the source-state level.  Candidate rows are never
treated as independent samples.  The locked protocol is documented in
``docs/knowledge/tail_validity_feedback_gate_protocol_20260723.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge


VALID_FIELDS = (
    'source_indices',
    'rows',
    'episodes',
    'starts',
    'initial_state',
    'next_state',
    'goal_state',
    'prefix_plan',
    'current_embedding',
    'predicted_prefix_embedding',
    'actual_prefix_embedding',
    'prefix_residual',
    'goal_embedding',
    'candidates',
    'terminal_embedding',
    'base_pred',
    'corrected_pred',
    'true',
    'success',
    'terminal_state',
    'mean',
    'mean_base_pred',
    'mean_corrected_pred',
    'mean_true',
    'mean_success',
    'mean_terminal_state',
)

REQUEST_FIELDS = (
    'requested_source_indices',
    'requested_rows',
    'prefix_terminated',
    'prefix_truncated',
    'prefix_steps',
    'prefix_roundtrip_error',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('shards', nargs='+', type=Path)
    parser.add_argument('--out-json', type=Path)
    parser.add_argument('--out-md', type=Path)
    parser.add_argument('--seed', type=int, default=20260723)
    parser.add_argument('--bootstrap-draws', type=int, default=20_000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def load_shards(paths: list[Path]) -> tuple[dict, list[dict]]:
    shards = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            item = {key: np.asarray(archive[key]) for key in archive.files}
        missing = sorted(
            {'audit', 'alphas', *VALID_FIELDS, *REQUEST_FIELDS} - set(item)
        )
        if missing:
            raise ValueError(f'{path} is missing fields {missing}')
        item['audit_dict'] = json.loads(str(scalar(item['audit'])))
        item['path'] = path
        shards.append(item)
    shards.sort(key=lambda item: int(item['audit_dict']['state_start']))

    reference = shards[0]
    reference_audit = reference['audit_dict']
    reference_alphas = reference['alphas']
    invariant_audit = (
        'version',
        'source_sha256',
        'dataset',
        'policy',
        'alphas',
        'prefix_blocks',
        'prefix_environment_steps',
        'history_len',
        'horizon',
        'goal_offset',
        'action_block',
        'cem_steps',
        'num_samples',
        'topk',
    )
    for item in shards[1:]:
        if not np.array_equal(item['alphas'], reference_alphas):
            raise ValueError(f'{item["path"]}: alpha grid differs')
        for key in invariant_audit:
            if item['audit_dict'][key] != reference_audit[key]:
                raise ValueError(
                    f'{item["path"]}: audit field {key!r} differs'
                )

    requested = {
        field: np.concatenate([item[field] for item in shards], axis=0)
        for field in REQUEST_FIELDS
    }
    data = {
        field: np.concatenate([item[field] for item in shards], axis=0)
        for field in VALID_FIELDS
    }
    order = np.argsort(data['source_indices'], kind='stable')
    data = {field: value[order] for field, value in data.items()}
    request_order = np.argsort(
        requested['requested_source_indices'], kind='stable'
    )
    requested = {
        field: value[request_order] for field, value in requested.items()
    }
    if len(np.unique(data['source_indices'])) != len(data['source_indices']):
        raise ValueError('valid source indices overlap across shards')
    if len(np.unique(requested['requested_source_indices'])) != len(
        requested['requested_source_indices']
    ):
        raise ValueError('requested source indices overlap across shards')
    if not np.all(np.diff(requested['requested_source_indices']) == 1):
        raise ValueError('requested source slices are not contiguous')
    if not np.array_equal(
        np.sort(data['source_indices']),
        np.sort(
            requested['requested_source_indices'][
                ~(requested['prefix_terminated'] | requested['prefix_truncated'])
            ]
        ),
    ):
        raise ValueError('valid rows do not match nonterminal prefix rows')

    data['alphas'] = reference_alphas.astype(np.float64)
    data['audit'] = reference_audit
    data['requested'] = requested
    inputs = [
        {
            'path': str(item['path'].resolve()),
            'sha256': sha256(item['path']),
            'state_start': int(item['audit_dict']['state_start']),
            'requested_states': int(item['audit_dict']['requested_states']),
            'valid_replans': int(item['audit_dict']['valid_replans']),
        }
        for item in shards
    ]
    return data, inputs


def topk_recall(scores: np.ndarray, true: np.ndarray, topk: int) -> float:
    predicted = set(
        np.argsort(scores, kind='stable')[:topk].astype(int).tolist()
    )
    oracle = set(np.argsort(true, kind='stable')[:topk].astype(int).tolist())
    return len(predicted & oracle) / topk


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def score_metrics(
    scores: np.ndarray,
    true: np.ndarray,
    success: np.ndarray,
    topk: int,
) -> dict[str, np.ndarray]:
    states, arms = scores.shape[:2]
    recall = np.empty((states, arms), dtype=np.float64)
    elite_true_mean = np.empty_like(recall)
    elite_success = np.empty_like(recall)
    selected_true = np.empty_like(recall)
    selected_rank = np.empty_like(recall)
    rho = np.empty_like(recall)
    for state_i in range(states):
        for arm_i in range(arms):
            order = np.argsort(scores[state_i, arm_i], kind='stable')
            elite = order[:topk]
            recall[state_i, arm_i] = topk_recall(
                scores[state_i, arm_i],
                true[state_i, arm_i],
                topk,
            )
            elite_true_mean[state_i, arm_i] = np.mean(
                true[state_i, arm_i, elite]
            )
            elite_success[state_i, arm_i] = np.mean(
                success[state_i, arm_i, elite]
            )
            selected_true[state_i, arm_i] = true[
                state_i, arm_i, order[0]
            ]
            selected_rank[state_i, arm_i] = rank_fraction(
                true[state_i, arm_i]
            )[order[0]]
            rho[state_i, arm_i] = float(
                spearmanr(
                    scores[state_i, arm_i],
                    true[state_i, arm_i],
                ).statistic
            )
    return {
        'topk_recall': recall,
        'elite_true_mean': elite_true_mean,
        'elite_success': elite_success,
        'selected_true': selected_true,
        'selected_true_rank': selected_rank,
        'spearman': rho,
    }


def path_metrics(data: dict, score: np.ndarray) -> dict[str, np.ndarray]:
    topk = int(data['audit']['topk'])
    metrics = score_metrics(score, data['true'], data['success'], topk)
    metrics.update(
        {
            'support': np.any(data['success'], axis=-1).astype(np.float64),
            'success_fraction': np.mean(data['success'], axis=-1),
            'oracle_min': np.min(data['true'], axis=-1),
            'mean_true': data['mean_true'].astype(np.float64),
            'mean_success': data['mean_success'].astype(np.float64),
        }
    )
    return metrics


def fixed_population_metrics(data: dict, baseline_i: int) -> dict[str, np.ndarray]:
    terminal = data['terminal_embedding'][:, baseline_i].astype(np.float64)
    goal = data['goal_embedding'].astype(np.float64)
    residual = data['prefix_residual'].astype(np.float64)
    alphas = data['alphas']
    scores = np.stack(
        [
            np.square(
                terminal + alpha * residual[:, None, :] - goal[:, None, :]
            ).sum(axis=-1)
            for alpha in alphas
        ],
        axis=1,
    )
    true = np.repeat(
        data['true'][:, baseline_i : baseline_i + 1],
        len(alphas),
        axis=1,
    )
    success = np.repeat(
        data['success'][:, baseline_i : baseline_i + 1],
        len(alphas),
        axis=1,
    )
    return score_metrics(
        scores,
        true,
        success,
        int(data['audit']['topk']),
    )


def bootstrap_mean(
    values: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'mean': None, 'ci95': [None, None], 'states': 0}
    sampled = values[
        rng.integers(0, len(values), size=(draws, len(values)))
    ].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return {
        'mean': float(values.mean()),
        'ci95': [float(low), float(high)],
        'states': int(len(values)),
    }


def paired_arm_summaries(
    metrics: dict[str, np.ndarray],
    alphas: np.ndarray,
    baseline_i: int,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    report = {}
    for metric, values in metrics.items():
        report[metric] = {}
        for arm_i, alpha in enumerate(alphas):
            report[metric][f'{alpha:g}'] = {
                'absolute': bootstrap_mean(values[:, arm_i], rng, draws),
                'minus_alpha0': bootstrap_mean(
                    values[:, arm_i] - values[:, baseline_i],
                    rng,
                    draws,
                ),
            }
    return report


def safe_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(
        right, axis=-1
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 1e-12,
    )


def residual_features(data: dict) -> tuple[np.ndarray, list[str]]:
    residual = data['prefix_residual'].astype(np.float64)
    current = data['current_embedding'].astype(np.float64)
    predicted = data['predicted_prefix_embedding'].astype(np.float64)
    actual = data['actual_prefix_embedding'].astype(np.float64)
    goal = data['goal_embedding'].astype(np.float64)
    predicted_step = predicted - current
    actual_goal = goal - actual
    predicted_goal = goal - predicted
    features = np.column_stack(
        [
            np.linalg.norm(residual, axis=-1),
            safe_cosine(residual, predicted_step),
            safe_cosine(residual, actual_goal),
            safe_cosine(residual, predicted_goal),
            np.linalg.norm(predicted_step, axis=-1),
        ]
    )
    names = (
        'residual_norm',
        'residual_cos_predicted_step',
        'residual_cos_actual_goal',
        'residual_cos_predicted_goal',
        'predicted_step_norm',
    )
    return features, list(names)


def oof_ridge(
    features: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> dict:
    prediction = np.full(len(target), np.nan, dtype=np.float64)
    constant = np.full(len(target), np.nan, dtype=np.float64)
    coefficients = []
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        valid = folds == fold
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale < 1e-8] = 1.0
        model = Ridge(alpha=1.0)
        model.fit((features[train] - mean) / scale, target[train])
        prediction[valid] = model.predict((features[valid] - mean) / scale)
        constant[valid] = float(np.mean(target[train]))
        coefficients.append(model.coef_.astype(float).tolist())
    residual_ss = float(np.sum(np.square(target - prediction)))
    total_ss = float(np.sum(np.square(target - np.mean(target))))
    return {
        'oof_r2': float(1.0 - residual_ss / total_ss) if total_ss else 0.0,
        'oof_mae': float(np.mean(np.abs(target - prediction))),
        'constant_oof_mae': float(np.mean(np.abs(target - constant))),
        'oof_spearman': float(spearmanr(prediction, target).statistic),
        'fold_coefficients': coefficients,
        'prediction': prediction,
    }


def bootstrap_spearman(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> dict:
    observed = float(spearmanr(left, right).statistic)
    samples = []
    for _ in range(draws):
        index = rng.integers(0, len(left), size=len(left))
        value = float(spearmanr(left[index], right[index]).statistic)
        if np.isfinite(value):
            samples.append(value)
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        'rho': observed,
        'ci95': [float(low), float(high)],
        'states': int(len(left)),
    }


def crossfit_select_alpha(
    metrics: dict[str, np.ndarray],
    alphas: np.ndarray,
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    eligible = np.flatnonzero((alphas >= 0.0) & (alphas <= 1.0))
    selected = np.full(len(folds), -1, dtype=np.int64)
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        valid = folds == fold
        means = np.mean(metrics['topk_recall'][train][:, eligible], axis=0)
        best_value = np.max(means)
        tied = eligible[np.flatnonzero(np.isclose(means, best_value))]
        best = tied[np.argmin(alphas[tied])]
        selected[valid] = best
    if np.any(selected < 0):
        raise RuntimeError('cross-fit alpha selection left unassigned states')
    row = np.arange(len(folds))
    selected_metrics = {
        name: values[row, selected] for name, values in metrics.items()
    }
    return selected_metrics, selected


def summarize_selected(
    selected: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    rng: np.random.Generator,
    draws: int,
) -> dict:
    return {
        metric: {
            'selected': bootstrap_mean(values, rng, draws),
            'baseline': bootstrap_mean(baseline[metric], rng, draws),
            'delta': bootstrap_mean(values - baseline[metric], rng, draws),
        }
        for metric, values in selected.items()
    }


def decision(
    *,
    crossfit: dict,
    fixed_oracle_gain: dict,
    predictability: dict,
    negative_recall_delta: dict | None,
) -> tuple[str, list[str]]:
    recall = crossfit['topk_recall']['delta']
    mean_true = crossfit['mean_true']['delta']
    mean_success = crossfit['mean_success']['delta']
    prediction_ok = (
        predictability['ridge']['oof_r2'] > 0
        or predictability['residual_norm_vs_misrank']['ci95'][0] > 0
        or predictability['residual_norm_vs_misrank']['ci95'][1] < 0
    )
    outcome_ok = (
        mean_true['ci95'][1] < 0 or mean_success['ci95'][0] > 0
    )
    negative_ok = (
        negative_recall_delta is None
        or negative_recall_delta['mean'] < recall['mean']
    )
    reasons = []
    if (
        recall['mean'] >= 0.05
        and recall['ci95'][0] > 0
        and outcome_ok
        and prediction_ok
        and negative_ok
    ):
        reasons.append('cross-fit recursive recall and returned action both pass')
        return 'OPEN', reasons
    if fixed_oracle_gain['mean'] < 0.05:
        reasons.append('per-state positive-alpha fixed-pop oracle gain is < .05')
        return 'CLOSE', reasons
    if (
        recall['ci95'][1] <= 0
        and mean_true['ci95'][0] >= 0
        and mean_success['ci95'][1] <= 0
    ):
        reasons.append('cross-fit recursive correction has no rank or outcome gain')
        return 'CLOSE', reasons
    reasons.append('rank signal and recursive action conversion do not jointly pass')
    return 'HOLD', reasons


def fmt(item: dict, digits: int = 3) -> str:
    if item['mean'] is None:
        return 'n/a'
    low, high = item['ci95']
    return f'{item["mean"]:.{digits}f} [{low:.{digits}f},{high:.{digits}f}]'


def markdown_report(report: dict) -> str:
    protocol = report['protocol']
    implementation = report['implementation']
    recursive = report['recursive_arms']
    fixed = report['fixed_population']
    fixed_oracle = report[
        'fixed_positive_alpha_per_state_oracle_recall_gain'
    ]
    crossfit = report['crossfit_recursive']
    lines = [
        '# Tail-validity feedback-channel gate（2026-07-23）',
        '',
        f'**判决：`{report["verdict"]}`。** '
        + '；'.join(report['verdict_reasons']),
        '',
        '## Protocol audit',
        '',
        f'- requested source states: `{protocol["requested_states"]}`; '
        f'valid next replans: `{protocol["valid_replans"]}`; '
        f'prefix terminal/truncated: `{protocol["prefix_terminal_or_truncated"]}`;',
        f'- source SHA-256: `{protocol["source_sha256"]}`;',
        f'- collector SHA-256: `{implementation["collector_sha256"]}`; '
        f'summarizer SHA-256: `{implementation["summarizer_sha256"]}`;',
        f'- alphas: `{protocol["alphas"]}`; state bootstrap: '
        f'`{protocol["bootstrap_draws"]}` draws.',
        '',
        '## Recursive next-replan arms',
        '',
        '| alpha | top30 recall | Δ recall | returned true cost | Δ true cost | returned success |',
        '|---:|---:|---:|---:|---:|---:|',
    ]
    for alpha in protocol['alphas']:
        key = f'{alpha:g}'
        lines.append(
            f'| `{key}` | '
            f'{fmt(recursive["topk_recall"][key]["absolute"])} | '
            f'{fmt(recursive["topk_recall"][key]["minus_alpha0"])} | '
            f'{fmt(recursive["mean_true"][key]["absolute"], 2)} | '
            f'{fmt(recursive["mean_true"][key]["minus_alpha0"], 2)} | '
            f'{fmt(recursive["mean_success"][key]["absolute"])} |'
        )
    lines.extend(
        [
            '',
            '## Fixed baseline population',
            '',
            '| alpha | top30 recall | Δ recall | elite true cost |',
            '|---:|---:|---:|---:|',
        ]
    )
    for alpha in protocol['alphas']:
        key = f'{alpha:g}'
        lines.append(
            f'| `{key}` | '
            f'{fmt(fixed["topk_recall"][key]["absolute"])} | '
            f'{fmt(fixed["topk_recall"][key]["minus_alpha0"])} | '
            f'{fmt(fixed["elite_true_mean"][key]["absolute"], 2)} |'
        )
    lines.extend(
        [
            '',
            'Per-state hindsight best of `{0,.5,1}`: top30-recall gain '
            f'{fmt(fixed_oracle)} (locked CLOSE threshold: `.050`).',
        ]
    )
    selected_counts = report['crossfit_alpha_counts']
    lines.extend(
        [
            '',
            '## State-held-out alpha selection',
            '',
            f'5-fold held-out state assignments: `{selected_counts}`.',
            '',
            '| metric | baseline | cross-fit selected | selected − baseline |',
            '|---|---:|---:|---:|',
        ]
    )
    for metric in (
        'topk_recall',
        'support',
        'oracle_min',
        'mean_true',
        'mean_success',
    ):
        digits = 2 if metric in ('oracle_min', 'mean_true') else 3
        lines.append(
            f'| {metric} | '
            f'{fmt(crossfit[metric]["baseline"], digits)} | '
            f'{fmt(crossfit[metric]["selected"], digits)} | '
            f'{fmt(crossfit[metric]["delta"], digits)} |'
        )
    prediction = report['predictability']
    rho = prediction['residual_norm_vs_misrank']
    ridge = prediction['ridge']
    lines.extend(
        [
            '',
            '## Prefix residual predicts next-replan misranking?',
            '',
            f'- residual norm vs `1-recall`: Spearman `{rho["rho"]:.3f}` '
            f'`[{rho["ci95"][0]:.3f},{rho["ci95"][1]:.3f}]`;',
            f'- fixed 5-fold OOF ridge: `R²={ridge["oof_r2"]:.3f}`; '
            f'`MAE={ridge["oof_mae"]:.3f}` vs constant '
            f'`{ridge["constant_oof_mae"]:.3f}`; '
            f'OOF Spearman `{ridge["oof_spearman"]:.3f}`.',
            '',
            '## Interpretation guardrail',
            '',
            'The correction is applied inside all 30 CEM rounds, so recursive '
            'differences are not a final-selector result. Alpha is nevertheless '
            'a one-parameter persistent-residual family. This `CLOSE` rejects '
            'the locked optimistic additive-residual channel; it is not a proof '
            'that every nonlinear or history-conditioned feedback model is '
            'impossible.',
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    data, inputs = load_shards(args.shards)
    rng = np.random.default_rng(args.seed)
    alphas = data['alphas']
    baseline_i = int(np.flatnonzero(alphas == 0.0)[0])
    topk = int(data['audit']['topk'])

    recursive_metrics = path_metrics(data, data['corrected_pred'])
    base_on_paths = path_metrics(data, data['base_pred'])
    fixed_metrics = fixed_population_metrics(data, baseline_i)
    recursive_report = paired_arm_summaries(
        recursive_metrics,
        alphas,
        baseline_i,
        rng,
        args.bootstrap_draws,
    )
    base_path_report = paired_arm_summaries(
        base_on_paths,
        alphas,
        baseline_i,
        rng,
        args.bootstrap_draws,
    )
    fixed_report = paired_arm_summaries(
        fixed_metrics,
        alphas,
        baseline_i,
        rng,
        args.bootstrap_draws,
    )

    folds = data['source_indices'] % 5
    selected_metrics, selected_indices = crossfit_select_alpha(
        recursive_metrics,
        alphas,
        folds,
    )
    baseline_metrics = {
        name: values[:, baseline_i]
        for name, values in recursive_metrics.items()
    }
    crossfit_report = summarize_selected(
        selected_metrics,
        baseline_metrics,
        rng,
        args.bootstrap_draws,
    )
    selected_counts = {
        f'{alpha:g}': int(np.sum(alphas[selected_indices] == alpha))
        for alpha in sorted(np.unique(alphas[selected_indices]).tolist())
    }

    features, feature_names = residual_features(data)
    misrank = 1.0 - recursive_metrics['topk_recall'][:, baseline_i]
    ridge = oof_ridge(features, misrank, folds)
    ridge.pop('prediction')
    predictability = {
        'target': '1 - alpha0 recursive true-top30 recall',
        'features': feature_names,
        'residual_norm_vs_misrank': bootstrap_spearman(
            features[:, 0],
            misrank,
            rng,
            args.bootstrap_draws,
        ),
        'ridge': ridge,
    }

    eligible = np.flatnonzero((alphas >= 0.0) & (alphas <= 1.0))
    fixed_gain = (
        np.max(fixed_metrics['topk_recall'][:, eligible], axis=1)
        - fixed_metrics['topk_recall'][:, baseline_i]
    )
    fixed_oracle_gain = bootstrap_mean(
        fixed_gain,
        rng,
        args.bootstrap_draws,
    )
    negative = np.flatnonzero(alphas < 0)
    negative_delta = (
        recursive_report['topk_recall'][f'{alphas[negative[0]]:g}'][
            'minus_alpha0'
        ]
        if len(negative)
        else None
    )
    verdict, reasons = decision(
        crossfit=crossfit_report,
        fixed_oracle_gain=fixed_oracle_gain,
        predictability=predictability,
        negative_recall_delta=negative_delta,
    )

    requested = data['requested']
    summarizer_path = Path(__file__).resolve()
    collector_path = summarizer_path.with_name(
        'tail_validity_feedback_gate.py'
    )
    if not collector_path.is_file():
        raise FileNotFoundError(
            f'collector next to summarizer is missing: {collector_path}'
        )
    report = {
        'version': 1,
        'verdict': verdict,
        'verdict_reasons': reasons,
        'protocol': {
            'requested_states': int(len(requested['requested_source_indices'])),
            'valid_replans': int(len(data['source_indices'])),
            'prefix_terminal_or_truncated': int(
                np.sum(
                    requested['prefix_terminated']
                    | requested['prefix_truncated']
                )
            ),
            'source_indices': data['source_indices'].astype(int).tolist(),
            'rows': data['rows'].astype(int).tolist(),
            'alphas': alphas.tolist(),
            'topk': topk,
            'fold': 'source_index modulo 5',
            'bootstrap_draws': args.bootstrap_draws,
            'source_sha256': data['audit']['source_sha256'],
            'dataset': data['audit']['dataset'],
            'policy': data['audit']['policy'],
            'prefix_environment_steps': data['audit'][
                'prefix_environment_steps'
            ],
            'cem_steps': data['audit']['cem_steps'],
            'num_samples': data['audit']['num_samples'],
        },
        'recursive_arms': recursive_report,
        'base_scorer_on_recursive_paths': base_path_report,
        'fixed_population': fixed_report,
        'fixed_positive_alpha_per_state_oracle_recall_gain': fixed_oracle_gain,
        'crossfit_alpha_counts': selected_counts,
        'crossfit_recursive': crossfit_report,
        'predictability': predictability,
        'implementation': {
            'collector': str(collector_path),
            'collector_sha256': sha256(collector_path),
            'summarizer': str(summarizer_path),
            'summarizer_sha256': sha256(summarizer_path),
        },
        'inputs': inputs,
    }

    rendered = markdown_report(report)
    print(rendered)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + '\n'
        )
        print(f'json -> {args.out_json}')
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(rendered)
        print(f'markdown -> {args.out_md}')


if __name__ == '__main__':
    main()
