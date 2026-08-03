"""Sparse true-query routing probe for discrete OE correction modes.

The learned router cannot identify the promising correction codebook modes
from current model features.  This probe measures how many high-fidelity
candidate labels would be needed to identify a mode on a new population.

Codebooks are fit only on outer-training states.  On held-out states, a query
strategy reveals true cost for ``m`` of the 300 stored candidates.  The best
10% of those queried candidates estimates an oracle update, which routes to
the closest learned correction mode (or no-op).  Evaluation always uses the
full hidden oracle update.

This is a query-efficiency ceiling, not a deployable no-oracle planner.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from oe_update_corrector_probe import (
    EPS,
    load_trace,
    paired_bootstrap,
    state_aggregate,
    update_metrics,
)
from oe_update_mode_codebook_probe import (
    oracle_route,
    spherical_kmeans,
)


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(len(values) - 1, 1)


def query_indices(
    predicted: np.ndarray,
    *,
    budget: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    # predicted is (scorers, candidates).
    num_candidates = predicted.shape[1]
    budget = min(budget, num_candidates)
    base_order = np.argsort(predicted[0], kind='stable')
    selected = []

    def add(items) -> None:
        seen = set(selected)
        for item in items:
            item = int(item)
            if item not in seen:
                selected.append(item)
                seen.add(item)
            if len(selected) == budget:
                break

    if strategy == 'random':
        add(rng.permutation(num_candidates))
    elif strategy == 'model_stratified':
        top_count = max(2, budget // 2)
        add(base_order[:top_count])
        add(rng.permutation(num_candidates))
    elif strategy == 'ensemble_disagreement':
        if predicted.shape[0] < 2:
            raise ValueError(
                'ensemble_disagreement requires multiple scorers'
            )
        third = max(1, budget // 3)
        add(base_order[:third])
        ranks = np.stack([rank_fraction(row) for row in predicted])
        disagreement = np.var(ranks, axis=0)
        add(np.argsort(-disagreement, kind='stable')[:third])
        add(rng.permutation(num_candidates))
    else:
        raise ValueError(f'unknown query strategy {strategy!r}')
    if len(selected) < budget:
        add(base_order)
    return np.asarray(selected[:budget], dtype=np.int64)


def partial_oracle_update(
    candidates: np.ndarray,
    true_cost: np.ndarray,
    query: np.ndarray,
    *,
    prev_mean: np.ndarray,
    proposal_std: np.ndarray,
    elite_fraction: float,
) -> np.ndarray:
    queried_cost = true_cost[query]
    elite_count = max(
        2,
        min(
            len(query),
            int(round(len(query) * elite_fraction)),
        ),
    )
    elite_query = query[
        np.argsort(queried_cost, kind='stable')[:elite_count]
    ]
    estimate = candidates[elite_query].astype(np.float64).mean(axis=0)
    return ((estimate - prev_mean) / proposal_std).reshape(-1)


def route_estimate(
    baseline: np.ndarray,
    estimate: np.ndarray,
    proposal_std: np.ndarray,
    prototypes: np.ndarray,
) -> np.ndarray:
    codebook = np.concatenate(
        [np.zeros((1, prototypes.shape[1])), prototypes],
        axis=0,
    )
    candidates = baseline[None] + codebook
    actual = candidates * proposal_std[None]
    target = estimate * proposal_std
    relative = (
        np.linalg.norm(actual - target[None], axis=1)
        / max(np.linalg.norm(target), EPS)
    )
    return candidates[int(np.argmin(relative))]


def analyze(
    source: Path,
    *,
    topk: int,
    clusters: int,
    budgets: list[int],
    strategies: list[str],
    repeats: int,
    bootstrap: int,
    seed: int,
) -> dict:
    trace = load_trace(source, topk=topk)
    with np.load(source, allow_pickle=False) as archive:
        populations = np.asarray(archive['candidates'])[:, 0].astype(
            np.float64
        )
        predicted = np.asarray(archive['pred'])[:, 0].astype(np.float64)
        true_cost = np.asarray(archive['true'])[:, 0].astype(np.float64)
        prev_mean = np.asarray(archive['prev_mean'])[:, 0].astype(
            np.float64
        )
        prev_var = np.asarray(archive['prev_var'])[:, 0].astype(np.float64)
    num_states, num_rounds = true_cost.shape[:2]
    residual = (
        trace.oracle_update_normalized
        - trace.model_update_normalized
    )
    states = np.arange(num_states)
    baseline_cosine, baseline_relative = update_metrics(
        trace.model_update_normalized,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    baseline_state_cosine = state_aggregate(
        baseline_cosine,
        trace.state_ids,
        states,
    )
    baseline_state_relative = state_aggregate(
        baseline_relative,
        trace.state_ids,
        states,
    )
    configurations = [
        (strategy, budget)
        for strategy in strategies
        for budget in budgets
    ]
    routed_repeats = {
        configuration: [
            np.empty_like(trace.model_update_normalized)
            for _ in range(repeats)
        ]
        for configuration in configurations
    }
    oracle_routed = np.empty_like(trace.model_update_normalized)

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
            seed=seed + 1000 * fold,
        )
        oracle_routed[val_indices], _ = oracle_route(
            trace.model_update_normalized[val_indices],
            trace.oracle_update_normalized[val_indices],
            trace.proposal_std[val_indices],
            prototypes,
        )
        for example_i in val_indices:
            state_i = int(trace.state_ids[example_i])
            round_i = int(trace.round_ids[example_i])
            # ``prev_var`` is the solver's legacy name for its sampling
            # standard deviation, not a mathematical variance.
            proposal_std = np.maximum(
                prev_var[state_i, round_i],
                1e-8,
            )
            for strategy, budget in configurations:
                for repeat in range(repeats):
                    rng = np.random.default_rng(
                        seed
                        + 10_000_000 * fold
                        + 100_000 * state_i
                        + 1000 * round_i
                        + 10 * budget
                        + repeat
                    )
                    query = query_indices(
                        predicted[state_i, round_i],
                        budget=budget,
                        strategy=strategy,
                        rng=rng,
                    )
                    estimate = partial_oracle_update(
                        populations[state_i, round_i],
                        true_cost[state_i, round_i],
                        query,
                        prev_mean=prev_mean[state_i, round_i],
                        proposal_std=proposal_std,
                        elite_fraction=topk / populations.shape[2],
                    )
                    routed_repeats[(strategy, budget)][repeat][
                        example_i
                    ] = route_estimate(
                        trace.model_update_normalized[example_i],
                        estimate,
                        trace.proposal_std[example_i],
                        prototypes,
                    )

    oracle_cosine, oracle_relative = update_metrics(
        oracle_routed,
        trace.oracle_update_normalized,
        trace.proposal_std,
    )
    oracle_state_cosine = state_aggregate(
        oracle_cosine,
        trace.state_ids,
        states,
    )
    oracle_state_relative = state_aggregate(
        oracle_relative,
        trace.state_ids,
        states,
    )
    rows = []
    for strategy, budget in configurations:
        repeat_state_cosine = []
        repeat_state_relative = []
        for routed in routed_repeats[(strategy, budget)]:
            cosine, relative = update_metrics(
                routed,
                trace.oracle_update_normalized,
                trace.proposal_std,
            )
            repeat_state_cosine.append(
                state_aggregate(cosine, trace.state_ids, states)
            )
            repeat_state_relative.append(
                state_aggregate(relative, trace.state_ids, states)
            )
        state_cosine = np.mean(repeat_state_cosine, axis=0)
        state_relative = np.mean(repeat_state_relative, axis=0)
        cosine_delta = state_cosine - baseline_state_cosine
        relative_delta = state_relative - baseline_state_relative
        rng = np.random.default_rng(seed + budget)
        rows.append(
            {
                'strategy': strategy,
                'budget': budget,
                'update_cosine': float(np.mean(state_cosine)),
                'cosine_delta': float(np.mean(cosine_delta)),
                'cosine_delta_ci': list(
                    paired_bootstrap(
                        cosine_delta,
                        samples=bootstrap,
                        rng=rng,
                    )
                ),
                'relative_update_error': float(
                    np.mean(state_relative)
                ),
                'relative_error_delta': float(
                    np.mean(relative_delta)
                ),
                'relative_error_delta_ci': list(
                    paired_bootstrap(
                        relative_delta,
                        samples=bootstrap,
                        rng=rng,
                    )
                ),
            }
        )
    return {
        'cell': trace.label,
        'source': str(source.resolve()),
        'clusters': clusters,
        'repeats': repeats,
        'baseline': {
            'update_cosine': float(np.mean(baseline_state_cosine)),
            'relative_update_error': float(
                np.mean(baseline_state_relative)
            ),
        },
        'oracle_router': {
            'update_cosine': float(np.mean(oracle_state_cosine)),
            'relative_update_error': float(
                np.mean(oracle_state_relative)
            ),
        },
        'rows': rows,
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--clusters', type=int, default=8)
    parser.add_argument('--budgets', default='10,20,30,60,100')
    parser.add_argument(
        '--strategies',
        default='random,model_stratified,ensemble_disagreement',
    )
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260728)
    args = parser.parse_args()
    budgets = [
        int(item) for item in args.budgets.split(',') if item.strip()
    ]
    strategies = [
        item.strip()
        for item in args.strategies.split(',')
        if item.strip()
    ]
    results = [
        analyze(
            source,
            topk=args.topk,
            clusters=args.clusters,
            budgets=budgets,
            strategies=strategies,
            repeats=args.repeats,
            bootstrap=args.bootstrap,
            seed=args.seed + source_i * 1000,
        )
        for source_i, source in enumerate(args.sources)
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': 1,
        'topk': args.topk,
        'clusters': args.clusters,
        'budgets': budgets,
        'strategies': strategies,
        'repeats': args.repeats,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'results': results,
    }
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    csv_rows = []
    for result in results:
        for row in result['rows']:
            csv_rows.append(
                {
                    'cell': result['cell'],
                    **{
                        key: value
                        for key, value in row.items()
                        if not key.endswith('_ci')
                    },
                    'cosine_delta_ci_low': row['cosine_delta_ci'][0],
                    'cosine_delta_ci_high': row['cosine_delta_ci'][1],
                    'relative_error_delta_ci_low': row[
                        'relative_error_delta_ci'
                    ][0],
                    'relative_error_delta_ci_high': row[
                        'relative_error_delta_ci'
                    ][1],
                }
            )
    with (args.out_dir / 'metrics.csv').open(
        'w',
        encoding='utf-8',
        newline='',
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        '# Sparse-query routing of discrete OE correction modes',
        '',
        'Codebooks use only outer-training states. Held-out true costs are '
        'revealed only for the stated query budget; full labels are used only '
        'for evaluation.',
        '',
        '| cell | strategy | queries / 300 | cosine | Δ cosine | rel. error '
        '| Δ rel. error |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for result in results:
        for row in result['rows']:
            lines.append(
                f'| {result["cell"]} | `{row["strategy"]}` '
                f'| {row["budget"]} '
                f'| {row["update_cosine"]:.3f} '
                f'| {row["cosine_delta"]:+.3f} '
                f'{format_ci(row["cosine_delta_ci"])} '
                f'| {row["relative_update_error"]:.3f} '
                f'| {row["relative_error_delta"]:+.3f} '
                f'{format_ci(row["relative_error_delta_ci"])} |'
            )
    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    for result in results:
        best = max(
            result['rows'],
            key=lambda row: row['cosine_delta'],
        )
        print(
            f'{result["cell"]}: best {best["strategy"]} '
            f'm={best["budget"]} cos Δ={best["cosine_delta"]:+.3f}, '
            f'rel Δ={best["relative_error_delta"]:+.3f}; '
            f'oracle cos={result["oracle_router"]["update_cosine"]:.3f}',
            flush=True,
        )
    print(f'results -> {args.out_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
