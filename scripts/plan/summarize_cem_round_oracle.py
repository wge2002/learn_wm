"""Analyze CEM-round proposal, scoring, and optimizer-overfit audits.

The input is a version-2 archive produced by ``cem_round_oracle.py``.  The
analysis keeps the state pairing intact and separates three questions:

1. Does the generated population still contain a good action?
2. Can a learned scorer identify it?
3. Does the CEM-returned mean improve or overfit as optimization continues?

It also evaluates pre-declared, deployable proposer/verifier interventions.
The primary intervention uses the K=3 model as proposer and the K=10 model as
an independent verifier, motivated by the earlier closed-loop and fixed-bank
audits.  Since sampled dataset rows are stored in sorted order, fixed-round
choices are tuned on alternating states and reported on the interleaved
held-out states rather than using a confounded early-row/late-row split.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


DEFAULT_PROPOSER = 'pd_d192_k3_eval'
DEFAULT_VERIFIER = 'pd_d192_k10_eval'


@dataclass(frozen=True)
class Strategy:
    """Per-state outcomes for one action-selection rule."""

    name: str
    family: str
    generator: str
    selector: str
    selection_space: str
    candidate_budget: int
    true: np.ndarray
    success: np.ndarray
    selected_step: np.ndarray
    tuned_on_dev: bool = False
    deployable: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('result', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--bootstrap', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20_260_718)
    parser.add_argument('--primary-generator', default=DEFAULT_PROPOSER)
    parser.add_argument('--primary-verifier', default=DEFAULT_VERIFIER)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def validate(result: dict[str, np.ndarray]) -> tuple[int, int, int, int, int]:
    required = {
        'version',
        'generators',
        'scorers',
        'steps',
        'rows',
        'candidate_indices',
        'pred',
        'true',
        'success',
        'topk_indices',
        'returned_pred',
        'returned_true',
        'returned_success',
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f'Missing result keys: {missing}')
    if int(result['version']) < 2:
        raise ValueError('returned-mean analysis requires result version >= 2')

    true = result['true']
    pred = result['pred']
    if true.ndim != 4:
        raise ValueError(f'true must be (S,G,R,N), got {true.shape}')
    n_states, n_generators, n_rounds, n_candidates = true.shape
    n_scorers = len(result['scorers'])
    expected = (n_states, n_generators, n_rounds, n_scorers, n_candidates)
    if pred.shape != expected:
        raise ValueError(f'pred has shape {pred.shape}, expected {expected}')
    if result['success'].shape != true.shape:
        raise ValueError('success must have the same shape as true')
    if result['candidate_indices'].shape != true.shape:
        raise ValueError('candidate_indices must have the same shape as true')
    if result['returned_true'].shape != true.shape[:3]:
        raise ValueError('returned_true must be (S,G,R)')
    if result['returned_success'].shape != true.shape[:3]:
        raise ValueError('returned_success must be (S,G,R)')
    if result['returned_pred'].shape != expected[:4]:
        raise ValueError('returned_pred must be (S,G,R,Q)')
    if len(result['generators']) != n_generators:
        raise ValueError('generator labels do not match result tensors')
    if len(result['steps']) != n_rounds:
        raise ValueError('steps do not match result tensors')
    if n_states < 4:
        raise ValueError('at least four paired states are required')
    if not np.all(np.diff(result['steps']) > 0):
        raise ValueError('recorded CEM steps must be strictly increasing')
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError('true and pred arrays must be finite')
    return n_states, n_generators, n_rounds, n_scorers, n_candidates


def rank_fraction(values: np.ndarray) -> np.ndarray:
    """Return stable fractional ranks along the final axis."""
    values = np.asarray(values)
    n_values = values.shape[-1]
    if n_values == 1:
        return np.zeros_like(values, dtype=np.float64)
    order = np.argsort(values, axis=-1, kind='stable')
    ranks = np.argsort(order, axis=-1, kind='stable')
    return ranks.astype(np.float64) / (n_values - 1)


def rank_consensus(
    scores: np.ndarray, scorer_indices: list[int]
) -> np.ndarray:
    """Borda cost for scores shaped ``(state, scorer, candidate)``."""
    if not scorer_indices:
        raise ValueError('rank consensus needs at least one scorer')
    return np.mean(rank_fraction(scores[:, scorer_indices]), axis=1)


def take_last_axis(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.take_along_axis(values, indices[..., None], axis=-1)[..., 0]


def normalized_regret(
    selected: np.ndarray, population: np.ndarray
) -> np.ndarray:
    best = np.min(population, axis=-1)
    span = np.max(population, axis=-1) - best
    return np.divide(
        selected - best,
        span,
        out=np.zeros_like(selected, dtype=np.float64),
        where=span > 0,
    )


def spearman_rows(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred_rank = rank_fraction(predicted)
    true_rank = rank_fraction(true)
    pred_rank -= np.mean(pred_rank, axis=-1, keepdims=True)
    true_rank -= np.mean(true_rank, axis=-1, keepdims=True)
    numerator = np.sum(pred_rank * true_rank, axis=-1)
    denominator = np.sqrt(
        np.sum(pred_rank**2, axis=-1) * np.sum(true_rank**2, axis=-1)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=np.float64),
        where=denominator > 0,
    )


def scorer_index(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError as error:
        raise ValueError(f'{name!r} is not among scorers {labels}') from error


def generator_index(labels: list[str], name: str) -> int:
    try:
        return labels.index(name)
    except ValueError as error:
        raise ValueError(
            f'{name!r} is not among generators {labels}'
        ) from error


def select_strategy(
    *,
    name: str,
    family: str,
    generator: str,
    selector: str,
    selection_space: str,
    candidate_budget: int,
    scores: np.ndarray,
    true: np.ndarray,
    success: np.ndarray,
    candidate_steps: np.ndarray,
    tuned_on_dev: bool = False,
    deployable: bool = True,
) -> Strategy:
    selected = np.argmin(scores, axis=-1)
    return Strategy(
        name=name,
        family=family,
        generator=generator,
        selector=selector,
        selection_space=selection_space,
        candidate_budget=candidate_budget,
        true=take_last_axis(true, selected),
        success=take_last_axis(success, selected),
        selected_step=np.asarray(candidate_steps)[selected],
        tuned_on_dev=tuned_on_dev,
        deployable=deployable,
    )


def fixed_strategy(
    *,
    name: str,
    family: str,
    generator: str,
    selection_space: str,
    true: np.ndarray,
    success: np.ndarray,
    steps: np.ndarray,
    round_index: int,
    tuned_on_dev: bool = False,
) -> Strategy:
    n_states = len(true)
    return Strategy(
        name=name,
        family=family,
        generator=generator,
        selector=f'fixed_step_{int(steps[round_index])}',
        selection_space=selection_space,
        candidate_budget=1,
        true=true[:, round_index],
        success=success[:, round_index],
        selected_step=np.full(n_states, steps[round_index], dtype=np.int64),
        tuned_on_dev=tuned_on_dev,
    )


def elite_mask(
    candidate_indices: np.ndarray,
    topk_indices: np.ndarray,
) -> np.ndarray:
    """Map original CEM elite indices onto a possibly subsetted population."""
    return np.any(
        candidate_indices[..., None] == topk_indices[..., None, :],
        axis=-1,
    )


def build_strategies(
    result: dict[str, np.ndarray],
    *,
    primary_generator: str,
    primary_verifier: str,
    dev_indices: np.ndarray,
) -> list[Strategy]:
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    steps = result['steps'].astype(np.int64)
    true = result['true'].astype(np.float64)
    success = result['success'].astype(bool)
    pred = result['pred'].astype(np.float64)
    returned_true = result['returned_true'].astype(np.float64)
    returned_success = result['returned_success'].astype(bool)
    returned_pred = result['returned_pred'].astype(np.float64)
    n_states, _, n_rounds, n_candidates = true.shape
    all_scorers = list(range(len(scorers)))
    strategies: list[Strategy] = []

    for generator_i, generator in enumerate(generators):
        try:
            self_i = scorer_index(scorers, generator)
        except ValueError:
            self_i = None
        nonself = [i for i in all_scorers if i != self_i]
        if not nonself:
            nonself = all_scorers

        generator_returned_true = returned_true[:, generator_i]
        generator_returned_success = returned_success[:, generator_i]
        final = fixed_strategy(
            name=f'{generator}:returned:final',
            family='baseline',
            generator=generator,
            selection_space='returned_means',
            true=generator_returned_true,
            success=generator_returned_success,
            steps=steps,
            round_index=n_rounds - 1,
        )
        strategies.append(final)

        for round_i, step in enumerate(steps):
            strategies.append(
                fixed_strategy(
                    name=f'{generator}:returned:fixed_step_{int(step)}',
                    family='fixed_round',
                    generator=generator,
                    selection_space='returned_means',
                    true=generator_returned_true,
                    success=generator_returned_success,
                    steps=steps,
                    round_index=round_i,
                )
            )

        dev_best_round = int(
            np.argmin(np.mean(generator_returned_true[dev_indices], axis=0))
        )
        strategies.append(
            fixed_strategy(
                name=f'{generator}:returned:dev_best_round',
                family='dev_tuned_fixed_round',
                generator=generator,
                selection_space='returned_means',
                true=generator_returned_true,
                success=generator_returned_success,
                steps=steps,
                round_index=dev_best_round,
                tuned_on_dev=True,
            )
        )

        returned_scores = returned_pred[:, generator_i].transpose(0, 2, 1)
        for scorer_i, scorer in enumerate(scorers):
            strategies.append(
                select_strategy(
                    name=f'{generator}:returned:score_by_{scorer}',
                    family='round_verification',
                    generator=generator,
                    selector=scorer,
                    selection_space='returned_means',
                    candidate_budget=n_rounds,
                    scores=returned_scores[:, scorer_i],
                    true=generator_returned_true,
                    success=generator_returned_success,
                    candidate_steps=steps,
                )
            )
        strategies.append(
            select_strategy(
                name=f'{generator}:returned:consensus',
                family='round_verification',
                generator=generator,
                selector='all_rank_consensus',
                selection_space='returned_means',
                candidate_budget=n_rounds * len(all_scorers),
                scores=rank_consensus(returned_scores, all_scorers),
                true=generator_returned_true,
                success=generator_returned_success,
                candidate_steps=steps,
            )
        )
        strategies.append(
            select_strategy(
                name=f'{generator}:returned:nonself_consensus',
                family='round_verification',
                generator=generator,
                selector='nonself_rank_consensus',
                selection_space='returned_means',
                candidate_budget=n_rounds * len(nonself),
                scores=rank_consensus(returned_scores, nonself),
                true=generator_returned_true,
                success=generator_returned_success,
                candidate_steps=steps,
            )
        )
        strategies.append(
            select_strategy(
                name=f'{generator}:returned:oracle_round',
                family='oracle_ceiling',
                generator=generator,
                selector='true_simulator_cost',
                selection_space='returned_means',
                candidate_budget=n_rounds,
                scores=generator_returned_true,
                true=generator_returned_true,
                success=generator_returned_success,
                candidate_steps=steps,
                deployable=False,
            )
        )

        flat_true = true[:, generator_i].reshape(n_states, -1)
        flat_success = success[:, generator_i].reshape(n_states, -1)
        flat_pred = (
            pred[:, generator_i]
            .transpose(0, 2, 1, 3)
            .reshape(
                n_states,
                len(scorers),
                -1,
            )
        )
        flat_steps = np.repeat(steps, n_candidates)
        flat_elite = elite_mask(
            result['candidate_indices'][:, generator_i],
            result['topk_indices'][:, generator_i],
        ).reshape(n_states, -1)
        if not np.all(flat_elite.any(axis=1)):
            raise ValueError(f'elite mapping failed for {generator}')

        for scorer_i, scorer in enumerate(scorers):
            strategies.append(
                select_strategy(
                    name=f'{generator}:population_all:score_by_{scorer}',
                    family='population_verification',
                    generator=generator,
                    selector=scorer,
                    selection_space='all_recorded_populations',
                    candidate_budget=n_rounds * n_candidates,
                    scores=flat_pred[:, scorer_i],
                    true=flat_true,
                    success=flat_success,
                    candidate_steps=flat_steps,
                )
            )
            elite_scores = np.where(
                flat_elite,
                flat_pred[:, scorer_i],
                np.inf,
            )
            strategies.append(
                select_strategy(
                    name=f'{generator}:population_elite:score_by_{scorer}',
                    family='elite_verification',
                    generator=generator,
                    selector=scorer,
                    selection_space='recorded_generator_elites',
                    candidate_budget=int(flat_elite.sum(axis=1).max()),
                    scores=elite_scores,
                    true=flat_true,
                    success=flat_success,
                    candidate_steps=flat_steps,
                )
            )

        for consensus_name, scorer_indices in (
            ('all_rank_consensus', all_scorers),
            ('nonself_rank_consensus', nonself),
        ):
            consensus = rank_consensus(flat_pred, scorer_indices)
            strategies.append(
                select_strategy(
                    name=f'{generator}:population_all:{consensus_name}',
                    family='population_verification',
                    generator=generator,
                    selector=consensus_name,
                    selection_space='all_recorded_populations',
                    candidate_budget=(
                        n_rounds * n_candidates * len(scorer_indices)
                    ),
                    scores=consensus,
                    true=flat_true,
                    success=flat_success,
                    candidate_steps=flat_steps,
                )
            )
            elite_consensus = np.where(flat_elite, consensus, np.inf)
            strategies.append(
                select_strategy(
                    name=f'{generator}:population_elite:{consensus_name}',
                    family='elite_verification',
                    generator=generator,
                    selector=consensus_name,
                    selection_space='recorded_generator_elites',
                    candidate_budget=(
                        int(flat_elite.sum(axis=1).max()) * len(scorer_indices)
                    ),
                    scores=elite_consensus,
                    true=flat_true,
                    success=flat_success,
                    candidate_steps=flat_steps,
                )
            )

    generator_index(generators, primary_generator)
    scorer_index(scorers, primary_verifier)
    portfolio_true = returned_true.reshape(n_states, -1)
    portfolio_success = returned_success.reshape(n_states, -1)
    portfolio_steps = np.tile(steps, len(generators))
    portfolio_pred = returned_pred.transpose(0, 3, 1, 2).reshape(
        n_states,
        len(scorers),
        -1,
    )
    for scorer_i, scorer in enumerate(scorers):
        strategies.append(
            select_strategy(
                name=f'portfolio:returned:score_by_{scorer}',
                family='multi_proposer_portfolio',
                generator='all',
                selector=scorer,
                selection_space='all_returned_means',
                candidate_budget=len(generators) * n_rounds,
                scores=portfolio_pred[:, scorer_i],
                true=portfolio_true,
                success=portfolio_success,
                candidate_steps=portfolio_steps,
            )
        )
    strategies.append(
        select_strategy(
            name='portfolio:returned:rank_consensus',
            family='multi_proposer_portfolio',
            generator='all',
            selector='all_rank_consensus',
            selection_space='all_returned_means',
            candidate_budget=len(generators) * n_rounds * len(scorers),
            scores=rank_consensus(portfolio_pred, all_scorers),
            true=portfolio_true,
            success=portfolio_success,
            candidate_steps=portfolio_steps,
        )
    )
    strategies.append(
        select_strategy(
            name='portfolio:returned:oracle',
            family='oracle_ceiling',
            generator='all',
            selector='true_simulator_cost',
            selection_space='all_returned_means',
            candidate_budget=len(generators) * n_rounds,
            scores=portfolio_true,
            true=portfolio_true,
            success=portfolio_success,
            candidate_steps=portfolio_steps,
            deployable=False,
        )
    )
    return strategies


def bootstrap_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    sampled = np.asarray(values, dtype=np.float64)[indices]
    means = np.mean(sampled, axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator='\n',
        )
        writer.writeheader()
        writer.writerows(rows)


def round_metrics(
    result: dict[str, np.ndarray],
    *,
    primary_verifier: str,
) -> list[dict]:
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    steps = result['steps'].astype(int)
    true = result['true'].astype(np.float64)
    success = result['success'].astype(bool)
    pred = result['pred'].astype(np.float64)
    returned_true = result['returned_true'].astype(np.float64)
    returned_success = result['returned_success'].astype(bool)
    returned_pred = result['returned_pred'].astype(np.float64)
    verifier_i = scorer_index(scorers, primary_verifier)
    rows = []

    for generator_i, generator in enumerate(generators):
        self_i = scorer_index(scorers, generator)
        for round_i, step in enumerate(steps):
            population_true = true[:, generator_i, round_i]
            population_success = success[:, generator_i, round_i]
            returned = returned_true[:, generator_i, round_i]
            self_scores = pred[:, generator_i, round_i, self_i]
            verifier_scores = pred[:, generator_i, round_i, verifier_i]
            self_selected = take_last_axis(
                population_true,
                np.argmin(self_scores, axis=-1),
            )
            verifier_selected = take_last_axis(
                population_true,
                np.argmin(verifier_scores, axis=-1),
            )
            rows.append(
                {
                    'generator': generator,
                    'step': int(step),
                    'n_states': len(population_true),
                    'proposal_oracle_true': float(
                        np.mean(np.min(population_true, axis=-1))
                    ),
                    'proposal_success_coverage': float(
                        np.mean(np.any(population_success, axis=-1))
                    ),
                    'returned_true': float(np.mean(returned)),
                    'returned_success': float(
                        np.mean(returned_success[:, generator_i, round_i])
                    ),
                    'returned_nreg': float(
                        np.mean(normalized_regret(returned, population_true))
                    ),
                    'returned_self_pred': float(
                        np.mean(returned_pred[:, generator_i, round_i, self_i])
                    ),
                    'returned_verifier_pred': float(
                        np.mean(
                            returned_pred[:, generator_i, round_i, verifier_i]
                        )
                    ),
                    'self_population_rho': float(
                        np.nanmean(spearman_rows(self_scores, population_true))
                    ),
                    'self_population_nreg': float(
                        np.mean(
                            normalized_regret(
                                self_selected,
                                population_true,
                            )
                        )
                    ),
                    'verifier_population_rho': float(
                        np.nanmean(
                            spearman_rows(verifier_scores, population_true)
                        )
                    ),
                    'verifier_population_nreg': float(
                        np.mean(
                            normalized_regret(
                                verifier_selected,
                                population_true,
                            )
                        )
                    ),
                }
            )
    return rows


def round_state_metrics(
    result: dict[str, np.ndarray],
    *,
    primary_verifier: str,
) -> list[dict]:
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    steps = result['steps'].astype(int)
    true = result['true'].astype(np.float64)
    success = result['success'].astype(bool)
    pred = result['pred'].astype(np.float64)
    returned_true = result['returned_true'].astype(np.float64)
    returned_success = result['returned_success'].astype(bool)
    returned_pred = result['returned_pred'].astype(np.float64)
    verifier_i = scorer_index(scorers, primary_verifier)
    rows = []

    for state_i in range(len(true)):
        for generator_i, generator in enumerate(generators):
            self_i = scorer_index(scorers, generator)
            for round_i, step in enumerate(steps):
                population_true = true[state_i, generator_i, round_i]
                population_success = success[state_i, generator_i, round_i]
                self_scores = pred[
                    state_i,
                    generator_i,
                    round_i,
                    self_i,
                ]
                verifier_scores = pred[
                    state_i,
                    generator_i,
                    round_i,
                    verifier_i,
                ]
                self_selected_i = int(np.argmin(self_scores))
                verifier_selected_i = int(np.argmin(verifier_scores))
                returned = returned_true[state_i, generator_i, round_i]
                rows.append(
                    {
                        'state_index': state_i,
                        'dataset_row': int(result['rows'][state_i]),
                        'episode': int(result['episodes'][state_i]),
                        'start': int(result['starts'][state_i]),
                        'generator': generator,
                        'step': int(step),
                        'proposal_oracle_true': float(np.min(population_true)),
                        'proposal_success_coverage': bool(
                            np.any(population_success)
                        ),
                        'returned_true': float(returned),
                        'returned_success': bool(
                            returned_success[
                                state_i,
                                generator_i,
                                round_i,
                            ]
                        ),
                        'returned_nreg': float(
                            normalized_regret(
                                np.asarray(returned),
                                population_true,
                            )
                        ),
                        'returned_self_pred': float(
                            returned_pred[
                                state_i,
                                generator_i,
                                round_i,
                                self_i,
                            ]
                        ),
                        'returned_verifier_pred': float(
                            returned_pred[
                                state_i,
                                generator_i,
                                round_i,
                                verifier_i,
                            ]
                        ),
                        'self_selected_true': float(
                            population_true[self_selected_i]
                        ),
                        'self_selected_success': bool(
                            population_success[self_selected_i]
                        ),
                        'self_population_rho': float(
                            spearman_rows(
                                self_scores[None],
                                population_true[None],
                            )[0]
                        ),
                        'self_population_nreg': float(
                            normalized_regret(
                                np.asarray(population_true[self_selected_i]),
                                population_true,
                            )
                        ),
                        'verifier_selected_true': float(
                            population_true[verifier_selected_i]
                        ),
                        'verifier_selected_success': bool(
                            population_success[verifier_selected_i]
                        ),
                        'verifier_population_rho': float(
                            spearman_rows(
                                verifier_scores[None],
                                population_true[None],
                            )[0]
                        ),
                        'verifier_population_nreg': float(
                            normalized_regret(
                                np.asarray(
                                    population_true[verifier_selected_i]
                                ),
                                population_true,
                            )
                        ),
                    }
                )
    return rows


def update_equivalence_rows(
    result: dict[str, np.ndarray],
    *,
    dev_indices: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    """Compare each learned elite update with a true-cost CEM update.

    Candidate ranking alone does not describe an adaptive CEM search.  The
    object that changes the next query distribution is the elite moment
    update.  These rows compare the learned and simulator-oracle top-k sets,
    and the corresponding mean update directions, on exactly the same
    population.
    """

    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    steps = result['steps'].astype(int)
    candidates = result['candidates'].astype(np.float64)
    predicted = result['pred'].astype(np.float64)
    true = result['true'].astype(np.float64)
    success = result['success'].astype(bool)
    previous_mean = result['prev_mean'].astype(np.float64)

    n_states, n_generators, n_rounds, n_candidates = true.shape
    elite_count = min(
        int(result['topk_indices'].shape[-1]),
        n_candidates,
    )
    if elite_count < 1:
        raise ValueError('elite-update audit needs at least one elite')
    dev = set(dev_indices.tolist())
    state_rows: list[dict] = []

    for state_i in range(n_states):
        split = 'dev' if state_i in dev else 'test'
        for generator_i, generator in enumerate(generators):
            for round_i, step in enumerate(steps):
                population = candidates[state_i, generator_i, round_i]
                oracle_indices = np.argsort(
                    true[state_i, generator_i, round_i],
                    kind='stable',
                )[:elite_count]
                oracle_elites = population[oracle_indices]
                oracle_mean = oracle_elites.mean(axis=0)
                origin = previous_mean[state_i, generator_i, round_i]
                oracle_update = (oracle_mean - origin).reshape(-1)
                oracle_norm = float(np.linalg.norm(oracle_update))

                for scorer_i, scorer in enumerate(scorers):
                    learned_indices = np.argsort(
                        predicted[
                            state_i,
                            generator_i,
                            round_i,
                            scorer_i,
                        ],
                        kind='stable',
                    )[:elite_count]
                    learned_elites = population[learned_indices]
                    learned_mean = learned_elites.mean(axis=0)
                    learned_update = (learned_mean - origin).reshape(-1)
                    learned_norm = float(np.linalg.norm(learned_update))
                    denominator = learned_norm * oracle_norm
                    cosine = (
                        float(
                            np.dot(learned_update, oracle_update) / denominator
                        )
                        if denominator > 1e-12
                        else float('nan')
                    )
                    relative_error = float(
                        np.linalg.norm(learned_update - oracle_update)
                        / max(oracle_norm, 1e-12)
                    )
                    overlap = (
                        len(
                            set(learned_indices.tolist())
                            & set(oracle_indices.tolist())
                        )
                        / elite_count
                    )
                    state_rows.append(
                        {
                            'state_index': state_i,
                            'split': split,
                            'dataset_row': int(result['rows'][state_i]),
                            'episode': int(result['episodes'][state_i]),
                            'start': int(result['starts'][state_i]),
                            'generator': generator,
                            'scorer': scorer,
                            'is_self_scorer': scorer == generator,
                            'step': int(step),
                            'elite_count': elite_count,
                            'elite_overlap': float(overlap),
                            'update_cosine': cosine,
                            'relative_update_error': relative_error,
                            'learned_update_norm': learned_norm,
                            'oracle_update_norm': oracle_norm,
                            'learned_elite_success_fraction': float(
                                success[
                                    state_i,
                                    generator_i,
                                    round_i,
                                    learned_indices,
                                ].mean()
                            ),
                            'oracle_elite_success_fraction': float(
                                success[
                                    state_i,
                                    generator_i,
                                    round_i,
                                    oracle_indices,
                                ].mean()
                            ),
                        }
                    )

    summary_rows: list[dict] = []
    for split in ('all', 'dev', 'test'):
        for generator in generators:
            for scorer in scorers:
                for step in steps:
                    selected = [
                        row
                        for row in state_rows
                        if row['generator'] == generator
                        and row['scorer'] == scorer
                        and row['step'] == int(step)
                        and (split == 'all' or row['split'] == split)
                    ]
                    if not selected:
                        continue
                    cosines = np.asarray(
                        [row['update_cosine'] for row in selected],
                        dtype=np.float64,
                    )
                    summary_rows.append(
                        {
                            'split': split,
                            'generator': generator,
                            'scorer': scorer,
                            'is_self_scorer': scorer == generator,
                            'step': int(step),
                            'n_states': len(selected),
                            'elite_count': elite_count,
                            'mean_elite_overlap': float(
                                np.mean(
                                    [row['elite_overlap'] for row in selected]
                                )
                            ),
                            'mean_update_cosine': float(np.nanmean(cosines)),
                            'mean_relative_update_error': float(
                                np.mean(
                                    [
                                        row['relative_update_error']
                                        for row in selected
                                    ]
                                )
                            ),
                            'mean_learned_elite_success_fraction': float(
                                np.mean(
                                    [
                                        row['learned_elite_success_fraction']
                                        for row in selected
                                    ]
                                )
                            ),
                            'mean_oracle_elite_success_fraction': float(
                                np.mean(
                                    [
                                        row['oracle_elite_success_fraction']
                                        for row in selected
                                    ]
                                )
                            ),
                        }
                    )
    return summary_rows, state_rows


def state_strategy_rows(
    strategies: list[Strategy],
    result: dict[str, np.ndarray],
    *,
    dev_indices: np.ndarray,
) -> list[dict]:
    dev = set(dev_indices.tolist())
    rows = []
    for strategy in strategies:
        for state_i in range(len(strategy.true)):
            rows.append(
                {
                    'strategy': strategy.name,
                    'family': strategy.family,
                    'generator': strategy.generator,
                    'selector': strategy.selector,
                    'selection_space': strategy.selection_space,
                    'deployable': strategy.deployable,
                    'tuned_on_dev': strategy.tuned_on_dev,
                    'state_index': state_i,
                    'split': 'dev' if state_i in dev else 'test',
                    'dataset_row': int(result['rows'][state_i]),
                    'episode': int(result['episodes'][state_i]),
                    'start': int(result['starts'][state_i]),
                    'true': float(strategy.true[state_i]),
                    'success': bool(strategy.success[state_i]),
                    'selected_step': int(strategy.selected_step[state_i]),
                }
            )
    return rows


def strategy_rows(
    strategies: list[Strategy],
    *,
    generators: list[str],
    primary_generator: str,
    dev_indices: np.ndarray,
    test_indices: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> list[dict]:
    lookup = {strategy.name: strategy for strategy in strategies}
    baselines = {
        generator: lookup[f'{generator}:returned:final']
        for generator in generators
    }
    primary_baseline = baselines[primary_generator]
    rows: list[dict] = []

    for split_name, split_indices in (
        ('all', np.arange(len(dev_indices) + len(test_indices))),
        ('dev', dev_indices),
        ('test', test_indices),
    ):
        boot_indices = rng.integers(
            0,
            len(split_indices),
            size=(bootstrap, len(split_indices)),
        )
        for strategy in strategies:
            baseline = baselines.get(strategy.generator, primary_baseline)
            values = strategy.true[split_indices]
            successes = strategy.success[split_indices].astype(np.float64)
            delta_true = values - baseline.true[split_indices]
            delta_success = successes - baseline.success[split_indices].astype(
                np.float64
            )
            true_low, true_high = bootstrap_ci(values, boot_indices)
            success_low, success_high = bootstrap_ci(
                successes,
                boot_indices,
            )
            delta_true_low, delta_true_high = bootstrap_ci(
                delta_true,
                boot_indices,
            )
            delta_success_low, delta_success_high = bootstrap_ci(
                delta_success,
                boot_indices,
            )
            rows.append(
                {
                    'strategy': strategy.name,
                    'family': strategy.family,
                    'generator': strategy.generator,
                    'selector': strategy.selector,
                    'selection_space': strategy.selection_space,
                    'candidate_budget': strategy.candidate_budget,
                    'deployable': strategy.deployable,
                    'tuned_on_dev': strategy.tuned_on_dev,
                    'split': split_name,
                    'n_states': len(split_indices),
                    'mean_true': float(np.mean(values)),
                    'true_ci_low': true_low,
                    'true_ci_high': true_high,
                    'mean_success': float(np.mean(successes)),
                    'success_ci_low': success_low,
                    'success_ci_high': success_high,
                    'mean_selected_step': float(
                        np.mean(strategy.selected_step[split_indices])
                    ),
                    'baseline': baseline.name,
                    'delta_true_vs_baseline': float(np.mean(delta_true)),
                    'delta_true_ci_low': delta_true_low,
                    'delta_true_ci_high': delta_true_high,
                    'delta_success_vs_baseline': float(np.mean(delta_success)),
                    'delta_success_ci_low': delta_success_low,
                    'delta_success_ci_high': delta_success_high,
                }
            )
    return rows


def key_strategy_names(
    primary_generator: str,
    primary_verifier: str,
) -> list[str]:
    return [
        f'{primary_generator}:returned:final',
        f'{primary_generator}:returned:dev_best_round',
        (f'{primary_generator}:returned:score_by_{primary_generator}'),
        (f'{primary_generator}:returned:score_by_{primary_verifier}'),
        f'{primary_generator}:returned:nonself_consensus',
        (f'{primary_generator}:population_elite:score_by_{primary_verifier}'),
        (f'{primary_generator}:population_all:score_by_{primary_verifier}'),
        f'{primary_generator}:returned:oracle_round',
        'portfolio:returned:rank_consensus',
        'portfolio:returned:oracle',
    ]


def write_markdown(
    path: Path,
    *,
    result_path: Path,
    result: dict[str, np.ndarray],
    strategy_metrics: list[dict],
    update_metrics: list[dict],
    primary_generator: str,
    primary_verifier: str,
    dev_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    lookup = {
        row['strategy']: row
        for row in strategy_metrics
        if row['split'] == 'test'
    }
    lines = [
        '# CEM round selection audit',
        '',
        f'- Source: `{result_path}`',
        (
            f'- Cell: H={int(result["horizon"])}, '
            f'offset={int(result["goal_offset"])}'
        ),
        (
            f'- Paired split: dev={dev_indices.tolist()}, '
            f'held-out={test_indices.tolist()}'
        ),
        (
            f'- Primary test: proposer `{primary_generator}`, '
            f'verifier `{primary_verifier}`'
        ),
        '',
        '## Held-out outcomes',
        '',
        '| strategy | true cost | success | Δ cost vs final | '
        'Δ success | mean CEM step |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name in key_strategy_names(primary_generator, primary_verifier):
        if name not in lookup:
            continue
        row = lookup[name]
        lines.append(
            f'| `{name}` | {row["mean_true"]:.3f} | '
            f'{row["mean_success"]:.3f} | '
            f'{row["delta_true_vs_baseline"]:+.3f} '
            f'[{row["delta_true_ci_low"]:+.3f}, '
            f'{row["delta_true_ci_high"]:+.3f}] | '
            f'{row["delta_success_vs_baseline"]:+.3f} | '
            f'{row["mean_selected_step"]:.1f} |'
        )
    lines.extend(
        [
            '',
            'Lower true cost and higher success are better. Oracle rows are '
            'ceilings, not deployable methods. The dev-best fixed round is '
            'chosen without inspecting held-out outcomes.',
            '',
            '## CEM update equivalence',
            '',
            "The table compares each generator's own learned top-k moment "
            'update with the top-k update induced by true simulator cost on '
            'the same population.',
            '',
            '| generator | step | elite overlap | update cosine | '
            'relative update error |',
            '|---|---:|---:|---:|---:|',
        ]
    )
    self_updates = [
        row
        for row in update_metrics
        if row['split'] == 'all' and row['is_self_scorer']
    ]
    for row in self_updates:
        lines.append(
            f'| `{row["generator"]}` | {row["step"]} | '
            f'{row["mean_elite_overlap"]:.3f} | '
            f'{row["mean_update_cosine"]:.3f} | '
            f'{row["mean_relative_update_error"]:.3f} |'
        )
    lines.extend(
        [
            '',
            'An overlap near zero or cosine near zero means that the learned '
            'cost and true cost would send the next CEM proposal in different '
            'directions. These are snapshot diagnostics, not deployable '
            'oracle interventions.',
            '',
        ]
    )
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError('--bootstrap must be positive')
    result_path = args.result.resolve()
    result = load_result(result_path)
    n_states, _, _, _, _ = validate(result)
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    generator_index(generators, args.primary_generator)
    scorer_index(scorers, args.primary_verifier)

    # ``sample_starts`` sorts selected dataset rows.  Alternating the ordered
    # states spreads both halves across episodes/time instead of confounding
    # the split with dataset position.  This split was fixed before formal
    # results were inspected.
    dev_indices = np.arange(0, n_states, 2)
    test_indices = np.arange(1, n_states, 2)
    strategies = build_strategies(
        result,
        primary_generator=args.primary_generator,
        primary_verifier=args.primary_verifier,
        dev_indices=dev_indices,
    )
    rng = np.random.default_rng(args.seed)
    strategy_metrics = strategy_rows(
        strategies,
        generators=generators,
        primary_generator=args.primary_generator,
        dev_indices=dev_indices,
        test_indices=test_indices,
        bootstrap=args.bootstrap,
        rng=rng,
    )
    rounds = round_metrics(
        result,
        primary_verifier=args.primary_verifier,
    )
    round_states = round_state_metrics(
        result,
        primary_verifier=args.primary_verifier,
    )
    update_metrics, update_states = update_equivalence_rows(
        result,
        dev_indices=dev_indices,
    )
    state_strategies = state_strategy_rows(
        strategies,
        result,
        dev_indices=dev_indices,
    )

    out_dir = (
        args.out_dir or result_path.parent / f'{result_path.stem}_summary'
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'round_metrics.csv', rounds)
    write_csv(out_dir / 'round_state_metrics.csv', round_states)
    write_csv(out_dir / 'strategy_metrics.csv', strategy_metrics)
    write_csv(out_dir / 'state_strategies.csv', state_strategies)
    write_csv(out_dir / 'update_equivalence.csv', update_metrics)
    write_csv(
        out_dir / 'update_equivalence_states.csv',
        update_states,
    )
    write_markdown(
        out_dir / 'report.md',
        result_path=result_path,
        result=result,
        strategy_metrics=strategy_metrics,
        update_metrics=update_metrics,
        primary_generator=args.primary_generator,
        primary_verifier=args.primary_verifier,
        dev_indices=dev_indices,
        test_indices=test_indices,
    )

    metadata = {
        'source': str(result_path),
        'version': int(result['version']),
        'horizon': int(result['horizon']),
        'goal_offset': int(result['goal_offset']),
        'generators': generators,
        'scorers': scorers,
        'steps': result['steps'].astype(int).tolist(),
        'n_states': n_states,
        'dev_indices': dev_indices.tolist(),
        'test_indices': test_indices.tolist(),
        'primary_generator': args.primary_generator,
        'primary_verifier': args.primary_verifier,
        'bootstrap': args.bootstrap,
        'bootstrap_seed': args.seed,
        'max_roundtrip_error': float(result['max_roundtrip_error']),
    }
    (out_dir / 'audit.json').write_text(
        json.dumps(metadata, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'analyzed {n_states} paired states -> {out_dir}')
    print((out_dir / 'report.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
