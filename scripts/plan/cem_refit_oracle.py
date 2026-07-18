"""Execute cross-model elite refits from a saved CEM population audit.

``cem_round_oracle.py`` records every candidate and every model's score.  This
follow-up takes each scorer's top-k candidates, averages their action
sequences, and executes that smooth refit in the real simulator.  A self-model
refit at the final CEM round is the ordinary CEM returned mean, so it is a
matched implementation control; changing only the elite scorer isolates the
effect of proposal/evaluation role separation.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig

import stable_worldmodel as swm

from candidate_oracle import execute_candidate, make_process
from eval_wm import get_dataset

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    audit = cfg.get('audit', {})
    source = Path(str(audit.get('source', '')))
    if not source.exists():
        raise FileNotFoundError(f'audit.source does not exist: {source}')
    out = Path(
        str(
            audit.get(
                'out',
                source.with_name(f'{source.stem}_refit.npz'),
            )
        )
    )
    topk = int(audit.get('refit_topk', 30))
    if topk < 1:
        raise ValueError('audit.refit_topk must be positive')
    include_oracle = bool(audit.get('include_oracle', False))
    include_consensus = bool(audit.get('include_consensus', False))

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    for key in (
        'candidates',
        'pred',
        'initial_state',
        'goal_state',
        'generators',
        'scorers',
        'steps',
        'horizon',
        'goal_offset',
        'action_block',
    ):
        if key not in result:
            raise ValueError(f'{source}: missing {key!r}')
    if include_oracle and 'true' not in result:
        raise ValueError(
            f'{source}: audit.include_oracle requires true candidate costs'
        )

    requested = {
        'horizon': int(cfg.plan_config.horizon),
        'goal_offset': int(cfg.eval.goal_offset_steps),
        'action_block': int(cfg.plan_config.action_block),
    }
    for key, value in requested.items():
        actual = int(result[key])
        if actual != value:
            raise ValueError(
                f'{source}: {key}={actual}, requested config has {value}'
            )

    candidates = result['candidates'].astype(np.float32)
    predicted = result['pred'].astype(np.float64)
    scorer_labels = result['scorers'].astype(str)
    if candidates.ndim != 6:
        raise ValueError(
            'candidates must be (state,generator,round,N,H,D), '
            f'got {candidates.shape}'
        )
    expected_pred = (
        *candidates.shape[:3],
        len(result['scorers']),
        candidates.shape[3],
    )
    if predicted.shape != expected_pred:
        raise ValueError(
            f'pred has shape {predicted.shape}, expected {expected_pred}'
        )
    if include_consensus:
        order = np.argsort(predicted, axis=-1, kind='stable')
        ranks = np.argsort(order, axis=-1, kind='stable').astype(np.float64)
        if candidates.shape[3] > 1:
            ranks /= candidates.shape[3] - 1
        consensus = ranks.mean(axis=3, keepdims=True)
        predicted = np.concatenate([predicted, consensus], axis=3)
        scorer_labels = np.concatenate(
            [scorer_labels, np.asarray(['rank_consensus'])]
        )
    if include_oracle:
        predicted = np.concatenate(
            [
                predicted,
                result['true'].astype(np.float64)[..., None, :],
            ],
            axis=3,
        )
        scorer_labels = np.concatenate(
            [scorer_labels, np.asarray(['oracle_true_cost'])]
        )
    refit_count = min(topk, candidates.shape[3])

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = (
        int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) + 5
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = make_process(dataset, cfg.dataset.keys_to_cache)

    n_states, n_generators, n_rounds = candidates.shape[:3]
    n_scorers = predicted.shape[3]
    true = np.empty(
        (n_states, n_generators, n_rounds, n_scorers),
        dtype=np.float64,
    )
    pos_l2 = np.empty_like(true)
    angle = np.empty_like(true)
    success = np.empty_like(true, dtype=bool)
    terminal_state = np.empty(
        (*true.shape, result['initial_state'].shape[-1]),
        dtype=np.float64,
    )
    roundtrip_error = np.empty_like(true)
    started = time.time()

    try:
        for state_i in range(n_states):
            for generator_i in range(n_generators):
                for round_i in range(n_rounds):
                    population = candidates[
                        state_i,
                        generator_i,
                        round_i,
                    ]
                    for scorer_i in range(n_scorers):
                        elite = np.argsort(
                            predicted[
                                state_i,
                                generator_i,
                                round_i,
                                scorer_i,
                            ],
                            kind='stable',
                        )[:refit_count]
                        refit = (
                            population[elite]
                            .mean(
                                axis=0,
                                dtype=np.float64,
                            )
                            .astype(np.float32)
                        )
                        execution = execute_candidate(
                            world.envs.envs[0],
                            initial_state=result['initial_state'][state_i],
                            goal_state=result['goal_state'][state_i],
                            candidate=refit,
                            action_scaler=process['action'],
                            action_block=int(result['action_block']),
                            seed=int(cfg.seed) + state_i,
                        )
                        index = (
                            state_i,
                            generator_i,
                            round_i,
                            scorer_i,
                        )
                        true[index] = execution['cost']
                        pos_l2[index] = execution['pos_l2']
                        angle[index] = execution['angle']
                        success[index] = execution['success']
                        terminal_state[index] = execution['terminal_state']
                        roundtrip_error[index] = execution['roundtrip_error']
            print(
                f'[{state_i + 1}/{n_states}] '
                f'elapsed={(time.time() - started):.1f}s'
            )
    finally:
        world.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        version=np.asarray(1),
        source=np.asarray(str(source)),
        generators=result['generators'],
        scorers=scorer_labels,
        steps=result['steps'],
        rows=result['rows'],
        episodes=result['episodes'],
        starts=result['starts'],
        horizon=result['horizon'],
        goal_offset=result['goal_offset'],
        action_block=result['action_block'],
        refit_topk=np.asarray(refit_count),
        true=true,
        true_pos_l2=pos_l2,
        true_angle=angle,
        success=success,
        terminal_state=terminal_state,
        max_roundtrip_error=np.asarray(roundtrip_error.max()),
        elapsed_seconds=np.asarray(time.time() - started),
    )

    print('\nFinal-round cross-refits:')
    for generator_i, generator in enumerate(result['generators'].astype(str)):
        for scorer_i, scorer in enumerate(scorer_labels):
            print(
                f'{generator} -> {scorer}: '
                f'true={true[:, generator_i, -1, scorer_i].mean():.2f} '
                f'success={success[:, generator_i, -1, scorer_i].mean():.3f}'
            )
    print(f'results -> {out}')


if __name__ == '__main__':
    run()
