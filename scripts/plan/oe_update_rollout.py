"""Recursively intervene on CEM updates with simulator-oracle moments.

Starting from a distribution saved after one learned CEM round, every branch
uses the same Gaussian noise but interpolates its elite update between the
world-model scorer and simulator true cost.  Unlike the one-step resampling
gate, this test lets the counterfactual proposal path compound for several
rounds and executes the final returned mean.
"""

from __future__ import annotations

import os
from pathlib import Path
import time
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

from candidate_oracle import (
    execute_candidate,
    make_process,
    prepare_model_info,
    prepare_world_info,
    score_candidates,
)
from cem_round_oracle import execute_population
from eval_wm import get_dataset, img_transform
from oe_update_resample import (
    comma_floats,
    elite_moments,
    interpolate_moments,
    quantize_candidates,
    sha256,
    validate_source,
)

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    oe = cfg.get('oe', {})
    source = Path(str(oe.get('source', '')))
    if not source.exists():
        raise FileNotFoundError(f'oe.source does not exist: {source}')
    out = Path(
        str(oe.get('out', source.with_name(f'{source.stem}_oe_rollout.npz')))
    )
    legacy_policy = str(oe.get('generator', 'pd_d192_k3_eval'))
    source_generator_name = str(oe.get('source_generator', legacy_policy))
    scorer_name = str(oe.get('scorer', legacy_policy))
    start_step = int(oe.get('start_step', 4))
    num_rounds = int(oe.get('num_rounds', 5))
    num_samples = int(oe.get('num_samples', 100))
    num_states = int(oe.get('num_states', -1))
    topk = int(oe.get('topk', 30))
    std_floor = float(oe.get('std_floor', 1e-4))
    alphas = np.asarray(
        comma_floats(
            oe.get('alphas', '0,0.25,0.5,0.75,1'),
            name='oe.alphas',
        ),
        dtype=np.float64,
    )
    if num_rounds < 1:
        raise ValueError('oe.num_rounds must be positive')
    if num_samples < 2:
        raise ValueError('oe.num_samples must be at least two')
    if topk < 2:
        raise ValueError('oe.topk must be at least two')
    if std_floor <= 0:
        raise ValueError('oe.std_floor must be positive')
    if np.any((alphas < 0) | (alphas > 1)):
        raise ValueError('oe.alphas must be inside [0, 1]')
    if len(np.unique(alphas)) != len(alphas):
        raise ValueError('oe.alphas must be unique')

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    for key in ('mean', 'var'):
        if key not in result:
            raise ValueError(f'source is missing {key!r}')

    generators = result['generators'].astype(str).tolist()
    if source_generator_name not in generators:
        raise ValueError(
            f'oe.source_generator={source_generator_name!r} '
            f'not in {generators}'
        )
    generator_i = generators.index(source_generator_name)

    source_steps = result['steps'].astype(int).tolist()
    if start_step not in source_steps:
        raise ValueError(
            f'oe.start_step={start_step} is not present in {source_steps}'
        )
    source_round_i = source_steps.index(start_step)

    total_states = result['candidates'].shape[0]
    if num_states < 0:
        num_states = total_states
    if not 1 <= num_states <= total_states:
        raise ValueError(f'oe.num_states must be in [1, {total_states}] or -1')
    state_indices = np.arange(num_states, dtype=np.int64)

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = (
        int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) + 5
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = make_process(dataset, cfg.dataset.keys_to_cache)
    transform = {
        'pixels': img_transform(cfg),
        'pixels_hist': img_transform(cfg),
        'goal': img_transform(cfg),
    }
    config = swm.PlanConfig(**cfg.plan_config)
    model = swm.wm.utils.load_pretrained(scorer_name).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)

    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    population_shape = (
        len(state_indices),
        len(alphas),
        num_rounds,
        num_samples,
    )
    population_true = np.empty(population_shape, dtype=np.float64)
    population_success = np.empty(population_shape, dtype=bool)
    population_pred = np.empty(population_shape, dtype=np.float32)
    action_shape = result['candidates'].shape[-2:]
    mean_history = np.empty(
        (
            len(state_indices),
            len(alphas),
            num_rounds + 1,
            *action_shape,
        ),
        dtype=np.float32,
    )
    std_history = np.empty_like(mean_history)
    mean_true = np.empty(population_shape[:3], dtype=np.float64)
    mean_success = np.empty(population_shape[:3], dtype=bool)
    final_mean_true = np.empty(population_shape[:2], dtype=np.float64)
    final_mean_success = np.empty(population_shape[:2], dtype=bool)
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_candidate_quantization_error = 0.0
    roundtrip_errors = []
    started = time.time()

    try:
        for output_state_i, state_i in enumerate(state_indices):
            info, initial_state, goal_state = prepare_world_info(
                world,
                dataset,
                episode=int(result['episodes'][state_i]),
                start=int(result['starts'][state_i]),
                goal_offset=int(result['goal_offset']),
                callables=callables,
                history_len=int(cfg.plan_config.history_len),
                action_block=int(cfg.plan_config.action_block),
            )
            max_state_mismatch = max(
                max_state_mismatch,
                float(
                    np.max(
                        np.abs(
                            initial_state - result['initial_state'][state_i]
                        )
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(np.abs(goal_state - result['goal_state'][state_i]))
                ),
            )
            model_info = prepare_model_info(policy, info)
            cache: dict[bytes, dict] = {}
            initial_mean = result['mean'][
                state_i,
                generator_i,
                source_round_i,
            ].astype(np.float32)
            initial_std = np.maximum(
                result['var'][
                    state_i,
                    generator_i,
                    source_round_i,
                ].astype(np.float32),
                np.float32(std_floor),
            )

            for alpha_i, alpha in enumerate(alphas):
                mean = initial_mean.copy()
                std = initial_std.copy()
                mean_history[output_state_i, alpha_i, 0] = mean
                std_history[output_state_i, alpha_i, 0] = std

                for branch_round in range(num_rounds):
                    noise_rng = np.random.default_rng(
                        np.random.SeedSequence(
                            [
                                int(cfg.seed),
                                int(state_i),
                                int(start_step),
                                int(branch_round),
                            ]
                        )
                    )
                    noise = noise_rng.standard_normal(
                        (num_samples, *action_shape),
                        dtype=np.float32,
                    )
                    raw_candidates = noise * std[None] + mean[None]
                    raw_candidates[0] = mean
                    candidates = quantize_candidates(raw_candidates)
                    max_candidate_quantization_error = max(
                        max_candidate_quantization_error,
                        float(np.max(np.abs(candidates - raw_candidates))),
                    )
                    predicted = score_candidates(
                        model,
                        model_info,
                        candidates,
                    )
                    execution = execute_population(
                        world.envs.envs[0],
                        candidates=candidates,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        action_scaler=process['action'],
                        action_block=int(result['action_block']),
                        seed=int(cfg.seed) + int(state_i),
                        cache=cache,
                    )
                    learned_mean, learned_std, _ = elite_moments(
                        candidates,
                        predicted,
                        topk=topk,
                        std_floor=std_floor,
                    )
                    oracle_mean, oracle_std, _ = elite_moments(
                        candidates,
                        execution['true'],
                        topk=topk,
                        std_floor=std_floor,
                    )
                    mean, std = interpolate_moments(
                        learned_mean,
                        learned_std,
                        oracle_mean,
                        oracle_std,
                        float(alpha),
                    )

                    index = (output_state_i, alpha_i, branch_round)
                    population_true[index] = execution['true']
                    population_success[index] = execution['success']
                    population_pred[index] = predicted
                    mean_true[index] = execution['true'][0]
                    mean_success[index] = execution['success'][0]
                    mean_history[
                        output_state_i,
                        alpha_i,
                        branch_round + 1,
                    ] = mean
                    std_history[
                        output_state_i,
                        alpha_i,
                        branch_round + 1,
                    ] = std
                    roundtrip_errors.extend(
                        execution['roundtrip_error'].tolist()
                    )
                    print(
                        f'[{output_state_i + 1}/{len(state_indices)}] '
                        f'alpha={alpha:.2f} '
                        f'branch_round={branch_round + 1}/{num_rounds} '
                        f'min={execution["true"].min():.2f} '
                        f'coverage={int(execution["success"].any())} '
                        f'mean={execution["true"][0]:.2f}/'
                        f'{int(execution["success"][0])} '
                        f'elapsed={(time.time() - started) / 60:.1f}m'
                    )

                final_execution = execute_candidate(
                    world.envs.envs[0],
                    initial_state=initial_state,
                    goal_state=goal_state,
                    candidate=mean,
                    action_scaler=process['action'],
                    action_block=int(result['action_block']),
                    seed=int(cfg.seed) + int(state_i),
                )
                final_mean_true[output_state_i, alpha_i] = final_execution[
                    'cost'
                ]
                final_mean_success[
                    output_state_i,
                    alpha_i,
                ] = final_execution['success']
                roundtrip_errors.append(final_execution['roundtrip_error'])
    finally:
        world.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        version=np.asarray(2),
        source=np.asarray(str(source)),
        source_sha256=np.asarray(sha256(source)),
        generator=np.asarray(source_generator_name),
        scorer=np.asarray(scorer_name),
        state_indices=state_indices,
        rows=result['rows'][state_indices],
        episodes=result['episodes'][state_indices],
        starts=result['starts'][state_indices],
        start_step=np.asarray(start_step),
        num_rounds=np.asarray(num_rounds),
        alphas=alphas,
        num_samples=np.asarray(num_samples),
        topk=np.asarray(topk),
        std_floor=np.asarray(std_floor),
        population_true=population_true,
        population_success=population_success,
        population_pred=population_pred,
        mean_history=mean_history,
        std_history=std_history,
        mean_true=mean_true,
        mean_success=mean_success,
        final_mean_true=final_mean_true,
        final_mean_success=final_mean_success,
        max_state_mismatch=np.asarray(max_state_mismatch),
        max_goal_mismatch=np.asarray(max_goal_mismatch),
        candidate_storage_dtype=np.asarray('float16'),
        max_candidate_quantization_error=np.asarray(
            max_candidate_quantization_error
        ),
        max_roundtrip_error=np.asarray(
            max(roundtrip_errors, default=float('nan'))
        ),
        elapsed_seconds=np.asarray(time.time() - started),
    )

    print('\nRecursive update intervention')
    print(
        'alpha average_coverage last_coverage '
        'last_min_true final_mean_true final_mean_success'
    )
    for alpha_i, alpha in enumerate(alphas):
        coverage = np.any(
            population_success[:, alpha_i],
            axis=-1,
        )
        min_true = np.min(population_true[:, alpha_i], axis=-1)
        print(
            f'{alpha:.2f} '
            f'{coverage.mean():.3f} '
            f'{coverage[:, -1].mean():.3f} '
            f'{min_true[:, -1].mean():.2f} '
            f'{final_mean_true[:, alpha_i].mean():.2f} '
            f'{final_mean_success[:, alpha_i].mean():.3f}'
        )
    print(
        f'\nstate_mismatch={max_state_mismatch:.3e} '
        f'goal_mismatch={max_goal_mismatch:.3e} '
        f'quantization={max_candidate_quantization_error:.3e} '
        f'roundtrip={max(roundtrip_errors, default=float("nan")):.3e}'
    )
    print(f'results -> {out}')
    print(f'elapsed={(time.time() - started) / 60:.1f} minutes')


if __name__ == '__main__':
    run()
