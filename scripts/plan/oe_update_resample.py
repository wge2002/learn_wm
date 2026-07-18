"""Causal one-step gate for optimizer-equivalent CEM updates.

The round audit records a population together with learned and simulator
costs.  For a selected generator, this script computes two counterfactual
updates on exactly that population:

* the ordinary learned-cost elite mean/std;
* the simulator-oracle elite mean/std.

It then interpolates those sufficient statistics, samples the *next*
population with common random numbers, executes every candidate in PushT, and
scores the new population with the unchanged world model.  This distinguishes
three claims before any OE-WM training is attempted:

1. an oracle-equivalent update moves the next proposal into a better region;
2. the effect is monotonic as the update is moved toward the oracle;
3. the existing learned scorer can or cannot exploit the improved proposal.

Example smoke run:

  CUDA_VISIBLE_DEVICES=0 python scripts/plan/oe_update_resample.py \
      +plan_config.history_len=3 \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.goal_offset_steps=40 eval.video=false \
      eval.dataset_name=/path/to/pusht_eval_state_only.h5 \
      +oe.source=/path/to/cem_round_h5_off40_n12_full_v2.npz \
      +oe.out=/path/to/oe_resample_smoke.npz \
      +oe.generator=pd_d192_k3_eval +oe.steps=9,29 \
      +oe.alphas=0,1 +oe.num_samples=20 +oe.num_states=2
"""

from __future__ import annotations

import hashlib
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

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


def comma_ints(value, *, name: str) -> list[int]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'{name} must contain at least one integer')
    return [int(item) for item in items]


def comma_floats(value, *, name: str) -> list[float]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'{name} must contain at least one float')
    return [float(item) for item in items]


def elite_moments(
    candidates: np.ndarray,
    costs: np.ndarray,
    *,
    topk: int,
    std_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elite_count = min(int(topk), len(candidates))
    if elite_count < 2:
        raise ValueError('topk must select at least two candidates')
    indices = np.argsort(costs, kind='stable')[:elite_count]
    elite = candidates[indices].astype(np.float64)
    mean = elite.mean(axis=0).astype(np.float32)
    std = elite.std(axis=0, ddof=1).astype(np.float32)
    std = np.maximum(std, np.float32(std_floor))
    return mean, std, indices


def interpolate_moments(
    learned_mean: np.ndarray,
    learned_std: np.ndarray,
    oracle_mean: np.ndarray,
    oracle_std: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = float(alpha)
    mean = (1.0 - alpha) * learned_mean + alpha * oracle_mean
    log_std = (1.0 - alpha) * np.log(learned_std) + alpha * np.log(oracle_std)
    return mean.astype(np.float32), np.exp(log_std).astype(np.float32)


def quantize_candidates(candidates: np.ndarray) -> np.ndarray:
    """Match the serialized trace precision before scoring and execution.

    ``execute_population`` keys and executes float16-quantized candidates.
    Returning float32 here keeps model inference numerically convenient while
    ensuring that the model and simulator see the same action sequence.
    """
    return np.asarray(candidates, dtype=np.float16).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(
    result: dict[str, np.ndarray],
    *,
    cfg: DictConfig,
) -> None:
    required = {
        'version',
        'generators',
        'scorers',
        'steps',
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
        'horizon',
        'goal_offset',
        'action_block',
        'candidates',
        'pred',
        'true',
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f'source is missing keys: {missing}')
    requested = {
        'horizon': int(cfg.plan_config.horizon),
        'goal_offset': int(cfg.eval.goal_offset_steps),
        'action_block': int(cfg.plan_config.action_block),
    }
    for key, expected in requested.items():
        actual = int(result[key])
        if actual != expected:
            raise ValueError(
                f'source {key}={actual}, requested config has {expected}'
            )
    candidates = result['candidates']
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
    if result['pred'].shape != expected_pred:
        raise ValueError(
            f'pred has shape {result["pred"].shape}, expected {expected_pred}'
        )
    expected_true = (*candidates.shape[:4],)
    if result['true'].shape != expected_true:
        raise ValueError(
            f'true has shape {result["true"].shape}, expected {expected_true}'
        )


def print_summary(
    *,
    steps: np.ndarray,
    alphas: np.ndarray,
    sample_true: np.ndarray,
    sample_success: np.ndarray,
    model_refit_true: np.ndarray,
    model_refit_success: np.ndarray,
    oracle_refit_true: np.ndarray,
    oracle_refit_success: np.ndarray,
) -> None:
    print('\nOne-step counterfactual proposal results')
    print(
        'step alpha coverage min_true mean_success '
        'model_refit_true model_refit_success '
        'oracle_refit_true oracle_refit_success'
    )
    for round_i, step in enumerate(steps):
        for alpha_i, alpha in enumerate(alphas):
            true = sample_true[:, round_i, alpha_i]
            success = sample_success[:, round_i, alpha_i]
            print(
                f'{int(step):02d} {alpha:.2f} '
                f'{np.any(success, axis=1).mean():.3f} '
                f'{np.min(true, axis=1).mean():.2f} '
                f'{success[:, 0].mean():.3f} '
                f'{model_refit_true[:, round_i, alpha_i].mean():.2f} '
                f'{model_refit_success[:, round_i, alpha_i].mean():.3f} '
                f'{oracle_refit_true[:, round_i, alpha_i].mean():.2f} '
                f'{oracle_refit_success[:, round_i, alpha_i].mean():.3f}'
            )

    print('\nAggregate over selected rounds')
    for alpha_i, alpha in enumerate(alphas):
        true = sample_true[:, :, alpha_i]
        success = sample_success[:, :, alpha_i]
        state_coverage = np.any(success, axis=-1).mean(axis=1)
        state_min = np.min(true, axis=-1).mean(axis=1)
        state_model_success = model_refit_success[:, :, alpha_i].mean(axis=1)
        state_oracle_success = oracle_refit_success[:, :, alpha_i].mean(axis=1)
        print(
            f'alpha={alpha:.2f} '
            f'coverage={state_coverage.mean():.3f} '
            f'min_true={state_min.mean():.2f} '
            f'model_refit_success={state_model_success.mean():.3f} '
            f'oracle_refit_success={state_oracle_success.mean():.3f}'
        )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    oe = cfg.get('oe', {})
    source = Path(str(oe.get('source', '')))
    if not source.exists():
        raise FileNotFoundError(f'oe.source does not exist: {source}')
    out = Path(
        str(oe.get('out', source.with_name(f'{source.stem}_oe_resample.npz')))
    )
    legacy_policy = str(oe.get('generator', 'pd_d192_k3_eval'))
    source_generator_name = str(oe.get('source_generator', legacy_policy))
    scorer_name = str(oe.get('scorer', legacy_policy))
    force_rescore = bool(oe.get('rescore', False))
    requested_steps = comma_ints(
        oe.get('steps', '4,9,19,29'),
        name='oe.steps',
    )
    alphas = np.asarray(
        comma_floats(
            oe.get('alphas', '0,0.25,0.5,0.75,1'),
            name='oe.alphas',
        ),
        dtype=np.float64,
    )
    if np.any((alphas < 0) | (alphas > 1)):
        raise ValueError('oe.alphas must be inside [0, 1]')
    if len(np.unique(alphas)) != len(alphas):
        raise ValueError('oe.alphas must be unique')
    num_samples = int(oe.get('num_samples', 100))
    topk = int(oe.get('topk', 30))
    num_states = int(oe.get('num_states', -1))
    std_floor = float(oe.get('std_floor', 1e-4))
    if num_samples < 2:
        raise ValueError('oe.num_samples must be at least two')
    if topk < 2:
        raise ValueError('oe.topk must be at least two')
    if std_floor <= 0:
        raise ValueError('oe.std_floor must be positive')

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)

    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    if source_generator_name not in generators:
        raise ValueError(
            f'oe.source_generator={source_generator_name!r} '
            f'not in {generators}'
        )
    generator_i = generators.index(source_generator_name)
    stored_scorer_i = (
        scorers.index(scorer_name) if scorer_name in scorers else None
    )

    source_steps = result['steps'].astype(int).tolist()
    missing_steps = sorted(set(requested_steps) - set(source_steps))
    if missing_steps:
        raise ValueError(
            f'oe.steps {missing_steps} are not present in {source_steps}'
        )
    round_indices = np.asarray(
        [source_steps.index(step) for step in requested_steps],
        dtype=np.int64,
    )
    if len(set(requested_steps)) != len(requested_steps):
        raise ValueError('oe.steps must be unique')

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

    shape = (
        len(state_indices),
        len(round_indices),
        len(alphas),
        num_samples,
    )
    sample_true = np.empty(shape, dtype=np.float64)
    sample_success = np.empty(shape, dtype=bool)
    sample_pred = np.empty(shape, dtype=np.float32)
    sample_mean = np.empty((*shape[:3], *result['candidates'].shape[-2:]))
    sample_std = np.empty_like(sample_mean)
    learned_mean = np.empty((*shape[:2], *sample_mean.shape[-2:]))
    learned_std = np.empty_like(learned_mean)
    oracle_mean = np.empty_like(learned_mean)
    oracle_std = np.empty_like(learned_mean)
    model_refit_true = np.empty(shape[:3], dtype=np.float64)
    model_refit_success = np.empty(shape[:3], dtype=bool)
    oracle_refit_true = np.empty(shape[:3], dtype=np.float64)
    oracle_refit_success = np.empty(shape[:3], dtype=bool)
    mean_true = np.empty(shape[:3], dtype=np.float64)
    mean_success = np.empty(shape[:3], dtype=bool)
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

            for output_round_i, round_i in enumerate(round_indices):
                population = result['candidates'][
                    state_i,
                    generator_i,
                    round_i,
                ].astype(np.float32)
                if stored_scorer_i is not None and not force_rescore:
                    learned_cost = result['pred'][
                        state_i,
                        generator_i,
                        round_i,
                        stored_scorer_i,
                    ].astype(np.float64)
                else:
                    learned_cost = score_candidates(
                        model,
                        model_info,
                        population,
                    ).astype(np.float64)
                oracle_cost = result['true'][
                    state_i,
                    generator_i,
                    round_i,
                ].astype(np.float64)
                learned_mu, learned_sigma, _ = elite_moments(
                    population,
                    learned_cost,
                    topk=topk,
                    std_floor=std_floor,
                )
                oracle_mu, oracle_sigma, _ = elite_moments(
                    population,
                    oracle_cost,
                    topk=topk,
                    std_floor=std_floor,
                )
                learned_mean[output_state_i, output_round_i] = learned_mu
                learned_std[output_state_i, output_round_i] = learned_sigma
                oracle_mean[output_state_i, output_round_i] = oracle_mu
                oracle_std[output_state_i, output_round_i] = oracle_sigma

                noise_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(cfg.seed),
                            int(state_i),
                            int(requested_steps[output_round_i]),
                        ]
                    )
                )
                noise = noise_rng.standard_normal(
                    (
                        num_samples,
                        population.shape[-2],
                        population.shape[-1],
                    ),
                    dtype=np.float32,
                )

                for alpha_i, alpha in enumerate(alphas):
                    mean, std = interpolate_moments(
                        learned_mu,
                        learned_sigma,
                        oracle_mu,
                        oracle_sigma,
                        float(alpha),
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
                    refit_count = min(topk, num_samples)
                    model_elite = np.argsort(
                        predicted,
                        kind='stable',
                    )[:refit_count]
                    oracle_elite = np.argsort(
                        execution['true'],
                        kind='stable',
                    )[:refit_count]
                    model_refit = candidates[model_elite].mean(
                        axis=0,
                        dtype=np.float64,
                    )
                    oracle_refit = candidates[oracle_elite].mean(
                        axis=0,
                        dtype=np.float64,
                    )
                    model_execution = execute_candidate(
                        world.envs.envs[0],
                        initial_state=initial_state,
                        goal_state=goal_state,
                        candidate=model_refit.astype(np.float32),
                        action_scaler=process['action'],
                        action_block=int(result['action_block']),
                        seed=int(cfg.seed) + int(state_i),
                    )
                    oracle_execution = execute_candidate(
                        world.envs.envs[0],
                        initial_state=initial_state,
                        goal_state=goal_state,
                        candidate=oracle_refit.astype(np.float32),
                        action_scaler=process['action'],
                        action_block=int(result['action_block']),
                        seed=int(cfg.seed) + int(state_i),
                    )

                    index = (output_state_i, output_round_i, alpha_i)
                    sample_true[index] = execution['true']
                    sample_success[index] = execution['success']
                    sample_pred[index] = predicted
                    sample_mean[index] = mean
                    sample_std[index] = std
                    mean_true[index] = execution['true'][0]
                    mean_success[index] = execution['success'][0]
                    model_refit_true[index] = model_execution['cost']
                    model_refit_success[index] = model_execution['success']
                    oracle_refit_true[index] = oracle_execution['cost']
                    oracle_refit_success[index] = oracle_execution['success']
                    roundtrip_errors.extend(
                        execution['roundtrip_error'].tolist()
                    )
                    roundtrip_errors.extend(
                        [
                            model_execution['roundtrip_error'],
                            oracle_execution['roundtrip_error'],
                        ]
                    )
                    print(
                        f'[{output_state_i + 1}/{len(state_indices)}] '
                        f'step={requested_steps[output_round_i]:02d} '
                        f'alpha={alpha:.2f} '
                        f'min={execution["true"].min():.2f} '
                        f'coverage={int(execution["success"].any())} '
                        f'model_refit={model_execution["cost"]:.2f}/'
                        f'{int(model_execution["success"])} '
                        f'oracle_refit={oracle_execution["cost"]:.2f}/'
                        f'{int(oracle_execution["success"])} '
                        f'elapsed={(time.time() - started) / 60:.1f}m'
                    )
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
        rescored_source=np.asarray(force_rescore or stored_scorer_i is None),
        state_indices=state_indices,
        rows=result['rows'][state_indices],
        episodes=result['episodes'][state_indices],
        starts=result['starts'][state_indices],
        steps=np.asarray(requested_steps, dtype=np.int64),
        alphas=alphas,
        num_samples=np.asarray(num_samples),
        topk=np.asarray(topk),
        std_floor=np.asarray(std_floor),
        sample_true=sample_true,
        sample_success=sample_success,
        sample_pred=sample_pred,
        sample_mean=sample_mean,
        sample_std=sample_std,
        learned_mean=learned_mean,
        learned_std=learned_std,
        oracle_mean=oracle_mean,
        oracle_std=oracle_std,
        mean_true=mean_true,
        mean_success=mean_success,
        model_refit_true=model_refit_true,
        model_refit_success=model_refit_success,
        oracle_refit_true=oracle_refit_true,
        oracle_refit_success=oracle_refit_success,
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

    print_summary(
        steps=np.asarray(requested_steps),
        alphas=alphas,
        sample_true=sample_true,
        sample_success=sample_success,
        model_refit_true=model_refit_true,
        model_refit_success=model_refit_success,
        oracle_refit_true=oracle_refit_true,
        oracle_refit_success=oracle_refit_success,
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
