"""Trace CEM optimization and separate proposal from scoring failure.

For each paired PushT state, this audit runs several world models as CEM
population generators. At selected CEM iterations it:

1. stores the complete quantized action population and elite boundary;
2. scores that same population with every requested world model;
3. executes every candidate in a reset simulator to obtain true task cost;
4. reports proposal coverage, scorer regret, and rank-consensus regret.

The key distinction is:

* proposal failure: even the true oracle cannot find a good candidate in the
  generated population;
* scoring failure: a good candidate exists, but the learned cost selects a
  worse one;
* optimization overfit: predicted elite cost improves across CEM iterations
  while true coverage or selected true cost deteriorates.

Example:

  MODELS=pd_d192_k3_eval,iter2_multistep_eval,pd_d192_k10_eval
  CUDA_VISIBLE_DEVICES=3 python scripts/plan/cem_round_oracle.py \
      +plan_config.history_len=3 \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.goal_offset_steps=40 eval.video=false \
      eval.dataset_name=/path/to/pusht_expert_train.h5 \
      +audit.generators=\"${MODELS}\" +audit.scorers=\"${MODELS}\" \
      +audit.num_states=8 +audit.steps=\"0,1,2,4,9,19,29\" \
      +audit.out=outputs/week1/cem_round_h5_off40.npz
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm
from stable_worldmodel.solver.callbacks import CEMPopulationRecorder

from candidate_oracle import (
    execute_candidate,
    make_process,
    prepare_model_info,
    prepare_world_info,
    rank_metrics,
    sample_starts,
    score_candidates,
)
from eval_wm import get_dataset, img_transform

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


def comma_list(value, *, name: str) -> list[str]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'{name} must contain at least one item')
    return items


def optional_paths(value) -> list[Path]:
    if value is None:
        return []
    return [
        Path(item.strip())
        for item in str(value).split(',')
        if item.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_excluded_rows(paths: list[Path]) -> tuple[np.ndarray, dict]:
    rows = []
    sources = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f'audit.exclude_sources entry does not exist: {path}'
            )
        with np.load(path, allow_pickle=False) as archive:
            if 'rows' not in archive:
                raise ValueError(
                    f'audit.exclude_sources entry has no rows: {path}'
                )
            source_rows = np.asarray(
                archive['rows'],
                dtype=np.int64,
            ).reshape(-1)
        rows.append(source_rows)
        sources.append(
            {
                'path': str(path.resolve()),
                'sha256': sha256(path),
                'num_rows': int(len(source_rows)),
                'num_unique_rows': int(len(np.unique(source_rows))),
            }
        )
    excluded = (
        np.unique(np.concatenate(rows))
        if rows
        else np.empty(0, dtype=np.int64)
    )
    return excluded, {
        'version': 1,
        'sources': sources,
        'num_source_rows': int(sum(len(row) for row in rows)),
        'num_unique_excluded_rows': int(len(excluded)),
    }


def load_state_source(
    path: Path,
    *,
    start: int,
    count: int,
    horizon: int,
    goal_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load an exact row slice for paired multi-generator audits."""
    if not path.is_file():
        raise FileNotFoundError(f'audit.state_source does not exist: {path}')
    with np.load(path, allow_pickle=False) as archive:
        required = {'rows', 'episodes', 'starts'}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f'audit.state_source is missing fields {missing}: {path}'
            )
        source_rows = np.asarray(archive['rows'], dtype=np.int64).reshape(-1)
        source_episodes = np.asarray(
            archive['episodes'],
            dtype=np.int64,
        ).reshape(-1)
        source_starts = np.asarray(
            archive['starts'],
            dtype=np.int64,
        ).reshape(-1)
        if not (
            len(source_rows)
            == len(source_episodes)
            == len(source_starts)
        ):
            raise ValueError('audit.state_source row metadata length mismatch')
        for field, expected in (
            ('horizon', horizon),
            ('goal_offset', goal_offset),
        ):
            if field in archive.files:
                actual = int(np.asarray(archive[field]).item())
                if actual != expected:
                    raise ValueError(
                        f'audit.state_source {field}={actual}, '
                        f'current config requires {expected}'
                    )
    if start < 0 or count < 1 or start + count > len(source_rows):
        raise ValueError(
            f'audit.state_start/count slice [{start},{start + count}) '
            f'is outside source size {len(source_rows)}'
        )
    selected = slice(start, start + count)
    return (
        source_rows[selected],
        source_episodes[selected],
        source_starts[selected],
        {
            'path': str(path.resolve()),
            'sha256': sha256(path),
            'source_states': int(len(source_rows)),
            'slice_start': start,
            'slice_count': count,
        },
    )


def parse_steps(value, *, n_steps: int) -> list[int]:
    steps = sorted(
        {int(item) for item in comma_list(value, name='audit.steps')}
    )
    if steps[0] < 0 or steps[-1] >= n_steps:
        raise ValueError(
            f'audit.steps={steps} must be inside [0, {n_steps - 1}]'
        )
    return steps


def normalized_regret(predicted: np.ndarray, true: np.ndarray) -> float:
    selected = int(np.argmin(predicted))
    best = float(np.min(true))
    span = float(np.max(true) - best)
    if span <= 0:
        return 0.0
    return float((true[selected] - best) / span)


def value_normalized_regret(
    value: float,
    population_true: np.ndarray,
) -> float:
    best = float(np.min(population_true))
    span = float(np.max(population_true) - best)
    if span <= 0:
        return 0.0
    return float((value - best) / span)


def rank_fraction(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def consensus_cost(predicted: np.ndarray) -> np.ndarray:
    """Borda-style cost invariant to each model's latent distance scale."""
    return np.mean(
        np.stack([rank_fraction(row) for row in predicted]),
        axis=0,
    )


def selected_indices(
    recorded_cost: np.ndarray,
    *,
    max_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(recorded_cost)
    if max_candidates <= 0 or max_candidates >= n:
        return np.arange(n, dtype=np.int64)

    # Smoke/debug mode: retain generator elites, quantiles, and an unbiased
    # random component. Formal proposal-coverage runs should execute all N.
    order = np.argsort(recorded_cost, kind='stable')
    elite = max(1, max_candidates // 3)
    quantile = max(1, max_candidates // 3)
    proposed = [
        *order[:elite],
        *order[np.linspace(0, n - 1, quantile, dtype=np.int64)],
        *rng.permutation(n),
        *order,
    ]
    chosen = []
    seen = set()
    for raw_index in proposed:
        index = int(raw_index)
        if index in seen:
            continue
        chosen.append(index)
        seen.add(index)
        if len(chosen) == max_candidates:
            break
    return np.asarray(chosen, dtype=np.int64)


def execute_population(
    env,
    *,
    candidates: np.ndarray,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    action_scaler,
    action_block: int,
    seed: int,
    cache: dict[bytes, dict],
) -> dict[str, np.ndarray]:
    results = []
    for candidate in candidates:
        contiguous = np.ascontiguousarray(candidate, dtype=np.float16)
        key = contiguous.tobytes()
        if key not in cache:
            cache[key] = execute_candidate(
                env,
                initial_state=initial_state,
                goal_state=goal_state,
                candidate=contiguous.astype(np.float32),
                action_scaler=action_scaler,
                action_block=action_block,
                seed=seed,
            )
        results.append(cache[key])
    return {
        'true': np.asarray([item['cost'] for item in results]),
        'true_pos_l2': np.asarray([item['pos_l2'] for item in results]),
        'true_angle': np.asarray([item['angle'] for item in results]),
        'success': np.asarray(
            [item['success'] for item in results],
            dtype=bool,
        ),
        'terminal_state': np.stack(
            [item['terminal_state'] for item in results]
        ),
        'roundtrip_error': np.asarray(
            [item['roundtrip_error'] for item in results]
        ),
    }


def skipped_population(
    *,
    num_candidates: int,
    state_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Return shape-compatible placeholders for a model-only trace."""
    return {
        'true': np.full(num_candidates, np.nan, dtype=np.float64),
        'true_pos_l2': np.full(
            num_candidates,
            np.nan,
            dtype=np.float64,
        ),
        'true_angle': np.full(
            num_candidates,
            np.nan,
            dtype=np.float64,
        ),
        'success': np.zeros(num_candidates, dtype=bool),
        'terminal_state': np.full(
            (num_candidates, *state_shape),
            np.nan,
            dtype=np.float64,
        ),
        'roundtrip_error': np.full(
            num_candidates,
            np.nan,
            dtype=np.float64,
        ),
    }


def summarize(
    *,
    generators: list[str],
    scorers: list[str],
    steps: list[int],
    predicted: np.ndarray,
    true: np.ndarray,
    success: np.ndarray,
    returned_true: np.ndarray,
    returned_success: np.ndarray,
) -> None:
    # Shapes: state, generator, round, scorer, candidate / true without scorer.
    for generator_i, generator in enumerate(generators):
        print(f'\nGENERATOR {generator}')
        for round_i, step in enumerate(steps):
            true_slice = true[:, generator_i, round_i]
            success_slice = success[:, generator_i, round_i]
            oracle = np.mean(np.min(true_slice, axis=1))
            coverage = np.mean(np.any(success_slice, axis=1))
            returned = returned_true[:, generator_i, round_i]
            returned_regret = np.mean(
                [
                    value_normalized_regret(
                        returned[state_i],
                        true_slice[state_i],
                    )
                    for state_i in range(len(true_slice))
                ]
            )
            fields = [
                f'step={step:02d}',
                f'oracle_true={oracle:.2f}',
                f'success_coverage={coverage:.3f}',
                f'returned_true={np.mean(returned):.2f}',
                (
                    'returned_success='
                    f'{np.mean(returned_success[:, generator_i, round_i]):.3f}'
                ),
                f'returned_nreg={returned_regret:.3f}',
            ]
            for scorer_i, scorer in enumerate(scorers):
                scorer_regret = np.mean(
                    [
                        normalized_regret(
                            predicted[state_i, generator_i, round_i, scorer_i],
                            true_slice[state_i],
                        )
                        for state_i in range(len(true_slice))
                    ]
                )
                spearman = np.nanmean(
                    [
                        rank_metrics(
                            predicted[state_i, generator_i, round_i, scorer_i],
                            true_slice[state_i],
                        )['spearman']
                        for state_i in range(len(true_slice))
                    ]
                )
                fields.append(
                    f'{Path(scorer).name}:rho={spearman:.3f},'
                    f'nreg={scorer_regret:.3f}'
                )

            consensus_regret = np.mean(
                [
                    normalized_regret(
                        consensus_cost(
                            predicted[state_i, generator_i, round_i]
                        ),
                        true_slice[state_i],
                    )
                    for state_i in range(len(true_slice))
                ]
            )
            fields.append(f'consensus_nreg={consensus_regret:.3f}')
            print(' '.join(fields))


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    audit = cfg.get('audit', {})
    generators = comma_list(
        audit.get(
            'generators',
            'pd_d192_k3_eval,iter2_multistep_eval,pd_d192_k10_eval',
        ),
        name='audit.generators',
    )
    scorers = comma_list(
        audit.get('scorers', ','.join(generators)),
        name='audit.scorers',
    )
    all_policies = list(dict.fromkeys([*generators, *scorers]))
    num_states = int(audit.get('num_states', 8))
    max_candidates = int(audit.get('max_candidates', -1))
    evaluate_simulator = bool(
        audit.get('evaluate_simulator', True)
    )
    exclusion_paths = optional_paths(audit.get('exclude_sources'))
    excluded_rows, exclusion_audit = load_excluded_rows(exclusion_paths)
    output_path = Path(
        str(audit.get('out', 'outputs/week1/cem_round_oracle.npz'))
    )

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

    print(f'Loading {len(all_policies)} audit models: {all_policies}')
    models = {}
    for policy_name in all_policies:
        model = swm.wm.utils.load_pretrained(policy_name).to('cuda').eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models[policy_name] = model

    solvers = {}
    recorders = {}
    for generator_name in generators:
        solver = hydra.utils.instantiate(
            cfg.solver,
            model=models[generator_name],
        )
        steps = parse_steps(
            audit.get('steps', '0,1,2,4,9,19,29'),
            n_steps=solver.n_steps,
        )
        recorder = CEMPopulationRecorder(steps)
        solver.callbacks.append(recorder)
        solvers[generator_name] = solver
        recorders[generator_name] = recorder

    template_policy = swm.policy.WorldModelPolicy(
        solver=solvers[generators[0]],
        config=config,
        process=process,
        transform=transform,
    )
    world.set_policy(template_policy)
    for generator_name in generators[1:]:
        solvers[generator_name].configure(
            action_space=world.envs.action_space,
            n_envs=1,
            config=config,
        )

    rng = np.random.default_rng(cfg.seed)
    state_source_value = str(audit.get('state_source', '')).strip()
    if state_source_value:
        rows, episodes, starts, state_source_audit = load_state_source(
            Path(state_source_value),
            start=int(audit.get('state_start', 0)),
            count=num_states,
            horizon=int(cfg.plan_config.horizon),
            goal_offset=int(cfg.eval.goal_offset_steps),
        )
    else:
        rows, episodes, starts = sample_starts(
            dataset,
            num_states,
            int(cfg.eval.goal_offset_steps),
            rng,
            excluded_rows=excluded_rows,
        )
        state_source_audit = None
    overlap = np.intersect1d(rows, excluded_rows)
    if len(overlap):
        raise RuntimeError(
            f'sampled rows overlap exclusion set: {overlap.tolist()}'
        )
    exclusion_audit['sampled_rows'] = int(len(rows))
    exclusion_audit['sampled_exclusion_overlap'] = int(len(overlap))
    exclusion_audit['state_source'] = state_source_audit

    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    all_candidates = []
    all_candidate_indices = []
    all_predicted = []
    all_true = []
    all_pos_l2 = []
    all_angle = []
    all_success = []
    all_terminal = []
    all_elite_indices = []
    all_mean = []
    all_var = []
    all_prev_mean = []
    all_prev_var = []
    all_returned_pred = []
    all_returned_true = []
    all_returned_pos_l2 = []
    all_returned_angle = []
    all_returned_success = []
    all_returned_terminal = []
    initial_states = []
    goal_states = []
    roundtrip_errors = []
    started = time.time()

    try:
        for state_i, (episode, start) in enumerate(
            zip(episodes, starts, strict=True)
        ):
            info, initial_state, goal_state = prepare_world_info(
                world,
                dataset,
                episode=int(episode),
                start=int(start),
                goal_offset=int(cfg.eval.goal_offset_steps),
                callables=callables,
                history_len=int(cfg.plan_config.history_len),
                action_block=int(cfg.plan_config.action_block),
            )
            model_info = prepare_model_info(template_policy, info)
            initial_states.append(initial_state)
            goal_states.append(goal_state)
            state_cache: dict[bytes, dict] = {}
            state_candidates = []
            state_candidate_indices = []
            state_predicted = []
            state_true = []
            state_pos_l2 = []
            state_angle = []
            state_success = []
            state_terminal = []
            state_elite_indices = []
            state_mean = []
            state_var = []
            state_prev_mean = []
            state_prev_var = []
            state_returned_pred = []
            state_returned_true = []
            state_returned_pos_l2 = []
            state_returned_angle = []
            state_returned_success = []
            state_returned_terminal = []

            for generator_i, generator_name in enumerate(generators):
                solver = solvers[generator_name]
                # Common random numbers make the first population exactly
                # paired and isolate divergence caused by each learned cost.
                solver.torch_gen.manual_seed(int(cfg.seed) * 100_000 + state_i)
                solver(deepcopy(model_info), init_action=None)
                trace = recorders[generator_name].history[-1]
                if [record['step'] for record in trace] != steps:
                    raise RuntimeError(
                        f'{generator_name}: trace steps do not match {steps}'
                    )

                generator_candidates = []
                generator_candidate_indices = []
                generator_predicted = []
                generator_true = []
                generator_pos_l2 = []
                generator_angle = []
                generator_success = []
                generator_terminal = []
                generator_elite_indices = []
                generator_mean = []
                generator_var = []
                generator_prev_mean = []
                generator_prev_var = []
                generator_returned_pred = []
                generator_returned_true = []
                generator_returned_pos_l2 = []
                generator_returned_angle = []
                generator_returned_success = []
                generator_returned_terminal = []

                for round_i, record in enumerate(trace):
                    population = record['candidates'][0].float().numpy()
                    recorded_cost = record['costs'][0].numpy()
                    indices = selected_indices(
                        recorded_cost,
                        max_candidates=max_candidates,
                        rng=rng,
                    )
                    candidates = population[indices]
                    predicted = np.stack(
                        [
                            score_candidates(
                                models[scorer_name],
                                model_info,
                                candidates,
                            )
                            for scorer_name in scorers
                        ]
                    )
                    returned_candidate = (
                        record['mean'][0].float().numpy()[None]
                    )
                    returned_pred = np.asarray(
                        [
                            score_candidates(
                                models[scorer_name],
                                model_info,
                                returned_candidate,
                            )[0]
                            for scorer_name in scorers
                        ]
                    )
                    if evaluate_simulator:
                        execution = execute_population(
                            world.envs.envs[0],
                            candidates=candidates,
                            initial_state=initial_state,
                            goal_state=goal_state,
                            action_scaler=process['action'],
                            action_block=int(cfg.plan_config.action_block),
                            seed=int(cfg.seed) + state_i,
                            cache=state_cache,
                        )
                        returned_execution = execute_population(
                            world.envs.envs[0],
                            candidates=returned_candidate,
                            initial_state=initial_state,
                            goal_state=goal_state,
                            action_scaler=process['action'],
                            action_block=int(cfg.plan_config.action_block),
                            seed=int(cfg.seed) + state_i,
                            cache=state_cache,
                        )
                    else:
                        execution = skipped_population(
                            num_candidates=len(candidates),
                            state_shape=tuple(goal_state.shape),
                        )
                        returned_execution = skipped_population(
                            num_candidates=1,
                            state_shape=tuple(goal_state.shape),
                        )

                    generator_candidates.append(
                        candidates.astype(np.float16, copy=False)
                    )
                    generator_candidate_indices.append(indices)
                    generator_predicted.append(predicted)
                    generator_true.append(execution['true'])
                    generator_pos_l2.append(execution['true_pos_l2'])
                    generator_angle.append(execution['true_angle'])
                    generator_success.append(execution['success'])
                    generator_terminal.append(execution['terminal_state'])
                    generator_elite_indices.append(
                        record['topk_inds'][0].numpy()
                    )
                    generator_mean.append(record['mean'][0].numpy())
                    generator_var.append(record['var'][0].numpy())
                    generator_prev_mean.append(record['prev_mean'][0].numpy())
                    generator_prev_var.append(record['prev_var'][0].numpy())
                    generator_returned_pred.append(returned_pred)
                    generator_returned_true.append(
                        returned_execution['true'][0]
                    )
                    generator_returned_pos_l2.append(
                        returned_execution['true_pos_l2'][0]
                    )
                    generator_returned_angle.append(
                        returned_execution['true_angle'][0]
                    )
                    generator_returned_success.append(
                        returned_execution['success'][0]
                    )
                    generator_returned_terminal.append(
                        returned_execution['terminal_state'][0]
                    )
                    if evaluate_simulator:
                        roundtrip_errors.extend(
                            execution['roundtrip_error'].tolist()
                        )
                        roundtrip_errors.extend(
                            returned_execution['roundtrip_error'].tolist()
                        )
                    elapsed = time.time() - started
                    simulator_summary = (
                        f'oracle={execution["true"].min():.2f} '
                        f'successes={int(execution["success"].sum())} '
                        f'returned={returned_execution["true"][0]:.2f}/'
                        f'{int(returned_execution["success"][0])}'
                        if evaluate_simulator
                        else 'simulator=skipped'
                    )
                    print(
                        f'[{state_i + 1}/{num_states}] '
                        f'generator={generator_i + 1}/{len(generators)} '
                        f'step={record["step"]} candidates={len(candidates)} '
                        f'{simulator_summary} '
                        f'elapsed={elapsed / 60:.1f}m'
                    )

                state_candidates.append(np.stack(generator_candidates))
                state_candidate_indices.append(
                    np.stack(generator_candidate_indices)
                )
                state_predicted.append(np.stack(generator_predicted))
                state_true.append(np.stack(generator_true))
                state_pos_l2.append(np.stack(generator_pos_l2))
                state_angle.append(np.stack(generator_angle))
                state_success.append(np.stack(generator_success))
                state_terminal.append(np.stack(generator_terminal))
                state_elite_indices.append(np.stack(generator_elite_indices))
                state_mean.append(np.stack(generator_mean))
                state_var.append(np.stack(generator_var))
                state_prev_mean.append(np.stack(generator_prev_mean))
                state_prev_var.append(np.stack(generator_prev_var))
                state_returned_pred.append(np.stack(generator_returned_pred))
                state_returned_true.append(np.asarray(generator_returned_true))
                state_returned_pos_l2.append(
                    np.asarray(generator_returned_pos_l2)
                )
                state_returned_angle.append(
                    np.asarray(generator_returned_angle)
                )
                state_returned_success.append(
                    np.asarray(generator_returned_success)
                )
                state_returned_terminal.append(
                    np.stack(generator_returned_terminal)
                )

            all_candidates.append(np.stack(state_candidates))
            all_candidate_indices.append(np.stack(state_candidate_indices))
            all_predicted.append(np.stack(state_predicted))
            all_true.append(np.stack(state_true))
            all_pos_l2.append(np.stack(state_pos_l2))
            all_angle.append(np.stack(state_angle))
            all_success.append(np.stack(state_success))
            all_terminal.append(np.stack(state_terminal))
            all_elite_indices.append(np.stack(state_elite_indices))
            all_mean.append(np.stack(state_mean))
            all_var.append(np.stack(state_var))
            all_prev_mean.append(np.stack(state_prev_mean))
            all_prev_var.append(np.stack(state_prev_var))
            all_returned_pred.append(np.stack(state_returned_pred))
            all_returned_true.append(np.stack(state_returned_true))
            all_returned_pos_l2.append(np.stack(state_returned_pos_l2))
            all_returned_angle.append(np.stack(state_returned_angle))
            all_returned_success.append(np.stack(state_returned_success))
            all_returned_terminal.append(np.stack(state_returned_terminal))
    finally:
        world.close()

    candidates = np.stack(all_candidates)
    candidate_indices = np.stack(all_candidate_indices)
    predicted = np.stack(all_predicted)
    true = np.stack(all_true)
    true_pos_l2 = np.stack(all_pos_l2)
    true_angle = np.stack(all_angle)
    success = np.stack(all_success)
    terminal_state = np.stack(all_terminal)
    returned_pred = np.stack(all_returned_pred)
    returned_true = np.stack(all_returned_true)
    returned_pos_l2 = np.stack(all_returned_pos_l2)
    returned_angle = np.stack(all_returned_angle)
    returned_success = np.stack(all_returned_success)
    returned_terminal = np.stack(all_returned_terminal)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        version=np.asarray(2),
        generators=np.asarray(generators),
        scorers=np.asarray(scorers),
        steps=np.asarray(steps),
        rows=rows,
        episodes=episodes,
        starts=starts,
        initial_state=np.stack(initial_states),
        goal_state=np.stack(goal_states),
        horizon=np.asarray(int(cfg.plan_config.horizon)),
        goal_offset=np.asarray(int(cfg.eval.goal_offset_steps)),
        action_block=np.asarray(int(cfg.plan_config.action_block)),
        candidates=candidates,
        candidate_indices=candidate_indices,
        pred=predicted,
        true=true,
        true_pos_l2=true_pos_l2,
        true_angle=true_angle,
        success=success,
        terminal_state=terminal_state,
        topk_indices=np.stack(all_elite_indices),
        mean=np.stack(all_mean),
        var=np.stack(all_var),
        prev_mean=np.stack(all_prev_mean),
        prev_var=np.stack(all_prev_var),
        returned_pred=returned_pred,
        returned_true=returned_true,
        returned_pos_l2=returned_pos_l2,
        returned_angle=returned_angle,
        returned_success=returned_success,
        returned_terminal_state=returned_terminal,
        max_roundtrip_error=np.asarray(
            max(roundtrip_errors, default=float('nan'))
        ),
        elapsed_seconds=np.asarray(time.time() - started),
        exclusion_audit=np.asarray(
            json.dumps(exclusion_audit, sort_keys=True)
        ),
        simulator_evaluated=np.asarray(evaluate_simulator),
    )

    if evaluate_simulator:
        summarize(
            generators=generators,
            scorers=scorers,
            steps=steps,
            predicted=predicted,
            true=true,
            success=success,
            returned_true=returned_true,
            returned_success=returned_success,
        )
    print(f'\nresults -> {output_path}')
    print(
        f'shape candidates={candidates.shape} pred={predicted.shape} '
        f'true={true.shape}'
    )
    print(f'elapsed={(time.time() - started) / 60:.1f} minutes')


if __name__ == '__main__':
    run()
