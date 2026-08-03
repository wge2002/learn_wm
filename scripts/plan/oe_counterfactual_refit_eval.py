"""Execute counterfactual CEM refits from a saved paired query bank.

For every recorded candidate population, this audit forms four global means:

* the mean of the K3-scored elite;
* the mean of the K10-scored elite;
* the mean of the simulator-true elite; and
* the mean returned by the population's generating CEM run.

It also splits each K3/K10/true elite into connected components on the same
proposal-whitened action graph and executes every component mean.  The result
distinguishes scorer-tail error, single-mean mode averaging, and proposal
support failure without rerunning the CEM trajectory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import stable_worldmodel as swm

from candidate_oracle import (
    make_process,
    prepare_model_info,
    prepare_world_info,
    score_candidates,
)
from cem_round_oracle import execute_population
from eval_wm import get_dataset, img_transform


GLOBAL_SELECTORS = ('k3', 'k10', 'true', 'stored')
COMPONENT_SELECTORS = ('k3', 'k10', 'true')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def comma_list(value, *, name: str) -> list[str]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'{name} must contain at least one item')
    return items


def symmetric_knn_fast(
    points: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    count = len(points)
    if neighbors < 1 or neighbors >= count:
        raise ValueError('neighbors must be in [1, N-1]')
    squared_norm = np.sum(np.square(points), axis=1)
    squared = (
        squared_norm[:, None]
        + squared_norm[None]
        - 2.0 * (points @ points.T)
    )
    np.fill_diagonal(squared, np.inf)
    nearest = np.argpartition(
        squared,
        kth=neighbors - 1,
        axis=1,
    )[:, :neighbors]
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[np.arange(count)[:, None], nearest] = True
    adjacency |= adjacency.T
    return adjacency


def component_labels(
    adjacency: np.ndarray,
    active: np.ndarray,
) -> np.ndarray:
    count = len(active)
    parent = np.arange(count)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left = find(left)
        right = find(right)
        if left != right:
            parent[right] = left

    active_indices = np.flatnonzero(active)
    for left in active_indices:
        for right in np.flatnonzero(adjacency[left] & active):
            if right > left:
                union(int(left), int(right))
    labels = np.full(count, -1, dtype=np.int32)
    root_to_label: dict[int, int] = {}
    for index in active_indices:
        root = find(int(index))
        if root not in root_to_label:
            root_to_label[root] = len(root_to_label)
        labels[index] = root_to_label[root]
    return labels


def load_source(
    path: Path,
    *,
    start: int,
    count: int,
) -> dict[str, np.ndarray]:
    required = {
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
        'candidates',
        'pred',
        'true',
        'success',
        'mean',
        'prev_mean',
        'prev_var',
        'returned_true',
        'returned_success',
        'generators',
        'scorers',
        'steps',
        'horizon',
        'goal_offset',
        'action_block',
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'source is missing fields: {missing}')
        data = {key: np.asarray(archive[key]) for key in required}
    total = len(data['rows'])
    if start < 0 or count < 1 or start + count > total:
        raise ValueError(
            f'refit slice [{start},{start + count}) outside source size {total}'
        )
    state_fields = {
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
        'candidates',
        'pred',
        'true',
        'success',
        'mean',
        'prev_mean',
        'prev_var',
        'returned_true',
        'returned_success',
    }
    selected = slice(start, start + count)
    return {
        key: value[selected] if key in state_fields else value
        for key, value in data.items()
    }


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    refit = cfg.get('refit', {})
    source_path = Path(str(refit.get('source', '')))
    output_path = Path(str(refit.get('out', '')))
    if not source_path.is_file():
        raise FileNotFoundError(f'refit.source does not exist: {source_path}')
    if output_path == Path('.'):
        raise ValueError('refit.out is required')
    start = int(refit.get('state_start', 0))
    count = int(refit.get('num_states', 1))
    elite = int(refit.get('elite', 30))
    neighbors = int(refit.get('neighbors', 12))
    model_names = comma_list(
        refit.get('models', 'pd_d192_k3_eval,pd_d192_k10_eval'),
        name='refit.models',
    )
    if len(model_names) != 2:
        raise ValueError('counterfactual audit requires exactly K3,K10 models')

    data = load_source(source_path, start=start, count=count)
    generator_names = data['generators'].astype(str).tolist()
    scorer_names = data['scorers'].astype(str).tolist()
    if generator_names != model_names or scorer_names != model_names:
        raise ValueError(
            f'source models {generator_names}/{scorer_names} '
            f'differ from requested {model_names}'
        )
    num_states, num_generators, num_rounds, population = data[
        'true'
    ].shape
    if elite < 2 or elite >= population:
        raise ValueError('elite must be in [2, population-1]')
    action_shape = tuple(data['candidates'].shape[-2:])
    terminal_shape = tuple(data['goal_state'].shape[1:])

    if int(data['horizon']) != int(cfg.plan_config.horizon):
        raise ValueError('source/config horizon mismatch')
    if int(data['goal_offset']) != int(cfg.eval.goal_offset_steps):
        raise ValueError('source/config goal offset mismatch')
    if int(data['action_block']) != int(cfg.plan_config.action_block):
        raise ValueError('source/config action block mismatch')

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

    print(f'Loading refit scorers: {model_names}', flush=True)
    models = {}
    for model_name in model_names:
        model = swm.wm.utils.load_pretrained(model_name).to('cuda').eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models[model_name] = model
    solver = hydra.utils.instantiate(
        cfg.solver,
        model=models[model_names[0]],
    )
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

    global_count = len(GLOBAL_SELECTORS)
    component_selector_count = len(COMPONENT_SELECTORS)
    max_components = elite
    global_action = np.full(
        (
            num_states,
            num_generators,
            num_rounds,
            global_count,
            *action_shape,
        ),
        np.nan,
        dtype=np.float16,
    )
    global_pred = np.full(
        (
            num_states,
            num_generators,
            num_rounds,
            global_count,
            len(model_names),
        ),
        np.nan,
        dtype=np.float32,
    )
    global_true = np.full(
        (num_states, num_generators, num_rounds, global_count),
        np.nan,
        dtype=np.float64,
    )
    global_pos_l2 = np.full_like(global_true, np.nan)
    global_angle = np.full_like(global_true, np.nan)
    global_success = np.zeros_like(global_true, dtype=bool)
    global_terminal = np.full(
        (*global_true.shape, *terminal_shape),
        np.nan,
        dtype=np.float64,
    )

    component_count = np.zeros(
        (
            num_states,
            num_generators,
            num_rounds,
            component_selector_count,
        ),
        dtype=np.int16,
    )
    component_size = np.zeros(
        (*component_count.shape, max_components),
        dtype=np.int16,
    )
    component_action = np.full(
        (*component_count.shape, max_components, *action_shape),
        np.nan,
        dtype=np.float16,
    )
    component_pred = np.full(
        (
            *component_count.shape,
            len(model_names),
            max_components,
        ),
        np.nan,
        dtype=np.float32,
    )
    component_true = np.full(
        (*component_count.shape, max_components),
        np.nan,
        dtype=np.float64,
    )
    component_pos_l2 = np.full_like(component_true, np.nan)
    component_angle = np.full_like(component_true, np.nan)
    component_success = np.zeros_like(component_true, dtype=bool)
    component_terminal = np.full(
        (*component_true.shape, *terminal_shape),
        np.nan,
        dtype=np.float64,
    )

    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_stored_true_mismatch = 0.0
    max_stored_success_mismatch = 0
    total_executions = 0
    try:
        for state_i in range(num_states):
            info, initial_state, goal_state = prepare_world_info(
                world,
                dataset,
                episode=int(data['episodes'][state_i]),
                start=int(data['starts'][state_i]),
                goal_offset=int(data['goal_offset']),
                callables=callables,
                history_len=int(cfg.plan_config.history_len),
                action_block=int(cfg.plan_config.action_block),
            )
            max_state_mismatch = max(
                max_state_mismatch,
                float(
                    np.max(
                        np.abs(initial_state - data['initial_state'][state_i])
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(np.abs(goal_state - data['goal_state'][state_i]))
                ),
            )
            model_info = prepare_model_info(policy, info)
            state_cache: dict[bytes, dict] = {}
            for generator_i in range(num_generators):
                for round_i in range(num_rounds):
                    candidates = data['candidates'][
                        state_i, generator_i, round_i
                    ].astype(np.float32)
                    predicted = data['pred'][state_i, generator_i, round_i]
                    true_cost = data['true'][state_i, generator_i, round_i]
                    selector_costs = (
                        predicted[0],
                        predicted[1],
                        true_cost,
                    )
                    global_means = [
                        candidates[
                            np.argsort(cost, kind='stable')[:elite]
                        ].mean(axis=0)
                        for cost in selector_costs
                    ]
                    global_means.append(
                        data['mean'][state_i, generator_i, round_i].astype(
                            np.float32
                        )
                    )
                    global_means_array = np.asarray(global_means)
                    global_execution = execute_population(
                        world.envs.envs[0],
                        candidates=global_means_array,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        action_scaler=process['action'],
                        action_block=int(cfg.plan_config.action_block),
                        seed=int(cfg.seed) + state_i,
                        cache=state_cache,
                    )
                    global_action[
                        state_i, generator_i, round_i
                    ] = global_means_array.astype(np.float16)
                    for scorer_i, model_name in enumerate(model_names):
                        global_pred[
                            state_i, generator_i, round_i, :, scorer_i
                        ] = score_candidates(
                            models[model_name],
                            model_info,
                            global_means_array,
                        )
                    global_true[state_i, generator_i, round_i] = (
                        global_execution['true']
                    )
                    global_pos_l2[state_i, generator_i, round_i] = (
                        global_execution['true_pos_l2']
                    )
                    global_angle[state_i, generator_i, round_i] = (
                        global_execution['true_angle']
                    )
                    global_success[state_i, generator_i, round_i] = (
                        global_execution['success']
                    )
                    global_terminal[state_i, generator_i, round_i] = (
                        global_execution['terminal_state']
                    )
                    total_executions += len(global_means_array)

                    stored_i = GLOBAL_SELECTORS.index('stored')
                    max_stored_true_mismatch = max(
                        max_stored_true_mismatch,
                        float(
                            abs(
                                global_execution['true'][stored_i]
                                - data['returned_true'][
                                    state_i, generator_i, round_i
                                ]
                            )
                        ),
                    )
                    max_stored_success_mismatch = max(
                        max_stored_success_mismatch,
                        int(
                            global_execution['success'][stored_i]
                            != data['returned_success'][
                                state_i, generator_i, round_i
                            ]
                        ),
                    )

                    normalized = (
                        (
                            candidates
                            - data['prev_mean'][
                                state_i, generator_i, round_i
                            ][None]
                        )
                        / np.maximum(
                            data['prev_var'][
                                state_i, generator_i, round_i
                            ][None],
                            1e-4,
                        )
                    ).reshape(population, -1)
                    normalized /= np.maximum(
                        normalized.std(axis=0),
                        1e-6,
                    )
                    adjacency = symmetric_knn_fast(
                        normalized,
                        neighbors=neighbors,
                    )
                    for selector_i, cost in enumerate(selector_costs):
                        active = np.zeros(population, dtype=bool)
                        active[
                            np.argsort(cost, kind='stable')[:elite]
                        ] = True
                        labels = component_labels(adjacency, active)
                        label_values = np.unique(labels[labels >= 0])
                        means = np.asarray(
                            [
                                candidates[labels == label].mean(axis=0)
                                for label in label_values
                            ]
                        )
                        sizes = np.asarray(
                            [
                                np.sum(labels == label)
                                for label in label_values
                            ],
                            dtype=np.int16,
                        )
                        width = len(means)
                        component_count[
                            state_i, generator_i, round_i, selector_i
                        ] = width
                        component_size[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = sizes
                        component_action[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = means.astype(np.float16)
                        execution = execute_population(
                            world.envs.envs[0],
                            candidates=means,
                            initial_state=initial_state,
                            goal_state=goal_state,
                            action_scaler=process['action'],
                            action_block=int(cfg.plan_config.action_block),
                            seed=int(cfg.seed) + state_i,
                            cache=state_cache,
                        )
                        for scorer_i, model_name in enumerate(model_names):
                            component_pred[
                                state_i,
                                generator_i,
                                round_i,
                                selector_i,
                                scorer_i,
                                :width,
                            ] = score_candidates(
                                models[model_name],
                                model_info,
                                means,
                            )
                        component_true[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = execution['true']
                        component_pos_l2[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = execution['true_pos_l2']
                        component_angle[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = execution['true_angle']
                        component_success[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = execution['success']
                        component_terminal[
                            state_i,
                            generator_i,
                            round_i,
                            selector_i,
                            :width,
                        ] = execution['terminal_state']
                        total_executions += width

            print(
                f'[{state_i + 1}/{num_states}] '
                f'rows={int(data["rows"][state_i])} '
                f'executions={total_executions} '
                f'stored_true_maxdiff={max_stored_true_mismatch:.3e}',
                flush=True,
            )
    finally:
        world.close()

    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    if max_stored_success_mismatch:
        raise RuntimeError('re-executed stored mean success differs from source')

    audit = {
        'version': 1,
        'source': str(source_path.resolve()),
        'source_sha256': sha256(source_path),
        'state_start': start,
        'num_states': num_states,
        'elite': elite,
        'neighbors': neighbors,
        'models': model_names,
        'generators': generator_names,
        'steps': data['steps'].astype(int).tolist(),
        'total_executions': total_executions,
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
        'max_stored_true_mismatch': max_stored_true_mismatch,
        'max_stored_success_mismatch': max_stored_success_mismatch,
    }
    atomic_savez(
        output_path,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=data['rows'].astype(np.int64),
        episodes=data['episodes'].astype(np.int64),
        starts=data['starts'].astype(np.int64),
        generators=data['generators'],
        scorers=data['scorers'],
        steps=data['steps'].astype(np.int64),
        global_selectors=np.asarray(GLOBAL_SELECTORS),
        component_selectors=np.asarray(COMPONENT_SELECTORS),
        global_action=global_action,
        global_pred=global_pred,
        global_true=global_true,
        global_pos_l2=global_pos_l2,
        global_angle=global_angle,
        global_success=global_success,
        global_terminal=global_terminal,
        component_count=component_count,
        component_size=component_size,
        component_action=component_action,
        component_pred=component_pred,
        component_true=component_true,
        component_pos_l2=component_pos_l2,
        component_angle=component_angle,
        component_success=component_success,
        component_terminal=component_terminal,
    )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    print(f'counterfactual refit -> {output_path}', flush=True)


if __name__ == '__main__':
    run()
