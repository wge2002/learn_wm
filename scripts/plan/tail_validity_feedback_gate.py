"""Closed-loop falsification gate for deployable tail-validity feedback.

The existing PushT oracle traces evaluate CEM populations at isolated dataset
states.  They cannot answer whether an *executed* model error is useful at the
next MPC replan, because the dynamics replay cache was deliberately sampled
away from those trace rows.  This collector closes exactly that gap:

1. load a frozen oracle trace and take its final CEM mean;
2. execute one action block in the real simulator;
3. form the honest latent residual ``actual - predicted``;
4. replan from the reached state with common random numbers; and
5. compare ordinary CEM with residual-persistent costs at every CEM round.

Only the final population of each recursive arm is simulator-scored.  The
correction is intentionally tiny and interpretable::

    corrected_terminal = predicted_terminal + alpha * prefix_residual

The script is a feasibility gate, not a claim that persistent additive error
is the final method.  State-held-out alpha selection and all statistics live in
``summarize_tail_validity_feedback_gate.py``.
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
import torch
from torch import nn

import stable_worldmodel as swm
from stable_worldmodel.solver.callbacks import CEMPopulationRecorder

from candidate_oracle import (
    execute_candidate,
    make_process,
    prepare_model_info,
    prepare_world_info,
)
from cem_round_oracle import execute_population
from eval_wm import get_dataset, img_transform


warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def comma_floats(value) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(',')]
    if not values or any(not np.isfinite(item) for item in values):
        raise ValueError('feedback.alphas must contain finite floats')
    if len(set(values)) != len(values):
        raise ValueError('feedback.alphas must not contain duplicates')
    if 0.0 not in values:
        raise ValueError('feedback.alphas must contain the baseline alpha=0')
    return values


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def expand_model_info(
    info: dict,
    *,
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    expanded = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            expanded[key] = (
                value.to(device=device, dtype=target_dtype)
                .unsqueeze(1)
                .expand(1, num_candidates, *value.shape[1:])
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(
                value[:, None, ...],
                num_candidates,
                axis=1,
            )
        else:
            expanded[key] = value
    return expanded


@torch.no_grad()
def cache_state_embeddings(
    model: nn.Module,
    model_info: dict,
    *,
    action_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    expanded = expand_model_info(
        model_info,
        num_candidates=1,
        device=device,
        dtype=dtype,
    )
    dummy = torch.zeros(
        (1, 1, *action_shape),
        device=device,
        dtype=dtype,
    )
    model.get_cost(expanded, dummy)
    history = int(expanded['emb'].shape[2])
    if 'past_action' not in expanded:
        raise ValueError('matched-history trace must provide past_action')
    return {
        'emb': expanded['emb'][:, :1].detach().clone(),
        'goal_emb': expanded['goal_emb'].detach().clone(),
        'past_action': expanded['past_action'][:, :1].detach().clone(),
        'history': torch.asarray(history, device=device),
    }


class PersistentResidualCost(nn.Module):
    """Apply a previous executed-prefix residual to every candidate endpoint."""

    def __init__(self, base: nn.Module, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.alpha = float(alpha)
        self.register_buffer('_residual', torch.empty(0), persistent=False)

    def set_residual(self, residual: torch.Tensor) -> None:
        if residual.ndim != 1:
            raise ValueError(
                f'prefix residual must be one-dimensional, got {residual.shape}'
            )
        self._residual = residual.detach().clone()

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        if self._residual.numel() == 0:
            raise RuntimeError('set_residual must be called before CEM')
        self.base.get_cost(info_dict, action_candidates)
        terminal = info_dict['predicted_emb'][..., -1, :]
        goal = info_dict['goal_emb'][:, None, -1, :]
        residual = self._residual.to(
            device=terminal.device,
            dtype=terminal.dtype,
        ).reshape(1, 1, -1)
        corrected = terminal + self.alpha * residual
        return (corrected - goal).square().sum(dim=-1)


def cached_rollout(
    model: nn.Module,
    cache: dict[str, torch.Tensor],
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact quantized-candidate trajectories and ordinary costs."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    candidate_tensor = torch.as_tensor(
        candidates,
        device=device,
        dtype=dtype,
    )
    num_candidates = len(candidate_tensor)
    history = int(cache['history'])
    info = {
        'pixels': torch.empty(
            (1, num_candidates, history),
            device=device,
            dtype=dtype,
        ),
        'goal': torch.empty(
            (1, num_candidates, 1),
            device=device,
            dtype=dtype,
        ),
        'past_action': cache['past_action'].expand(
            1,
            num_candidates,
            *cache['past_action'].shape[2:],
        ),
        'emb': cache['emb'].expand(
            1,
            num_candidates,
            *cache['emb'].shape[2:],
        ),
        'goal_emb': cache['goal_emb'],
    }
    with torch.inference_mode():
        cost = model.get_cost(info, candidate_tensor.unsqueeze(0))[0]
    return (
        info['predicted_emb'][0].float().cpu().numpy(),
        cost.float().cpu().numpy(),
    )


def encode_current(model: nn.Module, pixels: torch.Tensor) -> np.ndarray:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        encoded = model.encode(
            {'pixels': pixels.to(device=device, dtype=dtype)}
        )['emb']
    return encoded[0, -1].float().cpu().numpy()


def corrected_cost(
    terminal: np.ndarray,
    goal: np.ndarray,
    residual: np.ndarray,
    alpha: float,
) -> np.ndarray:
    delta = terminal + float(alpha) * residual[None] - goal[None]
    return np.square(delta, dtype=np.float64).sum(axis=-1)


def load_source(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f'feedback.source does not exist: {path}')
    with np.load(path, allow_pickle=False) as archive:
        source = {key: np.asarray(archive[key]) for key in archive.files}
    required = {
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
        'generators',
        'steps',
        'mean',
        'horizon',
        'goal_offset',
        'action_block',
    }
    missing = sorted(required - set(source))
    if missing:
        raise ValueError(f'feedback.source is missing fields {missing}')
    return source


def build_next_model_info(
    policy,
    world_info: dict,
    previous_model_info: dict,
    normalized_prefix: np.ndarray,
) -> dict:
    """Build the exact three-frame history after one executed action block."""
    prepared = policy._prepare_info(deepcopy(world_info))
    prepared.pop('pixels_hist', None)
    prepared.pop('action_hist', None)
    prepared.pop('_needs_flush', None)

    previous_pixels = previous_model_info['pixels']
    current_pixels = prepared['pixels']
    prepared['pixels'] = torch.cat(
        [previous_pixels[:, -2:], current_pixels[:, -1:]],
        dim=1,
    )

    previous_past = previous_model_info['past_action']
    executed_block = torch.as_tensor(normalized_prefix).reshape(1, 1, -1)
    prepared['past_action'] = torch.cat(
        [previous_past[:, -1:], executed_block],
        dim=1,
    )
    return prepared


def execute_prefix(
    world,
    *,
    normalized_prefix: np.ndarray,
    action_scaler,
    goal_snapshot: dict,
) -> dict:
    action_dim = world.envs.single_action_space.shape[-1]
    normalized = np.asarray(normalized_prefix, dtype=np.float32).reshape(
        -1, action_dim
    )
    actions = action_scaler.inverse_transform(normalized).astype(
        world.envs.single_action_space.dtype,
        copy=False,
    )
    roundtrip = action_scaler.transform(actions)
    roundtrip_error = float(np.max(np.abs(roundtrip - normalized)))

    terminated = truncated = False
    executed = 0
    for action in actions:
        _, _, term, trunc, info = world.envs.step(action[None])
        world.infos = info
        for key, value in goal_snapshot.items():
            world.infos[key] = deepcopy(value)
        terminated = bool(term[0])
        truncated = bool(trunc[0])
        executed += 1
        if terminated or truncated:
            break
    return {
        'terminated': terminated,
        'truncated': truncated,
        'executed': executed,
        'roundtrip_error': roundtrip_error,
    }


def stack_rows(rows: list[np.ndarray], *, name: str) -> np.ndarray:
    if not rows:
        raise RuntimeError(f'no valid rows collected for {name}')
    return np.stack(rows)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    feedback = cfg.get('feedback', {})
    source_path = Path(str(feedback.get('source', '')))
    output_path = Path(str(feedback.get('out', '')))
    if output_path == Path('.'):
        raise ValueError('feedback.out is required')
    if output_path.exists() and not bool(feedback.get('overwrite', False)):
        raise FileExistsError(
            f'feedback output exists: {output_path}; set overwrite=true'
        )

    source = load_source(source_path)
    policy_name = str(feedback.get('policy', 'pd_d192_k3_eval'))
    alphas = comma_floats(feedback.get('alphas', '-1,0,0.5,1'))
    state_start = int(feedback.get('state_start', 0))
    num_states = int(feedback.get('num_states', len(source['rows'])))
    prefix_blocks = int(feedback.get('prefix_blocks', 1))
    if prefix_blocks != 1:
        raise ValueError(
            'this locked gate requires exactly one executed action block'
        )
    if state_start < 0 or num_states < 1:
        raise ValueError('state_start must be nonnegative and num_states positive')
    if state_start + num_states > len(source['rows']):
        raise ValueError('requested source slice exceeds the source archive')

    for field, expected in (
        ('horizon', int(cfg.plan_config.horizon)),
        ('goal_offset', int(cfg.eval.goal_offset_steps)),
        ('action_block', int(cfg.plan_config.action_block)),
    ):
        actual = int(np.asarray(source[field]).item())
        if actual != expected:
            raise ValueError(
                f'source {field}={actual}, current config requires {expected}'
            )
    if int(cfg.plan_config.history_len) != 3:
        raise ValueError('the locked gate requires plan_config.history_len=3')

    generators = source['generators'].astype(str).tolist()
    if policy_name not in generators:
        raise ValueError(
            f'feedback policy {policy_name!r} is not a source generator '
            f'{generators}'
        )
    generator_i = generators.index(policy_name)
    final_round_i = int(np.argmax(source['steps']))
    horizon = int(cfg.plan_config.horizon)
    action_block = int(cfg.plan_config.action_block)
    action_shape = tuple(source['mean'].shape[-2:])

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = horizon * action_block + action_block + 5
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = make_process(dataset, cfg.dataset.keys_to_cache)
    transform = {
        'pixels': img_transform(cfg),
        'pixels_hist': img_transform(cfg),
        'goal': img_transform(cfg),
    }

    model = swm.wm.utils.load_pretrained(policy_name).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    wrappers = [PersistentResidualCost(model, alpha) for alpha in alphas]
    recorders = []
    solvers = []
    for wrapper in wrappers:
        solver = hydra.utils.instantiate(cfg.solver, model=wrapper)
        recorder = CEMPopulationRecorder([solver.n_steps - 1])
        solver.callbacks.append(recorder)
        solvers.append(solver)
        recorders.append(recorder)

    config = swm.PlanConfig(**cfg.plan_config)
    template_policy = swm.policy.WorldModelPolicy(
        solver=solvers[alphas.index(0.0)],
        config=config,
        process=process,
        transform=transform,
    )
    world.set_policy(template_policy)
    for solver in solvers:
        solver.configure(
            action_space=world.envs.action_space,
            n_envs=1,
            config=config,
        )

    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    requested_indices = np.arange(
        state_start,
        state_start + num_states,
        dtype=np.int64,
    )
    prefix_terminated = np.zeros(num_states, dtype=bool)
    prefix_truncated = np.zeros(num_states, dtype=bool)
    prefix_steps = np.zeros(num_states, dtype=np.int64)
    prefix_roundtrip = np.full(num_states, np.nan, dtype=np.float64)

    valid_source_indices = []
    rows = []
    episodes = []
    starts = []
    initial_states = []
    next_states = []
    goal_states = []
    prefix_plans = []
    current_embeddings = []
    predicted_prefix_embeddings = []
    actual_prefix_embeddings = []
    prefix_residuals = []
    goal_embeddings = []
    candidates_rows = []
    terminal_embedding_rows = []
    base_pred_rows = []
    corrected_pred_rows = []
    true_rows = []
    success_rows = []
    terminal_state_rows = []
    mean_rows = []
    mean_base_pred_rows = []
    mean_corrected_pred_rows = []
    mean_true_rows = []
    mean_success_rows = []
    mean_terminal_state_rows = []

    max_initial_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_recorded_cost_mismatch = 0.0
    max_roundtrip_error = 0.0
    started_at = time.time()

    try:
        for local_i, source_i in enumerate(requested_indices):
            episode = int(source['episodes'][source_i])
            start = int(source['starts'][source_i])
            info, initial_state, goal_state = prepare_world_info(
                world,
                dataset,
                episode=episode,
                start=start,
                goal_offset=int(cfg.eval.goal_offset_steps),
                callables=callables,
                history_len=int(cfg.plan_config.history_len),
                action_block=action_block,
            )
            max_initial_mismatch = max(
                max_initial_mismatch,
                float(
                    np.max(
                        np.abs(initial_state - source['initial_state'][source_i])
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(np.abs(goal_state - source['goal_state'][source_i]))
                ),
            )

            model_info_t0 = prepare_model_info(template_policy, info)
            cache_t0 = cache_state_embeddings(
                model,
                model_info_t0,
                action_shape=action_shape,
            )
            source_plan = np.asarray(
                source['mean'][source_i, generator_i, final_round_i],
                dtype=np.float32,
            )
            source_trajectory, _ = cached_rollout(
                model,
                cache_t0,
                source_plan[None],
            )
            history = int(cache_t0['history'])
            predicted_prefix = source_trajectory[0, history]
            current_embedding = (
                cache_t0['emb'][0, 0, -1].float().cpu().numpy()
            )

            goal_snapshot = {
                key: deepcopy(value)
                for key, value in world.infos.items()
                if key == 'goal' or key.startswith('goal_')
            }
            normalized_prefix = source_plan[:prefix_blocks]
            prefix_result = execute_prefix(
                world,
                normalized_prefix=normalized_prefix,
                action_scaler=process['action'],
                goal_snapshot=goal_snapshot,
            )
            prefix_terminated[local_i] = prefix_result['terminated']
            prefix_truncated[local_i] = prefix_result['truncated']
            prefix_steps[local_i] = prefix_result['executed']
            prefix_roundtrip[local_i] = prefix_result['roundtrip_error']
            max_roundtrip_error = max(
                max_roundtrip_error,
                prefix_result['roundtrip_error'],
            )
            if prefix_result['terminated'] or prefix_result['truncated']:
                print(
                    f'[{local_i + 1}/{num_states}] source={source_i} '
                    'prefix reached a terminal state; no next replan',
                    flush=True,
                )
                continue

            next_state = np.asarray(
                world.envs.envs[0].unwrapped._get_obs()
            ).copy()
            model_info_t1 = build_next_model_info(
                template_policy,
                world.infos,
                model_info_t0,
                normalized_prefix,
            )
            actual_prefix = encode_current(model, model_info_t1['pixels'][:, -1:])
            residual = actual_prefix - predicted_prefix
            cache_t1 = cache_state_embeddings(
                model,
                model_info_t1,
                action_shape=action_shape,
            )
            goal_embedding = (
                cache_t1['goal_emb'][0, -1].float().cpu().numpy()
            )

            state_candidates = []
            state_terminals = []
            state_base_pred = []
            state_corrected_pred = []
            state_true = []
            state_success = []
            state_terminal_state = []
            state_means = []
            state_mean_base_pred = []
            state_mean_corrected_pred = []
            state_mean_true = []
            state_mean_success = []
            state_mean_terminal_state = []
            execution_cache: dict[bytes, dict] = {}

            for arm_i, (alpha, wrapper, solver, recorder) in enumerate(
                zip(alphas, wrappers, solvers, recorders, strict=True)
            ):
                residual_tensor = torch.as_tensor(
                    residual,
                    device=next(model.parameters()).device,
                    dtype=next(model.parameters()).dtype,
                )
                wrapper.set_residual(residual_tensor)
                solver.torch_gen.manual_seed(
                    int(cfg.seed) + int(source_i) * 1009
                )
                outputs = solver(deepcopy(model_info_t1))
                record = recorder.history[-1][-1]
                candidates = record['candidates'][0].float().numpy()
                trajectory, base_cost = cached_rollout(
                    model,
                    cache_t1,
                    candidates,
                )
                terminal = trajectory[:, -1]
                calibrated_cost = corrected_cost(
                    terminal,
                    goal_embedding,
                    residual,
                    alpha,
                )
                mismatch = float(
                    np.max(
                        np.abs(
                            calibrated_cost
                            - record['costs'][0].float().numpy()
                        )
                    )
                )
                max_recorded_cost_mismatch = max(
                    max_recorded_cost_mismatch,
                    mismatch,
                )

                execution = execute_population(
                    world.envs.envs[0],
                    candidates=candidates,
                    initial_state=next_state,
                    goal_state=goal_state,
                    action_scaler=process['action'],
                    action_block=action_block,
                    seed=int(cfg.seed) + int(source_i),
                    cache=execution_cache,
                )
                mean = outputs['actions'][0].float().numpy().astype(
                    np.float16
                ).astype(np.float32)
                mean_trajectory, mean_base_cost = cached_rollout(
                    model,
                    cache_t1,
                    mean[None],
                )
                mean_calibrated_cost = corrected_cost(
                    mean_trajectory[:, -1],
                    goal_embedding,
                    residual,
                    alpha,
                )
                mean_execution = execute_candidate(
                    world.envs.envs[0],
                    initial_state=next_state,
                    goal_state=goal_state,
                    candidate=mean,
                    action_scaler=process['action'],
                    action_block=action_block,
                    seed=int(cfg.seed) + int(source_i),
                )
                max_roundtrip_error = max(
                    max_roundtrip_error,
                    float(np.max(execution['roundtrip_error'])),
                    float(mean_execution['roundtrip_error']),
                )

                state_candidates.append(candidates.astype(np.float16))
                state_terminals.append(terminal.astype(np.float16))
                state_base_pred.append(base_cost.astype(np.float32))
                state_corrected_pred.append(
                    calibrated_cost.astype(np.float32)
                )
                state_true.append(execution['true'])
                state_success.append(execution['success'])
                state_terminal_state.append(execution['terminal_state'])
                state_means.append(mean.astype(np.float16))
                state_mean_base_pred.append(float(mean_base_cost[0]))
                state_mean_corrected_pred.append(
                    float(mean_calibrated_cost[0])
                )
                state_mean_true.append(float(mean_execution['cost']))
                state_mean_success.append(bool(mean_execution['success']))
                state_mean_terminal_state.append(
                    mean_execution['terminal_state']
                )
                print(
                    f'[{local_i + 1}/{num_states}] source={source_i} '
                    f'arm={arm_i + 1}/{len(alphas)} alpha={alpha:g} '
                    f'residual={np.linalg.norm(residual):.3f} '
                    f'recall-ready support={int(execution["success"].any())} '
                    f'mean={mean_execution["cost"]:.2f}/'
                    f'{int(mean_execution["success"])} '
                    f'elapsed={(time.time() - started_at) / 60:.1f}m',
                    flush=True,
                )

            valid_source_indices.append(int(source_i))
            rows.append(int(source['rows'][source_i]))
            episodes.append(episode)
            starts.append(start)
            initial_states.append(initial_state)
            next_states.append(next_state)
            goal_states.append(goal_state)
            prefix_plans.append(normalized_prefix)
            current_embeddings.append(current_embedding)
            predicted_prefix_embeddings.append(predicted_prefix)
            actual_prefix_embeddings.append(actual_prefix)
            prefix_residuals.append(residual)
            goal_embeddings.append(goal_embedding)
            candidates_rows.append(np.stack(state_candidates))
            terminal_embedding_rows.append(np.stack(state_terminals))
            base_pred_rows.append(np.stack(state_base_pred))
            corrected_pred_rows.append(np.stack(state_corrected_pred))
            true_rows.append(np.stack(state_true))
            success_rows.append(np.stack(state_success))
            terminal_state_rows.append(np.stack(state_terminal_state))
            mean_rows.append(np.stack(state_means))
            mean_base_pred_rows.append(np.asarray(state_mean_base_pred))
            mean_corrected_pred_rows.append(
                np.asarray(state_mean_corrected_pred)
            )
            mean_true_rows.append(np.asarray(state_mean_true))
            mean_success_rows.append(np.asarray(state_mean_success))
            mean_terminal_state_rows.append(
                np.stack(state_mean_terminal_state)
            )
    finally:
        world.close()

    if max_initial_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            'source reconstruction mismatch: '
            f'initial={max_initial_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )

    audit = {
        'version': 1,
        'source': str(source_path.resolve()),
        'source_sha256': sha256(source_path),
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'policy': policy_name,
        'alphas': alphas,
        'state_start': state_start,
        'requested_states': num_states,
        'valid_replans': len(valid_source_indices),
        'prefix_terminal_or_truncated': int(
            np.sum(prefix_terminated | prefix_truncated)
        ),
        'prefix_blocks': prefix_blocks,
        'prefix_environment_steps': prefix_blocks * action_block,
        'history_len': int(cfg.plan_config.history_len),
        'horizon': horizon,
        'goal_offset': int(cfg.eval.goal_offset_steps),
        'action_block': action_block,
        'cem_steps': int(solvers[0].n_steps),
        'num_samples': int(solvers[0].num_samples),
        'topk': int(solvers[0].topk),
        'max_initial_mismatch': max_initial_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
        'max_recorded_cost_mismatch': max_recorded_cost_mismatch,
        'max_roundtrip_error': max_roundtrip_error,
        'elapsed_seconds': time.time() - started_at,
    }
    atomic_savez(
        output_path,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        alphas=np.asarray(alphas, dtype=np.float64),
        requested_source_indices=requested_indices,
        requested_rows=source['rows'][requested_indices].astype(np.int64),
        prefix_terminated=prefix_terminated,
        prefix_truncated=prefix_truncated,
        prefix_steps=prefix_steps,
        prefix_roundtrip_error=prefix_roundtrip,
        source_indices=np.asarray(valid_source_indices, dtype=np.int64),
        rows=np.asarray(rows, dtype=np.int64),
        episodes=np.asarray(episodes, dtype=np.int64),
        starts=np.asarray(starts, dtype=np.int64),
        initial_state=stack_rows(initial_states, name='initial_state'),
        next_state=stack_rows(next_states, name='next_state'),
        goal_state=stack_rows(goal_states, name='goal_state'),
        prefix_plan=stack_rows(prefix_plans, name='prefix_plan').astype(
            np.float16
        ),
        current_embedding=stack_rows(
            current_embeddings,
            name='current_embedding',
        ).astype(np.float16),
        predicted_prefix_embedding=stack_rows(
            predicted_prefix_embeddings,
            name='predicted_prefix_embedding',
        ).astype(np.float16),
        actual_prefix_embedding=stack_rows(
            actual_prefix_embeddings,
            name='actual_prefix_embedding',
        ).astype(np.float16),
        prefix_residual=stack_rows(
            prefix_residuals,
            name='prefix_residual',
        ).astype(np.float16),
        goal_embedding=stack_rows(
            goal_embeddings,
            name='goal_embedding',
        ).astype(np.float16),
        candidates=stack_rows(candidates_rows, name='candidates').astype(
            np.float16
        ),
        terminal_embedding=stack_rows(
            terminal_embedding_rows,
            name='terminal_embedding',
        ).astype(np.float16),
        base_pred=stack_rows(base_pred_rows, name='base_pred').astype(
            np.float32
        ),
        corrected_pred=stack_rows(
            corrected_pred_rows,
            name='corrected_pred',
        ).astype(np.float32),
        true=stack_rows(true_rows, name='true').astype(np.float64),
        success=stack_rows(success_rows, name='success').astype(bool),
        terminal_state=stack_rows(
            terminal_state_rows,
            name='terminal_state',
        ).astype(np.float64),
        mean=stack_rows(mean_rows, name='mean').astype(np.float16),
        mean_base_pred=stack_rows(
            mean_base_pred_rows,
            name='mean_base_pred',
        ).astype(np.float32),
        mean_corrected_pred=stack_rows(
            mean_corrected_pred_rows,
            name='mean_corrected_pred',
        ).astype(np.float32),
        mean_true=stack_rows(mean_true_rows, name='mean_true').astype(
            np.float64
        ),
        mean_success=stack_rows(
            mean_success_rows,
            name='mean_success',
        ).astype(bool),
        mean_terminal_state=stack_rows(
            mean_terminal_state_rows,
            name='mean_terminal_state',
        ).astype(np.float64),
    )
    print(f'feedback gate shard -> {output_path}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
