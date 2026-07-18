"""Candidate-rank oracle for the Horizon-Bundle Gates A/B.

For each dataset-seeded PushT state, this script either:

1. runs one CEM solve, selects a stratified subset of the final candidate
   population, executes those candidates in a reset simulator, and saves a
   reusable candidate bank; or
2. loads that bank and scores exactly the same candidates with another
   checkpoint.

The resulting comparisons are paired over state, goal, and action sequence.
Reported metrics include Spearman/Kendall rank correlation, pairwise inversion
rate, top-k precision, and simple regret.

Reference-bank smoke run:

  python scripts/plan/candidate_oracle.py \
      policy=iter2_multistep/weights_epoch_30.pt \
      +plan_config.history_len=3 \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.dataset_name=/path/to/pusht_expert_train.h5 \
      eval.goal_offset_steps=40 \
      +oracle.num_states=2 +oracle.per_state=8 \
      +oracle.bank=outputs/week1/bank_h5_off40.npz \
      +oracle.out=outputs/week1/oracle_K5_smoke.npz

Paired evaluation of another checkpoint:

  python scripts/plan/candidate_oracle.py \
      policy=iter2_baseline/weights_epoch_30.pt \
      +plan_config.history_len=3 \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.dataset_name=/path/to/pusht_expert_train.h5 \
      eval.goal_offset_steps=40 \
      +oracle.bank=outputs/week1/bank_h5_off40.npz \
      +oracle.out=outputs/week1/oracle_K1_h5_off40.npz
"""

from copy import deepcopy
import math
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn import preprocessing

import stable_worldmodel as swm
from stable_worldmodel.solver.callbacks.candidate_recorder import (
    CandidateRecorder,
)
from stable_worldmodel.world.world import (
    _apply_callables,
    _extract_init_goal,
    _refresh_dataset_rendered_images,
    _seed_history_infos,
)

# Reuse the canonical evaluation data/transform setup.
from eval_wm import get_dataset, img_transform  # noqa: E402


def stratify(costs: np.ndarray, per_state: int, rng) -> np.ndarray:
    """Select distinct top, near-tie, quantile, and random candidates."""
    costs = np.asarray(costs)
    if costs.ndim != 1:
        raise ValueError(f'costs must be one-dimensional, got {costs.shape}')
    k = min(int(per_state), len(costs))
    if k < 2:
        raise ValueError(
            'oracle.per_state must select at least two candidates'
        )

    order = np.argsort(costs, kind='stable')
    quart = max(1, k // 4)
    proposed = [
        *order[:quart],
        *order[quart : 2 * quart],
        *order[np.linspace(0, len(order) - 1, quart, dtype=int)],
        *rng.permutation(len(order)),
        *order,
    ]
    selected = []
    seen = set()
    for idx in proposed:
        idx = int(idx)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) == k:
            break
    return np.asarray(selected, dtype=np.int64)


def sample_starts(dataset, num_states: int, goal_offset: int, rng):
    """Exactly match eval_wm's row-uniform sampling protocol."""
    col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_idx = np.asarray(dataset.get_col_data(col))
    step_idx = np.asarray(dataset.get_col_data('step_idx'))
    episodes, inverse = np.unique(episode_idx, return_inverse=True)
    max_step = np.full(len(episodes), -1, dtype=np.int64)
    np.maximum.at(max_step, inverse, step_idx)
    valid = np.flatnonzero(step_idx <= max_step[inverse] - goal_offset)
    # ``eval_wm`` samples integer positions from ``len(valid) - 1`` and then
    # indexes the valid-row array. Preserve that historical off-by-one here so
    # the same seed gives the exact same ordered starts for paired audits.
    population = len(valid) - 1
    if population < num_states:
        raise ValueError(
            f'Only {population} evaluator-matched starts for goal offset '
            f'{goal_offset}, '
            f'but oracle.num_states={num_states}'
        )
    positions = rng.choice(population, size=num_states, replace=False)
    rows = np.sort(valid[positions])
    return rows, episode_idx[rows], step_idx[rows]


def make_process(dataset, keys) -> dict:
    process = {}
    for col in keys:
        if col == 'pixels':
            continue
        proc = preprocessing.StandardScaler()
        col_data = np.asarray(dataset.get_col_data(col))
        proc.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = proc
        if col != 'action':
            process[f'goal_{col}'] = proc
    process['action_hist'] = process['action']
    return process


def prepare_world_info(
    world,
    dataset,
    *,
    episode: int,
    start: int,
    goal_offset: int,
    callables,
    history_len: int,
    action_block: int,
):
    """Reproduce World._evaluate_from_dataset up to the first plan call."""
    init_state, goal_state, _ = _extract_init_goal(
        dataset, [int(episode)], [int(start)], int(goal_offset)
    )
    world.reset(seed=init_state.get('seed'), options=None)

    if callables:
        merged = {**init_state, **goal_state}
        env_init = {key: value[0] for key, value in merged.items()}
        _apply_callables(world.envs.envs[0].unwrapped, callables, env_init)

    shape_prefix = world.infos['pixels'].shape[:2]
    for source in (init_state, goal_state):
        for key, value in source.items():
            if key in ('pixels', 'goal'):
                continue
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()

    history_frames = max(0, int(history_len) - 1)
    if history_frames:
        _seed_history_infos(
            world.envs.envs,
            world.infos,
            dataset,
            [int(episode)],
            [int(start)],
            history_frames,
            int(action_block),
        )

    _refresh_dataset_rendered_images(
        world.envs.envs, world.infos, init_state, goal_state
    )
    return (
        deepcopy(world.infos),
        np.asarray(init_state['state'][0]).copy(),
        np.asarray(goal_state['goal_state'][0]).copy(),
    )


def prepare_model_info(policy, info: dict) -> dict:
    """Prepare the exact first-plan model inputs without mutating MPC state."""
    prepared = policy._prepare_info(deepcopy(info))
    pixels_hist = prepared.pop('pixels_hist', None)
    action_hist = prepared.pop('action_hist', None)
    prepared.pop('_needs_flush', None)

    history_len = int(policy.cfg.history_len)
    if history_len > 1:
        if pixels_hist is None:
            pixels_hist = prepared['pixels'].repeat(
                1, history_len - 1, 1, 1, 1
            )
        prepared['pixels'] = torch.cat(
            [pixels_hist, prepared['pixels']], dim=1
        )[:, -history_len:]

        need = (history_len - 1) * int(policy.cfg.action_block)
        action_dim = policy.env.single_action_space.shape[-1]
        if action_hist is None:
            action_hist = torch.zeros(
                1, need, action_dim, dtype=prepared['pixels'].dtype
            )
        prepared['past_action'] = action_hist[:, -need:].reshape(
            1, history_len - 1, -1
        )
    return prepared


@torch.inference_mode()
def score_candidates(model, model_info: dict, candidates: np.ndarray):
    """Score a fixed normalized-action candidate set with one checkpoint."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    candidates_t = torch.as_tensor(
        candidates, device=device, dtype=dtype
    ).unsqueeze(0)
    n_candidates = candidates_t.shape[1]

    expanded = {}
    for key, value in model_info.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            expanded[key] = (
                value.to(device=device, dtype=target_dtype)
                .unsqueeze(1)
                .expand(1, n_candidates, *value.shape[1:])
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(
                value[:, None, ...], n_candidates, axis=1
            )
        else:
            expanded[key] = value
    return (
        model.get_cost(expanded, candidates_t).squeeze(0).float().cpu().numpy()
    )


def task_cost(goal_state: np.ndarray, state: np.ndarray):
    """PushT eval-aligned pose cost with a wrapped, threshold-scaled angle."""
    pos_l2 = float(np.linalg.norm(goal_state[:4] - state[:4]))
    angle = float(abs(goal_state[4] - state[4]) % (2 * math.pi))
    angle = min(angle, 2 * math.pi - angle)
    angle_scale = 20.0 / (math.pi / 9.0)
    cost = float(math.hypot(pos_l2, angle_scale * angle))
    success = bool(pos_l2 < 20.0 and angle < math.pi / 9.0)
    return cost, pos_l2, angle, success


def execute_candidate(
    env,
    *,
    initial_state: np.ndarray,
    goal_state: np.ndarray,
    candidate: np.ndarray,
    action_scaler,
    action_block: int,
    seed: int,
):
    """Reset and execute one normalized candidate open-loop."""
    env.reset(seed=int(seed))
    raw = env.unwrapped
    raw._set_goal_state(goal_state)
    if hasattr(raw, 'goal_pose'):
        raw.goal_pose = np.asarray(
            [goal_state[2], goal_state[3], goal_state[4]]
        )
    raw._set_state(initial_state)

    action_dim = env.action_space.shape[-1]
    expected = int(action_block) * action_dim
    if candidate.shape[-1] != expected:
        raise ValueError(
            f'Candidate dim {candidate.shape[-1]} != '
            f'action_block*action_dim {expected}'
        )
    normalized = candidate.reshape(-1, action_dim)
    actions = action_scaler.inverse_transform(normalized).astype(
        env.action_space.dtype, copy=False
    )
    roundtrip = action_scaler.transform(actions)
    roundtrip_error = float(np.max(np.abs(roundtrip - normalized)))

    terminated = truncated = False
    for action in actions:
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    terminal_state = np.asarray(raw._get_obs()).copy()
    cost, pos_l2, angle, success = task_cost(goal_state, terminal_state)
    return {
        'cost': cost,
        'pos_l2': pos_l2,
        'angle': angle,
        'success': success,
        'terminal_state': terminal_state,
        'roundtrip_error': roundtrip_error,
    }


def rank_metrics(pred: np.ndarray, true: np.ndarray):
    from scipy.stats import kendalltau, spearmanr

    pred = np.asarray(pred)
    true = np.asarray(true)
    n = len(pred)
    pairs = np.asarray(
        [(i, j) for i in range(n) for j in range(i + 1, n)],
        dtype=np.int64,
    )
    pred_delta = pred[pairs[:, 0]] - pred[pairs[:, 1]]
    true_delta = true[pairs[:, 0]] - true[pairs[:, 1]]
    non_tie = (np.abs(pred_delta) > 1e-12) & (np.abs(true_delta) > 1e-12)
    inversion = (
        float(
            np.mean(
                np.sign(pred_delta[non_tie]) != np.sign(true_delta[non_tie])
            )
        )
        if non_tie.any()
        else float('nan')
    )
    k = max(1, n // 4)
    pred_top = set(np.argsort(pred)[:k].tolist())
    true_top = set(np.argsort(true)[:k].tolist())
    return {
        'spearman': float(spearmanr(pred, true).statistic),
        'kendall': float(kendalltau(pred, true).statistic),
        'inversion': inversion,
        'topk_precision': len(pred_top & true_top) / k,
        'regret': float(true[np.argmin(pred)] - true.min()),
    }


def load_bank(path: Path, cfg):
    bank = np.load(path, allow_pickle=False)
    expected = {
        'goal_offset': int(cfg.eval.goal_offset_steps),
        'horizon': int(cfg.plan_config.horizon),
        'action_block': int(cfg.plan_config.action_block),
    }
    for key, value in expected.items():
        actual = int(np.asarray(bank[key]).item())
        if actual != value:
            raise ValueError(
                f'Bank {key}={actual}, requested configuration has {value}'
            )
    return bank


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    ocfg = cfg.get('oracle', {})
    num_states = int(ocfg.get('num_states', 40))
    per_state = int(ocfg.get('per_state', 24))
    out_path = Path(str(ocfg.get('out', 'outputs/week1/oracle.npz')))
    bank_raw = ocfg.get('bank', None)
    bank_path = Path(str(bank_raw)) if bank_raw else None

    # This audit plans and replays one state at a time.
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

    model = swm.wm.utils.load_pretrained(cfg.policy).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    recorder = CandidateRecorder(n_steps=solver.n_steps)
    solver.callbacks.append(recorder)
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )
    world.set_policy(policy)

    rng = np.random.default_rng(cfg.seed)
    loaded = bank_path is not None and bank_path.exists()
    if loaded:
        bank = load_bank(bank_path, cfg)
        rows = np.asarray(bank['rows'])
        episodes = np.asarray(bank['episodes'])
        starts = np.asarray(bank['starts'])
        fixed_candidates = np.asarray(bank['candidates'])
        fixed_true = np.asarray(bank['true_cost'])
        fixed_pos = np.asarray(bank['true_pos_l2'])
        fixed_angle = np.asarray(bank['true_angle'])
        fixed_success = np.asarray(bank['success'])
        fixed_terminal = np.asarray(bank['terminal_state'])
        num_states = len(episodes)
    else:
        rows, episodes, starts = sample_starts(
            dataset,
            num_states,
            int(cfg.eval.goal_offset_steps),
            rng,
        )
        fixed_candidates = fixed_true = fixed_pos = fixed_angle = None
        fixed_success = fixed_terminal = None

    candidate_rows = []
    predicted_rows = []
    true_rows = []
    pos_rows = []
    angle_rows = []
    success_rows = []
    terminal_rows = []
    reset_errors = []
    roundtrip_errors = []

    callables = cfg.eval.get('callables')
    if callables is not None:
        from omegaconf import OmegaConf

        callables = OmegaConf.to_container(callables, resolve=True)

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
            model_info = prepare_model_info(policy, info)

            if loaded:
                candidates = fixed_candidates[state_i]
                predicted = score_candidates(model, model_info, candidates)
                true = fixed_true[state_i]
                pos_l2 = fixed_pos[state_i]
                angle = fixed_angle[state_i]
                successes = fixed_success[state_i]
                terminals = fixed_terminal[state_i]
            else:
                solver(model_info)
                record = recorder.history[-1][-1]
                population = record['candidates'][0].float().numpy()
                population_cost = record['costs'][0].numpy()
                selected = stratify(population_cost, per_state, rng)
                candidates = population[selected]
                # CandidateRecorder stores float16 to keep full CEM populations
                # cheap. Re-score the stored candidates so the reference model
                # and every paired model see exactly the same quantized bank.
                predicted = score_candidates(model, model_info, candidates)

                executions = [
                    execute_candidate(
                        world.envs.envs[0],
                        initial_state=initial_state,
                        goal_state=goal_state,
                        candidate=candidate,
                        action_scaler=process['action'],
                        action_block=int(cfg.plan_config.action_block),
                        seed=int(cfg.seed) + state_i,
                    )
                    for candidate in candidates
                ]
                true = np.asarray([item['cost'] for item in executions])
                pos_l2 = np.asarray([item['pos_l2'] for item in executions])
                angle = np.asarray([item['angle'] for item in executions])
                successes = np.asarray(
                    [item['success'] for item in executions], dtype=bool
                )
                terminals = np.stack(
                    [item['terminal_state'] for item in executions]
                )
                roundtrip_errors.extend(
                    item['roundtrip_error'] for item in executions
                )

                # First-state reset determinism audit.
                if state_i == 0:
                    repeat = execute_candidate(
                        world.envs.envs[0],
                        initial_state=initial_state,
                        goal_state=goal_state,
                        candidate=candidates[0],
                        action_scaler=process['action'],
                        action_block=int(cfg.plan_config.action_block),
                        seed=int(cfg.seed) + state_i,
                    )
                    reset_errors.append(
                        float(
                            np.max(
                                np.abs(repeat['terminal_state'] - terminals[0])
                            )
                        )
                    )

            candidate_rows.append(candidates)
            predicted_rows.append(predicted)
            true_rows.append(true)
            pos_rows.append(pos_l2)
            angle_rows.append(angle)
            success_rows.append(successes)
            terminal_rows.append(terminals)
            print(
                f'[{state_i + 1}/{num_states}] episode={int(episode)} '
                f'start={int(start)} '
                f'spearman={rank_metrics(predicted, true)["spearman"]:.3f}'
            )
    finally:
        world.close()

    candidates = np.stack(candidate_rows)
    predicted = np.stack(predicted_rows)
    true = np.stack(true_rows)
    true_pos = np.stack(pos_rows)
    true_angle = np.stack(angle_rows)
    successes = np.stack(success_rows)
    terminals = np.stack(terminal_rows)
    metrics = [rank_metrics(predicted[i], true[i]) for i in range(num_states)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        rows=rows,
        episodes=episodes,
        starts=starts,
        candidates=candidates,
        pred=predicted,
        true=true,
        true_pos_l2=true_pos,
        true_angle=true_angle,
        success=successes,
        terminal_state=terminals,
        spearman=np.asarray([item['spearman'] for item in metrics]),
        kendall=np.asarray([item['kendall'] for item in metrics]),
        inversion=np.asarray([item['inversion'] for item in metrics]),
        topk_precision=np.asarray(
            [item['topk_precision'] for item in metrics]
        ),
        regret=np.asarray([item['regret'] for item in metrics]),
    )

    if not loaded and bank_path is not None:
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            bank_path,
            version=np.asarray(1),
            reference_policy=np.asarray(str(cfg.policy)),
            goal_offset=np.asarray(int(cfg.eval.goal_offset_steps)),
            horizon=np.asarray(int(cfg.plan_config.horizon)),
            action_block=np.asarray(int(cfg.plan_config.action_block)),
            rows=rows,
            episodes=episodes,
            starts=starts,
            candidates=candidates,
            true_cost=true,
            true_pos_l2=true_pos,
            true_angle=true_angle,
            success=successes,
            terminal_state=terminals,
        )

    means = {
        key: float(np.nanmean([item[key] for item in metrics]))
        for key in metrics[0]
    }
    reset_error = max(reset_errors, default=float('nan'))
    roundtrip_error = max(roundtrip_errors, default=float('nan'))
    print(
        f'{cfg.policy}: '
        f'spearman={means["spearman"]:.3f} '
        f'kendall={means["kendall"]:.3f} '
        f'inversion={means["inversion"]:.3f} '
        f'topk_precision={means["topk_precision"]:.3f} '
        f'regret={means["regret"]:.3f}'
    )
    print(
        f'smoke: reset_terminal_max_abs={reset_error:.3g} '
        f'action_roundtrip_max_abs={roundtrip_error:.3g} '
        f'successes={int(successes.sum())}/{successes.size}'
    )
    print(f'results -> {out_path}')
    if bank_path is not None:
        print(f'candidate bank -> {bank_path}')


if __name__ == '__main__':
    run()
