"""Audit which terminal-state oracle should supervise search-aligned LeWM.

The existing optimizer-equivalence traces contain, for every saved CEM
population, both the model score and the terminal simulator state of every
candidate.  This script adds a third score without executing any action:

    visual true-terminal cost =
        ||phi_bar(render(s_T)) - phi_bar(render(s_goal))||^2

``phi_bar`` is a frozen LeWM encoder/projector.  On the same candidate
population, the script compares the CEM update induced by:

* the learned rollout cost already stored in the trace;
* the frozen visual true-terminal cost;
* the privileged PushT physical pose cost stored in the trace.

This separates two possible methods before expensive training:

* if the visual oracle tracks the physical oracle, operator supervision can
  remain observation-only;
* otherwise a physical/reward teacher is only a task-aware mechanism proof,
  and a learned reachability metric is required for the general method.

Example:

  CUDA_VISIBLE_DEVICES=0 python scripts/plan/oe_oracle_semantics_audit.py \
      +plan_config.history_len=3 \
      plan_config.horizon=5 plan_config.receding_horizon=5 \
      eval.goal_offset_steps=40 eval.video=false \
      eval.dataset_name=/path/to/pusht_eval_state_only.h5 \
      +semantics.source=/path/to/cem_round_h5_off40_n12_full_v2.npz \
      +semantics.out=/path/to/h5_visual_semantics.npz \
      +semantics.policy=pd_d192_k3_eval \
      +semantics.generator=pd_d192_k3_eval \
      +semantics.scorer=pd_d192_k3_eval \
      +semantics.steps=\"4,9,19,29\"
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import time

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import stable_worldmodel as swm
from stable_worldmodel.world.world import _set_goal_pose_from_state

from candidate_oracle import (
    prepare_world_info,
    rank_metrics,
    task_cost,
)
from eval_wm import get_dataset, img_transform
from oe_update_resample import (
    comma_ints,
    elite_moments,
    sha256,
    validate_source,
)


EPS = 1e-12


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    os.replace(temporary, path)


def select_named(values: np.ndarray, requested: str, *, name: str) -> int:
    labels = np.asarray(values).astype(str).tolist()
    if requested not in labels:
        raise ValueError(f'{name}={requested!r} is not in {labels}')
    return labels.index(requested)


def select_rounds(
    source_steps: np.ndarray,
    requested: str | None,
) -> tuple[list[int], list[int]]:
    steps = np.asarray(source_steps, dtype=np.int64).tolist()
    if requested is None or not str(requested).strip():
        return list(range(len(steps))), steps
    selected = comma_ints(requested, name='semantics.steps')
    missing = sorted(set(selected) - set(steps))
    if missing:
        raise ValueError(
            f'semantics.steps {missing} are not present in source {steps}'
        )
    return [steps.index(step) for step in selected], selected


def set_state_for_render(env, state: np.ndarray) -> float:
    """Restore a PushT observation without advancing physics another step.

    PushT's public ``_set_state`` advances pymunk by one ``dt``. That is useful
    when initializing an episode but would render a slightly later state than
    the terminal observation stored in the trace. Direct body assignment
    reconstructs the post-action framebuffer represented by ``state``.
    """
    state = np.asarray(state, dtype=np.float64)
    if not all(hasattr(env, name) for name in ('agent', 'block', '_get_obs')):
        env._set_state(state)
    else:
        env.agent.position = tuple(state[:2])
        # PushT's block has an offset center of gravity; match the canonical
        # setter's angle-before-position order.
        env.block.angle = float(state[4])
        env.block.position = tuple(state[2:4])
        if len(state) == 7:
            env.agent.velocity = tuple(state[-2:])
        if hasattr(env, 'space'):
            env.space.reindex_shapes_for_body(env.agent)
            env.space.reindex_shapes_for_body(env.block)
    restored = np.asarray(env._get_obs())
    # PushT pixels depend on agent xy, block xy, and block angle. Kinematic
    # velocity may be overwritten internally and is intentionally excluded.
    mismatch = float(np.max(np.abs(restored[:5] - state[:5])))
    if mismatch > 1e-5 and not getattr(
        set_state_for_render,
        '_reported_mismatch',
        False,
    ):
        print(
            'render-state mismatch: '
            f'target={state.tolist()} restored={restored.tolist()}',
            flush=True,
        )
        set_state_for_render._reported_mismatch = True
    return mismatch


@torch.inference_mode()
def encode_frames(
    model,
    frames: list[np.ndarray],
    *,
    transform,
    batch_size: int,
) -> np.ndarray:
    if not frames:
        raise ValueError('encode_frames requires at least one frame')
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    encoded = []
    for offset in range(0, len(frames), batch_size):
        batch = frames[offset : offset + batch_size]
        images = torch.stack([transform(frame) for frame in batch]).to(
            device=device,
            dtype=dtype,
        )
        info = model.encode({'pixels': images.unsqueeze(1)})
        encoded.append(info['emb'][:, 0].float().cpu().numpy())
    return np.concatenate(encoded, axis=0)


def update_metrics(
    costs: np.ndarray,
    *,
    candidates: np.ndarray,
    origin: np.ndarray,
    physical_cost: np.ndarray,
    physical_indices: np.ndarray,
    physical_mean: np.ndarray,
    topk: int,
    success: np.ndarray,
) -> dict[str, float | np.ndarray]:
    mean, std, indices = elite_moments(
        candidates,
        costs,
        topk=topk,
        std_floor=1e-6,
    )
    update = (mean - origin).reshape(-1).astype(np.float64)
    physical_update = (
        physical_mean - origin
    ).reshape(-1).astype(np.float64)
    norm = float(np.linalg.norm(update))
    physical_norm = float(np.linalg.norm(physical_update))
    cosine = float(
        np.dot(update, physical_update)
        / max(norm * physical_norm, EPS)
    )
    relative = float(
        np.linalg.norm(update - physical_update)
        / max(physical_norm, EPS)
    )
    overlap = len(
        set(indices.tolist()) & set(physical_indices.tolist())
    ) / len(physical_indices)
    rank = rank_metrics(costs, physical_cost)
    return {
        'mean': mean,
        'std': std,
        'indices': indices,
        'update_cosine': cosine,
        'relative_update_error': relative,
        'elite_overlap': float(overlap),
        'physical_elite_cost': float(np.mean(physical_cost[indices])),
        'elite_success_fraction': float(np.mean(success[indices])),
        'physical_top1_cost': float(physical_cost[np.argmin(costs)]),
        'top1_success': float(success[np.argmin(costs)]),
        'spearman': rank['spearman'],
        'kendall': rank['kendall'],
        'inversion': rank['inversion'],
    }


def diagonal_symmetric_kl(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    mean_b: np.ndarray,
    std_b: np.ndarray,
) -> float:
    mean_a = np.asarray(mean_a, dtype=np.float64).reshape(-1)
    mean_b = np.asarray(mean_b, dtype=np.float64).reshape(-1)
    var_a = np.square(np.asarray(std_a, dtype=np.float64).reshape(-1))
    var_b = np.square(np.asarray(std_b, dtype=np.float64).reshape(-1))
    var_a = np.maximum(var_a, 1e-12)
    var_b = np.maximum(var_b, 1e-12)
    delta_sq = np.square(mean_a - mean_b)
    kl_ab = 0.5 * np.sum(
        np.log(var_b / var_a) + (var_a + delta_sq) / var_b - 1.0
    )
    kl_ba = 0.5 * np.sum(
        np.log(var_a / var_b) + (var_b + delta_sq) / var_a - 1.0
    )
    return float(0.5 * (kl_ab + kl_ba) / len(mean_a))


def bootstrap_state_summary(
    rows: list[dict],
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row['state_index'])].append(row)
    state_ids = sorted(grouped)

    def state_values(key: str) -> np.ndarray:
        return np.asarray(
            [
                np.mean([float(row[key]) for row in grouped[state]])
                for state in state_ids
            ],
            dtype=np.float64,
        )

    learned_cosine = state_values('learned_update_cosine')
    visual_cosine = state_values('visual_update_cosine')
    learned_overlap = state_values('learned_elite_overlap')
    visual_overlap = state_values('visual_elite_overlap')
    learned_true = state_values('learned_physical_elite_cost')
    visual_true = state_values('visual_physical_elite_cost')
    physical_true = state_values('physical_elite_cost')
    visual_spearman = state_values('visual_physical_spearman')

    def calculate(indices: np.ndarray) -> dict[str, float]:
        learned = float(np.mean(learned_true[indices]))
        visual = float(np.mean(visual_true[indices]))
        physical = float(np.mean(physical_true[indices]))
        denominator = learned - physical
        recovery = (
            (learned - visual) / denominator
            if denominator > EPS
            else float('nan')
        )
        return {
            'learned_update_cosine': float(
                np.mean(learned_cosine[indices])
            ),
            'visual_update_cosine': float(np.mean(visual_cosine[indices])),
            'delta_update_cosine': float(
                np.mean(visual_cosine[indices] - learned_cosine[indices])
            ),
            'learned_elite_overlap': float(
                np.mean(learned_overlap[indices])
            ),
            'visual_elite_overlap': float(
                np.mean(visual_overlap[indices])
            ),
            'learned_physical_elite_cost': learned,
            'visual_physical_elite_cost': visual,
            'physical_elite_cost': physical,
            'visual_true_cost_gain': learned - visual,
            'visual_recovery_fraction': recovery,
            'visual_physical_spearman': float(
                np.mean(visual_spearman[indices])
            ),
        }

    observed = calculate(np.arange(len(state_ids)))
    if bootstrap <= 0:
        return {
            'num_states': len(state_ids),
            'mean': observed,
            'ci95': {},
        }

    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(bootstrap):
        indices = rng.integers(0, len(state_ids), size=len(state_ids))
        values = calculate(indices)
        for key, value in values.items():
            if np.isfinite(value):
                samples[key].append(value)
    intervals = {
        key: [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for key, values in samples.items()
        if values
    }
    return {
        'num_states': len(state_ids),
        'mean': observed,
        'ci95': intervals,
    }


def per_step_summary(rows: list[dict]) -> list[dict]:
    output = []
    for step in sorted({int(row['step']) for row in rows}):
        selected = [row for row in rows if int(row['step']) == step]
        learned = float(
            np.mean(
                [row['learned_physical_elite_cost'] for row in selected]
            )
        )
        visual = float(
            np.mean(
                [row['visual_physical_elite_cost'] for row in selected]
            )
        )
        physical = float(
            np.mean([row['physical_elite_cost'] for row in selected])
        )
        denominator = learned - physical
        output.append(
            {
                'step': step,
                'learned_update_cosine': float(
                    np.mean(
                        [row['learned_update_cosine'] for row in selected]
                    )
                ),
                'visual_update_cosine': float(
                    np.mean(
                        [row['visual_update_cosine'] for row in selected]
                    )
                ),
                'learned_elite_overlap': float(
                    np.mean(
                        [row['learned_elite_overlap'] for row in selected]
                    )
                ),
                'visual_elite_overlap': float(
                    np.mean(
                        [row['visual_elite_overlap'] for row in selected]
                    )
                ),
                'visual_physical_spearman': float(
                    np.mean(
                        [
                            row['visual_physical_spearman']
                            for row in selected
                        ]
                    )
                ),
                'visual_true_cost_gain': learned - visual,
                'visual_recovery_fraction': (
                    (learned - visual) / denominator
                    if denominator > EPS
                    else float('nan')
                ),
            }
        )
    return output


def format_number(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return 'nan'
    return f'{value:.{digits}f}'


def build_report(audit: dict, rows: list[dict]) -> str:
    summary = audit['summary']['mean']
    ci = audit['summary']['ci95']
    gate = audit['gate']
    lines = [
        '# OE oracle-semantics audit',
        '',
        f'- Source: `{audit["source"]}`',
        f'- Policy / generator / scorer: `{audit["policy"]}` / '
        f'`{audit["generator"]}` / `{audit["scorer"]}`',
        f'- Cell: H{audit["horizon"]} / off{audit["goal_offset"]}; '
        f'{audit["num_states"]} states; steps={audit["steps"]}; '
        f'N={audit["num_candidates"]}; top-k={audit["topk"]}',
        '',
        '## Integrity checks',
        '',
        f'- initial-state reconstruction max abs: '
        f'`{audit["integrity"]["max_initial_state_mismatch"]:.3e}`',
        f'- goal-state reconstruction max abs: '
        f'`{audit["integrity"]["max_goal_state_mismatch"]:.3e}`',
        f'- recomputed physical-cost max abs: '
        f'`{audit["integrity"]["max_physical_cost_mismatch"]:.3e}`',
        f'- recomputed success disagreements: '
        f'`{audit["integrity"]["success_disagreements"]}`',
        f'- exact render-visible terminal-state max abs: '
        f'`{audit["integrity"]["max_terminal_render_state_mismatch"]:.3e}`',
        '',
        '## State-blocked aggregate',
        '',
        '| metric | mean | paired-state 95% bootstrap CI |',
        '| --- | ---: | ---: |',
    ]
    keys = [
        ('learned update cosine', 'learned_update_cosine'),
        ('visual update cosine', 'visual_update_cosine'),
        ('visual - learned cosine', 'delta_update_cosine'),
        ('learned elite overlap', 'learned_elite_overlap'),
        ('visual elite overlap', 'visual_elite_overlap'),
        ('visual/physical Spearman', 'visual_physical_spearman'),
        ('visual true-cost gain', 'visual_true_cost_gain'),
        ('visual recovery fraction', 'visual_recovery_fraction'),
    ]
    for label, key in keys:
        interval = ci.get(key, [float('nan'), float('nan')])
        lines.append(
            f'| {label} | {format_number(summary[key])} | '
            f'[{format_number(interval[0])}, '
            f'{format_number(interval[1])}] |'
        )
    lines.extend(
        [
            '',
            'The recovery fraction uses physical elite cost and is computed '
            'as `(learned - visual) / (learned - physical oracle)`. Positive '
            'values mean that true-terminal visual geometry closes part of '
            'the learned-rollout selection gap.',
            '',
            '## By CEM step',
            '',
            '| step | learned cos | visual cos | learned overlap | visual '
            'overlap | visual/physical rho | true-cost gain | recovery |',
            '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for row in audit['per_step']:
        lines.append(
            f'| {row["step"]} '
            f'| {format_number(row["learned_update_cosine"])} '
            f'| {format_number(row["visual_update_cosine"])} '
            f'| {format_number(row["learned_elite_overlap"])} '
            f'| {format_number(row["visual_elite_overlap"])} '
            f'| {format_number(row["visual_physical_spearman"])} '
            f'| {format_number(row["visual_true_cost_gain"], 2)} '
            f'| {format_number(row["visual_recovery_fraction"])} |'
        )
    decision = 'PASS' if gate['passed'] else 'MISS'
    lines.extend(
        [
            '',
            '## Predeclared cell gate',
            '',
            f'**{decision}**: visual update cosine '
            f'`{summary["visual_update_cosine"]:.3f}` '
            f'(required `{gate["min_update_cosine"]:.3f}`), visual elite '
            f'overlap `{summary["visual_elite_overlap"]:.3f}` '
            f'(required `{gate["min_elite_overlap"]:.3f}`), and physical-gap '
            f'recovery `{summary["visual_recovery_fraction"]:.3f}` '
            f'(required `{gate["min_recovery_fraction"]:.3f}`).',
            '',
            'Passing this cell is not a method result: H5 and H8 must both '
            'pass, followed by a recursive proposal-resampling intervention. '
            'A miss means the raw LeWM terminal metric should not be used as '
            'the general oracle; it does not invalidate a task-aware physical '
            'teacher or a learned reachability-metric teacher.',
            '',
            f'Elapsed: `{audit["elapsed_seconds"] / 60:.1f}` minutes.',
            '',
        ]
    )
    del rows
    return '\n'.join(lines)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    started = time.time()
    semantics = cfg.get('semantics', {})
    source = Path(str(semantics.get('source', '')))
    output = Path(str(semantics.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'semantics.source does not exist: {source}')
    if output == Path('.'):
        raise ValueError('semantics.out is required')
    overwrite = bool(semantics.get('overwrite', False))
    json_path = output.with_suffix('.json')
    report_path = output.with_name(output.stem + '_report.md')
    existing = [
        path
        for path in (output, json_path, report_path)
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f'outputs already exist: {existing}; set semantics.overwrite=true'
        )

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    required = {'terminal_state', 'prev_mean', 'success'}
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f'source is missing fields {missing}')

    generators = result['generators'].astype(str).tolist()
    generator = str(semantics.get('generator', generators[0]))
    generator_i = select_named(
        result['generators'],
        generator,
        name='semantics.generator',
    )
    policy_name = str(semantics.get('policy', generator))
    scorer = str(semantics.get('scorer', policy_name))
    scorer_i = select_named(
        result['scorers'],
        scorer,
        name='semantics.scorer',
    )
    round_indices, steps = select_rounds(
        result['steps'],
        semantics.get('steps'),
    )
    max_states = int(semantics.get('max_states', 0))
    num_source_states = len(result['rows'])
    num_states = (
        min(max_states, num_source_states) if max_states > 0
        else num_source_states
    )
    state_indices = list(range(num_states))
    topk = int(semantics.get('topk', 30))
    batch_size = int(semantics.get('batch_size', 128))
    bootstrap = int(semantics.get('bootstrap', 20000))
    seed = int(semantics.get('seed', cfg.seed))
    if topk < 2 or batch_size < 1:
        raise ValueError('semantics.topk >= 2 and batch_size >= 1 required')

    candidates = result['candidates'][
        np.ix_(
            state_indices,
            [generator_i],
            round_indices,
        )
    ][:, 0].astype(np.float32)
    learned_cost = result['pred'][
        np.ix_(
            state_indices,
            [generator_i],
            round_indices,
            [scorer_i],
        )
    ][:, 0, :, 0].astype(np.float32)
    physical_cost = result['true'][
        np.ix_(state_indices, [generator_i], round_indices)
    ][:, 0].astype(np.float64)
    terminal_state = result['terminal_state'][
        np.ix_(state_indices, [generator_i], round_indices)
    ][:, 0].astype(np.float64)
    successes = result['success'][
        np.ix_(state_indices, [generator_i], round_indices)
    ][:, 0].astype(bool)
    prev_mean = result['prev_mean'][
        np.ix_(state_indices, [generator_i], round_indices)
    ][:, 0].astype(np.float32)

    model = swm.wm.utils.load_pretrained(policy_name).to('cuda').eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    transform = img_transform(cfg)

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = (
        int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) + 5
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    visual_cost = np.empty_like(physical_cost, dtype=np.float32)
    visual_embeddings_dim = None
    max_initial_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_cost_mismatch = 0.0
    max_render_state_mismatch = 0.0
    success_disagreements = 0
    try:
        for output_state_i, source_state_i in enumerate(state_indices):
            info, initial, goal = prepare_world_info(
                world,
                dataset,
                episode=int(result['episodes'][source_state_i]),
                start=int(result['starts'][source_state_i]),
                goal_offset=int(result['goal_offset']),
                callables=callables,
                history_len=1,
                action_block=int(cfg.plan_config.action_block),
            )
            max_initial_mismatch = max(
                max_initial_mismatch,
                float(
                    np.max(
                        np.abs(
                            initial - result['initial_state'][source_state_i]
                        )
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(
                        np.abs(goal - result['goal_state'][source_state_i])
                    )
                ),
            )
            goal_embedding = encode_frames(
                model,
                [np.asarray(info['goal'][0, 0])],
                transform=transform,
                batch_size=1,
            )[0]
            visual_embeddings_dim = int(len(goal_embedding))

            raw = world.envs.envs[0].unwrapped
            _set_goal_pose_from_state(raw, goal)
            if hasattr(raw, '_set_goal_state'):
                raw._set_goal_state(goal)

            for output_round_i, source_round_i in enumerate(round_indices):
                frames = []
                for candidate_i, state in enumerate(
                    terminal_state[output_state_i, output_round_i]
                ):
                    max_render_state_mismatch = max(
                        max_render_state_mismatch,
                        set_state_for_render(raw, state),
                    )
                    frames.append(np.asarray(raw.render()).copy())
                    recomputed, _, _, recomputed_success = task_cost(
                        goal,
                        state,
                    )
                    max_cost_mismatch = max(
                        max_cost_mismatch,
                        abs(
                            recomputed
                            - float(
                                physical_cost[
                                    output_state_i,
                                    output_round_i,
                                    candidate_i,
                                ]
                            )
                        ),
                    )
                    success_disagreements += int(
                        recomputed_success
                        != bool(
                            successes[
                                output_state_i,
                                output_round_i,
                                candidate_i,
                            ]
                        )
                    )
                terminal_embeddings = encode_frames(
                    model,
                    frames,
                    transform=transform,
                    batch_size=batch_size,
                )
                visual_cost[output_state_i, output_round_i] = np.sum(
                    np.square(terminal_embeddings - goal_embedding[None]),
                    axis=-1,
                )
                print(
                    f'[{output_state_i + 1}/{num_states}] '
                    f'step={steps[output_round_i]} '
                    f'encoded={len(frames)}',
                    flush=True,
                )
    finally:
        world.close()

    if max_initial_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            'trace reconstruction mismatch: '
            f'initial={max_initial_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    if (
        max_cost_mismatch > 1e-4
        or success_disagreements
        or max_render_state_mismatch > 1e-5
    ):
        raise RuntimeError(
            'terminal-state integrity mismatch: '
            f'cost={max_cost_mismatch:.3e}, '
            f'success_disagreements={success_disagreements}, '
            f'render_state={max_render_state_mismatch:.3e}'
        )

    rows = []
    learned_indices = np.empty(
        physical_cost.shape[:2] + (min(topk, physical_cost.shape[-1]),),
        dtype=np.int64,
    )
    visual_indices = np.empty_like(learned_indices)
    physical_indices_out = np.empty_like(learned_indices)
    for state_i in range(num_states):
        for round_i, step in enumerate(steps):
            population = candidates[state_i, round_i]
            physical = physical_cost[state_i, round_i]
            success = successes[state_i, round_i]
            physical_mean, physical_std, physical_indices = elite_moments(
                population,
                physical,
                topk=topk,
                std_floor=1e-6,
            )
            learned = update_metrics(
                learned_cost[state_i, round_i],
                candidates=population,
                origin=prev_mean[state_i, round_i],
                physical_cost=physical,
                physical_indices=physical_indices,
                physical_mean=physical_mean,
                topk=topk,
                success=success,
            )
            visual = update_metrics(
                visual_cost[state_i, round_i],
                candidates=population,
                origin=prev_mean[state_i, round_i],
                physical_cost=physical,
                physical_indices=physical_indices,
                physical_mean=physical_mean,
                topk=topk,
                success=success,
            )
            visual_learned_rank = rank_metrics(
                visual_cost[state_i, round_i],
                learned_cost[state_i, round_i],
            )
            learned_indices[state_i, round_i] = learned['indices']
            visual_indices[state_i, round_i] = visual['indices']
            physical_indices_out[state_i, round_i] = physical_indices
            rows.append(
                {
                    'state_index': state_i,
                    'source_state_index': state_indices[state_i],
                    'dataset_row': int(
                        result['rows'][state_indices[state_i]]
                    ),
                    'episode': int(
                        result['episodes'][state_indices[state_i]]
                    ),
                    'start': int(
                        result['starts'][state_indices[state_i]]
                    ),
                    'step': int(step),
                    'learned_update_cosine': learned['update_cosine'],
                    'visual_update_cosine': visual['update_cosine'],
                    'learned_relative_update_error': learned[
                        'relative_update_error'
                    ],
                    'visual_relative_update_error': visual[
                        'relative_update_error'
                    ],
                    'learned_elite_overlap': learned['elite_overlap'],
                    'visual_elite_overlap': visual['elite_overlap'],
                    'learned_physical_elite_cost': learned[
                        'physical_elite_cost'
                    ],
                    'visual_physical_elite_cost': visual[
                        'physical_elite_cost'
                    ],
                    'physical_elite_cost': float(
                        np.mean(physical[physical_indices])
                    ),
                    'learned_elite_success_fraction': learned[
                        'elite_success_fraction'
                    ],
                    'visual_elite_success_fraction': visual[
                        'elite_success_fraction'
                    ],
                    'physical_elite_success_fraction': float(
                        np.mean(success[physical_indices])
                    ),
                    'learned_physical_top1_cost': learned[
                        'physical_top1_cost'
                    ],
                    'visual_physical_top1_cost': visual[
                        'physical_top1_cost'
                    ],
                    'learned_top1_success': learned['top1_success'],
                    'visual_top1_success': visual['top1_success'],
                    'learned_physical_spearman': learned['spearman'],
                    'visual_physical_spearman': visual['spearman'],
                    'visual_learned_spearman': visual_learned_rank[
                        'spearman'
                    ],
                    'learned_physical_kendall': learned['kendall'],
                    'visual_physical_kendall': visual['kendall'],
                    'learned_physical_inversion': learned['inversion'],
                    'visual_physical_inversion': visual['inversion'],
                    'learned_physical_symkl': diagonal_symmetric_kl(
                        learned['mean'],
                        learned['std'],
                        physical_mean,
                        physical_std,
                    ),
                    'visual_physical_symkl': diagonal_symmetric_kl(
                        visual['mean'],
                        visual['std'],
                        physical_mean,
                        physical_std,
                    ),
                }
            )

    summary = bootstrap_state_summary(
        rows,
        bootstrap=bootstrap,
        seed=seed,
    )
    per_step = per_step_summary(rows)
    minimum_cosine = float(semantics.get('gate_update_cosine', 0.7))
    minimum_overlap = float(semantics.get('gate_elite_overlap', 0.5))
    minimum_recovery = float(
        semantics.get('gate_recovery_fraction', 0.5)
    )
    mean = summary['mean']
    gate_passed = bool(
        mean['visual_update_cosine'] >= minimum_cosine
        and mean['visual_elite_overlap'] >= minimum_overlap
        and mean['visual_recovery_fraction'] >= minimum_recovery
    )
    elapsed = time.time() - started
    audit = {
        'version': 2,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'output': str(output.resolve()),
        'policy': policy_name,
        'generator': generator,
        'scorer': scorer,
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'horizon': int(result['horizon']),
        'goal_offset': int(result['goal_offset']),
        'steps': steps,
        'num_states': num_states,
        'num_candidates': int(candidates.shape[2]),
        'topk': topk,
        'encoder_embedding_dim': visual_embeddings_dim,
        'integrity': {
            'max_initial_state_mismatch': max_initial_mismatch,
            'max_goal_state_mismatch': max_goal_mismatch,
            'max_physical_cost_mismatch': max_cost_mismatch,
            'success_disagreements': success_disagreements,
            'max_terminal_render_state_mismatch': (
                max_render_state_mismatch
            ),
        },
        'summary': summary,
        'per_step': per_step,
        'gate': {
            'min_update_cosine': minimum_cosine,
            'min_elite_overlap': minimum_overlap,
            'min_recovery_fraction': minimum_recovery,
            'passed': gate_passed,
        },
        'elapsed_seconds': elapsed,
    }

    atomic_savez(
        output,
        version=np.asarray(2, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        source_sha256=np.asarray(audit['source_sha256']),
        state_indices=np.asarray(state_indices, dtype=np.int64),
        steps=np.asarray(steps, dtype=np.int64),
        learned_cost=learned_cost,
        visual_cost=visual_cost,
        physical_cost=physical_cost,
        success=successes,
        learned_topk_indices=learned_indices,
        visual_topk_indices=visual_indices,
        physical_topk_indices=physical_indices_out,
    )
    atomic_write(json_path, json.dumps(audit, indent=2, sort_keys=True) + '\n')
    atomic_write(report_path, build_report(audit, rows))
    print(json.dumps(audit['summary'], indent=2, sort_keys=True), flush=True)
    print(
        f'cell gate={"PASS" if gate_passed else "MISS"} '
        f'cos={mean["visual_update_cosine"]:.3f} '
        f'overlap={mean["visual_elite_overlap"]:.3f} '
        f'recovery={mean["visual_recovery_fraction"]:.3f}',
        flush=True,
    )
    print(f'results -> {output}', flush=True)
    print(f'report -> {report_path}', flush=True)


if __name__ == '__main__':
    run()
