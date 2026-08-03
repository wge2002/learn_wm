"""Cache candidate-wise imagined terminal vectors for OE probes.

The ordinary LeWM scorer reduces every imagined terminal embedding to one
scalar squared distance from the goal.  That scalar discards the direction of
the predicted error.  This diagnostic preserves the frozen model's terminal
embedding for every candidate in an exact saved CEM population, so a
population operator can test whether vector-valued imagined outcomes expose
the otherwise ambiguous optimizer correction mode.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import stable_worldmodel as swm

from candidate_oracle import (
    make_process,
    prepare_model_info,
    prepare_world_info,
)
from eval_wm import get_dataset, img_transform
from oe_fixed_trace_train import cache_state_embeddings
from oe_update_resample import sha256, validate_source


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def rollout_terminal(
    model,
    cache: dict[str, torch.Tensor],
    candidates: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    if candidates.ndim != 3:
        raise ValueError(
            f'candidates must be (N,H,D), got {tuple(candidates.shape)}'
        )
    num_candidates = len(candidates)
    history = int(cache['history'])
    info = {
        'pixels': torch.empty(
            (1, num_candidates, history),
            device=candidates.device,
            dtype=candidates.dtype,
        ),
        'goal': torch.empty(
            (1, num_candidates, 1),
            device=candidates.device,
            dtype=candidates.dtype,
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
    cost = model.get_cost(info, candidates.unsqueeze(0))[0]
    terminal = info['predicted_emb'][0, :, -1]
    return (
        terminal.float().cpu().numpy(),
        cost.float().cpu().numpy(),
    )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    outcome = cfg.get('outcome', {})
    source = Path(str(outcome.get('source', '')))
    output = Path(str(outcome.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'outcome.source does not exist: {source}')
    if output == Path('.'):
        raise ValueError('outcome.out is required')
    overwrite = bool(outcome.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'candidate outcome cache exists: {output}; '
            'set outcome.overwrite=true'
        )
    policy_name = str(outcome.get('policy', 'pd_d192_k3_eval'))

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    required = {
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
        'candidates',
        'pred',
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f'source is missing fields {missing}')
    if result['candidates'].shape[1] != 1:
        raise ValueError('outcome cache currently requires one generator')
    scorer_names = result['scorers'].astype(str).tolist()
    scorer_i = (
        scorer_names.index(policy_name)
        if policy_name in scorer_names
        else None
    )
    candidates = result['candidates'][:, 0]
    stored_cost = (
        result['pred'][:, 0, :, scorer_i]
        if scorer_i is not None
        else None
    )
    num_states, num_rounds, population = candidates.shape[:3]
    action_shape = tuple(candidates.shape[-2:])

    device = torch.device('cuda')
    model = swm.wm.utils.load_pretrained(policy_name).to(device).eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    model_dtype = next(model.parameters()).dtype

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

    terminal_rows = []
    recomputed_cost_rows = []
    goal_rows = []
    cost_mae_rows = []
    topk_overlap_rows = []
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_cost_mismatch = 0.0
    try:
        for state_i in range(num_states):
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
                            initial_state
                            - result['initial_state'][state_i]
                        )
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(
                        np.abs(goal_state - result['goal_state'][state_i])
                    )
                ),
            )
            cache = cache_state_embeddings(
                model,
                prepare_model_info(policy, info),
                action_shape=action_shape,
            )
            goal_rows.append(
                cache['goal_emb'][0, -1].float().cpu().numpy()
            )
            state_terminals = []
            state_costs = []
            for round_i in range(num_rounds):
                terminal, cost = rollout_terminal(
                    model,
                    cache,
                    torch.as_tensor(
                        candidates[state_i, round_i],
                        device=device,
                        dtype=model_dtype,
                    ),
                )
                state_terminals.append(terminal)
                state_costs.append(cost)
                if stored_cost is not None:
                    difference = np.abs(
                        cost - stored_cost[state_i, round_i]
                    )
                    cost_mae_rows.append(float(np.mean(difference)))
                    stored_top = set(
                        np.argsort(
                            stored_cost[state_i, round_i],
                            kind='stable',
                        )[:30].tolist()
                    )
                    recomputed_top = set(
                        np.argsort(cost, kind='stable')[:30].tolist()
                    )
                    topk_overlap_rows.append(
                        len(stored_top & recomputed_top) / 30
                    )
                    max_cost_mismatch = max(
                        max_cost_mismatch,
                        float(np.max(difference)),
                    )
            terminal_rows.append(np.asarray(state_terminals))
            recomputed_cost_rows.append(np.asarray(state_costs))
            cost_status = (
                f'max_cost_diff={max_cost_mismatch:.3e}'
                if stored_cost is not None
                else 'external_scorer'
            )
            print(
                f'[{state_i + 1}/{num_states}] '
                f'rounds={num_rounds} population={population} '
                f'{cost_status}',
                flush=True,
            )
    finally:
        world.close()

    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    terminal_array = np.asarray(terminal_rows, dtype=np.float16)
    recomputed_cost_array = np.asarray(
        recomputed_cost_rows,
        dtype=np.float32,
    )
    goal_array = np.asarray(goal_rows, dtype=np.float16)
    audit = {
        'version': 1,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'policy': policy_name,
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'rows': num_states,
        'rounds': num_rounds,
        'population': population,
        'terminal_shape': list(terminal_array.shape[1:]),
        'goal_shape': list(goal_array.shape[1:]),
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
        'max_cost_mismatch': (
            max_cost_mismatch if stored_cost is not None else None
        ),
        'mean_cost_mae': (
            float(np.mean(cost_mae_rows))
            if cost_mae_rows
            else None
        ),
        'mean_top30_overlap': (
            float(np.mean(topk_overlap_rows))
            if topk_overlap_rows
            else None
        ),
        'source_has_matching_scorer': scorer_i is not None,
    }
    atomic_savez(
        output,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=result['rows'].astype(np.int64),
        terminal_embeddings=terminal_array,
        goal_embeddings=goal_array,
        recomputed_cost=recomputed_cost_array,
    )
    print(f'candidate outcome cache -> {output}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
