"""Add cross-model scorer predictions to an existing CEM population trace.

No candidates are regenerated and no simulator rollout is repeated.  The
script reconstructs each recorded first-plan state, caches embeddings for the
requested frozen checkpoints, and scores the exact stored populations.  The
result is used to test model-disagreement features as a discrete OE
correction-mode routing signal.
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
from oe_fixed_trace_train import cache_state_embeddings, score_cached
from oe_update_resample import sha256, validate_source


def comma_list(value) -> list[str]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError('augment.scorers must contain at least one policy')
    return items


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    augment = cfg.get('augment', {})
    source = Path(str(augment.get('source', '')))
    output = Path(str(augment.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'augment.source does not exist: {source}')
    if output == Path('.'):
        raise ValueError('augment.out is required')
    overwrite = bool(augment.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'augmented trace exists: {output}; set augment.overwrite=true'
        )
    scorers = comma_list(
        augment.get(
            'scorers',
            'pd_d192_k3_eval,iter2_multistep_eval,pd_d192_k10_eval',
        )
    )
    generator_name = str(augment.get('generator_name', '')).strip() or None

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    if result['candidates'].shape[1] != 1:
        raise ValueError('augmentation currently requires one generator')
    original_generators = result['generators'].astype(str).tolist()
    existing_scorers = result['scorers'].astype(str).tolist()
    existing_pred = result['pred']
    candidates = result['candidates'][:, 0].astype(np.float32)
    num_states, num_rounds = candidates.shape[:2]
    action_shape = tuple(candidates.shape[-2:])

    device = torch.device('cuda')
    models = {}
    for scorer in scorers:
        model = swm.wm.utils.load_pretrained(scorer).to(device).eval()
        model.interpolate_pos_encoding = True
        model.requires_grad_(False)
        models[scorer] = model

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
    solver = hydra.utils.instantiate(
        cfg.solver,
        model=models[scorers[0]],
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

    predicted = np.empty(
        (
            num_states,
            1,
            num_rounds,
            len(scorers),
            candidates.shape[2],
        ),
        dtype=np.float32,
    )
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    base_mae = []
    base_topk_overlap = []
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
            model_info = prepare_model_info(policy, info)
            for scorer_i, scorer in enumerate(scorers):
                if scorer in existing_scorers:
                    stored_i = existing_scorers.index(scorer)
                    values = existing_pred[
                        state_i,
                        0,
                        :,
                        stored_i,
                    ].astype(np.float32)
                    predicted[state_i, 0, :, scorer_i] = values
                    continue
                cache = cache_state_embeddings(
                    models[scorer],
                    model_info,
                    action_shape=action_shape,
                )
                for round_i in range(num_rounds):
                    values = (
                        score_cached(
                            models[scorer],
                            cache,
                            torch.as_tensor(
                                candidates[state_i, round_i],
                                device=device,
                                dtype=next(
                                    models[scorer].parameters()
                                ).dtype,
                            ),
                        )
                        .cpu()
                        .numpy()
                    )
                    predicted[
                        state_i,
                        0,
                        round_i,
                        scorer_i,
                    ] = values
            if scorers[0] in existing_scorers:
                old_i = existing_scorers.index(scorers[0])
                old = existing_pred[state_i, 0, :, old_i]
                new = predicted[state_i, 0, :, 0]
                base_mae.append(float(np.mean(np.abs(old - new))))
                for round_i in range(num_rounds):
                    old_top = set(
                        np.argsort(old[round_i], kind='stable')[:30].tolist()
                    )
                    new_top = set(
                        np.argsort(new[round_i], kind='stable')[:30].tolist()
                    )
                    base_topk_overlap.append(
                        len(old_top & new_top) / 30
                    )
            print(
                f'[{state_i + 1}/{num_states}] '
                f'scorers={len(scorers)}',
                flush=True,
            )
    finally:
        world.close()

    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    audit = {
        'version': 1,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'scorers': scorers,
        'original_generators': original_generators,
        'generator_name': generator_name,
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
        'base_score_mae': (
            float(np.mean(base_mae)) if base_mae else None
        ),
        'base_topk_overlap': (
            float(np.mean(base_topk_overlap))
            if base_topk_overlap
            else None
        ),
    }
    output_arrays = dict(result)
    if generator_name is not None:
        output_arrays['generators'] = np.asarray([generator_name])
    output_arrays['scorers'] = np.asarray(scorers)
    output_arrays['pred'] = predicted
    output_arrays['augmentation_audit'] = np.asarray(
        json.dumps(audit, sort_keys=True)
    )
    atomic_savez(output, output_arrays)
    print(f'augmented trace -> {output}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
