"""Cache frozen LeWM state/goal/history features for OE corrector probes.

The output is deliberately checkpoint-frozen and state-aligned with a saved
CEM population trace.  It allows a small optimizer-update head to be tested
without fine-tuning or repeatedly invoking the image encoder.
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


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    latent = cfg.get('latent', {})
    source = Path(str(latent.get('source', '')))
    output = Path(str(latent.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'latent.source does not exist: {source}')
    if not str(output):
        raise ValueError('latent.out is required')
    overwrite = bool(latent.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'latent cache exists: {output}; set latent.overwrite=true'
        )
    policy_name = str(latent.get('policy', 'pd_d192_k3_eval'))

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
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f'source is missing fields {missing}')

    device = torch.device('cuda')
    model = swm.wm.utils.load_pretrained(policy_name).to(device).eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)

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

    action_shape = tuple(result['candidates'].shape[-2:])
    features = []
    component_shapes = None
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    try:
        for state_i in range(len(result['rows'])):
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
            components = [
                cache['emb'].float().cpu().numpy().reshape(-1),
                cache['goal_emb'].float().cpu().numpy().reshape(-1),
                cache['past_action'].float().cpu().numpy().reshape(-1),
            ]
            shapes = {
                key: list(cache[key].shape)
                for key in ('emb', 'goal_emb', 'past_action')
            }
            if component_shapes is None:
                component_shapes = shapes
            elif component_shapes != shapes:
                raise RuntimeError(
                    f'latent component shape changed: '
                    f'{component_shapes} vs {shapes}'
                )
            features.append(np.concatenate(components))
            print(
                f'[{state_i + 1}/{len(result["rows"])}] '
                f'latent width={len(features[-1])}',
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
        'policy': policy_name,
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'rows': int(len(result['rows'])),
        'feature_width': int(len(features[0])),
        'component_shapes': component_shapes,
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
    }
    atomic_savez(
        output,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=result['rows'].astype(np.int64),
        features=np.asarray(features, dtype=np.float32),
    )
    print(f'latent cache -> {output}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
