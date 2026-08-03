"""Cache observable one-step LeWM innovations at planner query states.

A static image and CEM population may alias states that require opposite
optimizer corrections.  Receding-horizon control also observes the model's
most recent prediction error.  This cache reconstructs one additional history
frame, predicts the current projected latent from the previous three frames
and intervening actions, and stores the signed innovation

    encoded_current - predicted_current.

The feature is deployable after the first real transition: it uses only past
observations, past actions, and the frozen world model, never simulator cost
or privileged state at inference.
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
def one_step_innovation(model, prepared: dict) -> tuple[np.ndarray, ...]:
    pixels = prepared['pixels']
    past_action = prepared.get('past_action')
    if pixels.ndim != 5 or pixels.shape[1] != 4:
        raise ValueError(
            f'innovation requires four frames, got {tuple(pixels.shape)}'
        )
    if past_action is None or past_action.shape[1] != 3:
        raise ValueError(
            'innovation requires three aligned past actions, got '
            f'{None if past_action is None else tuple(past_action.shape)}'
        )
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    encoded = model.encode(
        {'pixels': pixels.to(device=device, dtype=dtype)}
    )['emb']
    action_embedding = model.action_encoder(
        past_action.to(device=device, dtype=dtype)
    )
    predicted = model.predict(encoded[:, :3], action_embedding)[:, -1]
    actual = encoded[:, -1]
    innovation = actual - predicted
    return tuple(
        value.float().cpu().numpy()[0]
        for value in (innovation, predicted, actual)
    )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    innovation_cfg = cfg.get('innovation', {})
    source = Path(str(innovation_cfg.get('source', '')))
    output = Path(str(innovation_cfg.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'innovation.source does not exist: {source}')
    if output == Path('.'):
        raise ValueError('innovation.out is required')
    overwrite = bool(innovation_cfg.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'innovation cache exists: {output}; '
            'set innovation.overwrite=true'
        )
    policy_name = str(
        innovation_cfg.get('policy', 'pd_d192_k3_eval')
    )

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    required = {
        'rows',
        'episodes',
        'starts',
        'initial_state',
        'goal_state',
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f'source is missing fields {missing}')
    if int(cfg.plan_config.history_len) != 4:
        raise ValueError(
            'set +plan_config.history_len=4 to expose one held-back frame'
        )

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

    innovations = []
    predicted_rows = []
    actual_rows = []
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
                history_len=4,
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
            innovation, predicted, actual = one_step_innovation(
                model,
                prepare_model_info(policy, info),
            )
            innovations.append(innovation)
            predicted_rows.append(predicted)
            actual_rows.append(actual)
            print(
                f'[{state_i + 1}/{len(result["rows"])}] '
                f'innovation_norm={np.linalg.norm(innovation):.4f}',
                flush=True,
            )
    finally:
        world.close()

    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    innovation_array = np.asarray(innovations, dtype=np.float32)
    predicted_array = np.asarray(predicted_rows, dtype=np.float32)
    actual_array = np.asarray(actual_rows, dtype=np.float32)
    norms = np.linalg.norm(innovation_array, axis=1)
    audit = {
        'version': 1,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'policy': policy_name,
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'rows': int(len(result['rows'])),
        'feature_width': int(innovation_array.shape[1]),
        'mean_innovation_norm': float(np.mean(norms)),
        'median_innovation_norm': float(np.median(norms)),
        'max_innovation_norm': float(np.max(norms)),
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
    }
    atomic_savez(
        output,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=result['rows'].astype(np.int64),
        features=innovation_array,
        predicted=predicted_array,
        actual=actual_array,
    )
    print(f'innovation cache -> {output}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
