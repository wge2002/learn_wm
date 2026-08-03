"""Cache frozen LeWM dense patch tokens for planner-facing probes.

LeWM planning uses the SIGReg-trained projected CLS token.  The optimizer
audits suggest that the missing correction modes are largely spatial, so this
cache preserves the final-layer patch grid from the same frozen encoder for
the history and goal frames.  It is a read-only diagnostic cache: no encoder
parameter is changed and state/goal reconstruction is checked exactly.
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
def encode_dense(model, frames: torch.Tensor) -> np.ndarray:
    if frames.ndim != 5:
        raise ValueError(
            f'frames must be (B,T,C,H,W), got {tuple(frames.shape)}'
        )
    batch, time = frames.shape[:2]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    flat = frames.reshape(-1, *frames.shape[2:]).to(
        device=device,
        dtype=dtype,
    )
    output = model.encoder(
        flat,
        interpolate_pos_encoding=True,
    ).last_hidden_state
    patches = output[:, 1:].reshape(
        batch,
        time,
        output.shape[1] - 1,
        output.shape[2],
    )
    return patches.float().cpu().numpy()


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    dense = cfg.get('dense', {})
    source = Path(str(dense.get('source', '')))
    output = Path(str(dense.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'dense.source does not exist: {source}')
    if output == Path('.'):
        raise ValueError('dense.out is required')
    overwrite = bool(dense.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'dense cache exists: {output}; set dense.overwrite=true'
        )
    policy_name = str(dense.get('policy', 'pd_d192_k3_eval'))

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

    state_tokens = []
    goal_tokens = []
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
            prepared = prepare_model_info(policy, info)
            pixels = prepared['pixels']
            goal = prepared['goal']
            if goal.ndim == 4:
                goal = goal[:, None]
            state_tokens.append(encode_dense(model, pixels)[0])
            goal_tokens.append(encode_dense(model, goal)[0])
            print(
                f'[{state_i + 1}/{len(result["rows"])}] '
                f'state={state_tokens[-1].shape} '
                f'goal={goal_tokens[-1].shape}',
                flush=True,
            )
    finally:
        world.close()

    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )
    state_array = np.asarray(state_tokens, dtype=np.float16)
    goal_array = np.asarray(goal_tokens, dtype=np.float16)
    patch_count = int(state_array.shape[2])
    grid_size = int(round(patch_count**0.5))
    if grid_size * grid_size != patch_count:
        raise RuntimeError(f'patch count {patch_count} is not a square grid')
    audit = {
        'version': 1,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'policy': policy_name,
        'dataset': str(Path(str(cfg.eval.dataset_name)).resolve()),
        'rows': int(len(result['rows'])),
        'state_shape': list(state_array.shape[1:]),
        'goal_shape': list(goal_array.shape[1:]),
        'grid_size': grid_size,
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
    }
    atomic_savez(
        output,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=result['rows'].astype(np.int64),
        state_tokens=state_array,
        goal_tokens=goal_array,
    )
    print(f'dense token cache -> {output}', flush=True)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


if __name__ == '__main__':
    run()
