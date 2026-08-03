"""Build a frozen-latent dynamics replay cache for OE fine-tuning.

The fixed-trace OE feasibility runs only supervise planner-query populations.
That is intentionally useful as a causal probe, but it is not the proposed
OE-WM training objective: the original world-model prediction objective must
remain active.  This script reconstructs ordinary PushT state/action windows
from a state-only dataset, renders them through the live environment, and
caches encoder embeddings from the base checkpoint.

Because the image encoder and projector remain frozen during the bridge
experiment, replaying these embeddings is equivalent to replaying the
prediction portion of the original LeWM objective while being substantially
cheaper than decoding and encoding images in every cross-fit run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig
import torch

import stable_worldmodel as swm
from stable_worldmodel.data import get_cache_dir

from candidate_oracle import make_process
from eval_wm import get_dataset, img_transform
from oe_update_resample import sha256


def comma_paths(value) -> list[Path]:
    if value is None:
        return []
    return [
        Path(item.strip())
        for item in str(value).split(',')
        if item.strip()
    ]


def exclusion_rows(paths: list[Path]) -> tuple[np.ndarray, dict[str, str]]:
    rows = []
    hashes = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f'exclusion source does not exist: {path}')
        with np.load(path, allow_pickle=False) as archive:
            if 'rows' not in archive:
                raise ValueError(f'exclusion source has no rows: {path}')
            rows.append(np.asarray(archive['rows'], dtype=np.int64))
        hashes[str(path.resolve())] = sha256(path)
    if not rows:
        return np.empty(0, dtype=np.int64), hashes
    return np.unique(np.concatenate(rows)), hashes


def sample_rows(
    dataset,
    *,
    num_windows: int,
    history_size: int,
    action_block: int,
    goal_offset: int,
    excluded: np.ndarray,
    exclusion_radius: int,
    rng: np.random.Generator,
) -> np.ndarray:
    episode_idx = np.asarray(dataset.get_col_data('episode_idx'))
    step_idx = np.asarray(dataset.get_col_data('step_idx'))
    episodes, inverse = np.unique(episode_idx, return_inverse=True)
    del episodes
    max_step = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
    np.maximum.at(max_step, inverse, step_idx)
    required_future = max(history_size * action_block, goal_offset)
    valid = np.flatnonzero(
        step_idx <= max_step[inverse] - required_future
    )

    if len(excluded):
        keep = np.ones(len(valid), dtype=bool)
        for row in excluded:
            keep &= np.abs(valid - int(row)) > exclusion_radius
        valid = valid[keep]
    if len(valid) < num_windows:
        raise ValueError(
            f'only {len(valid)} replay starts after exclusions, '
            f'but replay.num_windows={num_windows}'
        )
    return rng.choice(valid, size=num_windows, replace=False).astype(
        np.int64
    )


@torch.inference_mode()
def encode_images(
    model,
    images: list[np.ndarray],
    *,
    batch_size: int,
    num_frames: int,
    transform,
) -> np.ndarray:
    if len(images) % num_frames:
        raise ValueError('image count must be divisible by num_frames')
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    encoded = []
    frame_batch = batch_size * num_frames
    for offset in range(0, len(images), frame_batch):
        transformed = torch.stack(
            [
                transform(image)
                for image in images[offset : offset + frame_batch]
            ]
        )
        transformed = transformed.to(device=device, dtype=dtype)
        info = {'pixels': transformed.unsqueeze(1)}
        encoded.append(model.encode(info)['emb'][:, 0].float().cpu().numpy())
    flat = np.concatenate(encoded, axis=0)
    return flat.reshape(-1, num_frames, flat.shape[-1])


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
    replay = cfg.get('replay', {})
    output_value = str(replay.get('out', '')).strip()
    if not output_value:
        raise ValueError('replay.out is required')
    output = Path(output_value)
    overwrite = bool(replay.get('overwrite', False))
    if output.exists() and not overwrite:
        raise FileExistsError(
            f'replay cache already exists: {output}; '
            'set replay.overwrite=true to replace it'
        )

    base_policy = str(replay.get('policy', 'pd_d192_k3_eval'))
    num_windows = int(replay.get('num_windows', 2048))
    history_size = int(
        replay.get('history_size', cfg.plan_config.get('history_len', 3))
    )
    action_block = int(
        replay.get('action_block', cfg.plan_config.action_block)
    )
    goal_offset = int(replay.get('goal_offset', 60))
    exclusion_radius = int(
        replay.get('exclusion_radius', goal_offset)
    )
    batch_size = int(replay.get('batch_size', 64))
    validation_fraction = float(replay.get('validation_fraction', 0.2))
    seed = int(replay.get('seed', cfg.seed))
    if num_windows < 2 or history_size < 1 or action_block < 1:
        raise ValueError(
            'num_windows >= 2, history_size >= 1, and action_block >= 1 '
            'are required'
        )
    if goal_offset < history_size * action_block:
        raise ValueError(
            'goal_offset must cover the complete prediction replay window'
        )
    if exclusion_radius < 0 or batch_size < 1:
        raise ValueError(
            'exclusion_radius must be non-negative and batch_size positive'
        )
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be inside (0, 1)')

    exclude_paths = comma_paths(replay.get('exclude_sources'))
    excluded, source_hashes = exclusion_rows(exclude_paths)
    dataset_path = Path(str(cfg.eval.dataset_name))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = make_process(dataset, cfg.dataset.keys_to_cache)
    rng = np.random.default_rng(seed)
    rows = sample_rows(
        dataset,
        num_windows=num_windows,
        history_size=history_size,
        action_block=action_block,
        goal_offset=goal_offset,
        excluded=excluded,
        exclusion_radius=exclusion_radius,
        rng=rng,
    )

    device = torch.device('cuda')
    model = swm.wm.utils.load_pretrained(base_policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    model_history = int(getattr(model.predictor, 'num_frames', history_size))
    if model_history != history_size:
        raise ValueError(
            f'cache history_size={history_size} does not match '
            f'{base_policy} predictor.num_frames={model_history}'
        )
    checkpoint_path, _ = swm.wm.utils._resolve(
        base_policy,
        get_cache_dir(sub_folder='checkpoints'),
    )

    episode_idx = np.asarray(dataset.get_col_data('episode_idx'))
    step_idx = np.asarray(dataset.get_col_data('step_idx'))
    states = np.asarray(dataset.get_col_data('state'))
    raw_actions = np.asarray(dataset.get_col_data('action'))
    normalized_actions = process['action'].transform(
        np.nan_to_num(raw_actions, nan=0.0)
    ).astype(np.float32)

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = goal_offset + 5
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.reset(seed=seed)
    raw = world.envs.envs[0].unwrapped
    transform = img_transform(cfg)
    num_frames = history_size + 1
    embedding_chunks = []
    action_chunks = []
    render_batch = int(replay.get('render_batch', batch_size))
    if render_batch < 1:
        raise ValueError('replay.render_batch must be positive')

    started = time.time()
    try:
        for offset in range(0, num_windows, render_batch):
            batch_rows = rows[offset : offset + render_batch]
            images: list[np.ndarray] = []
            batch_actions = []
            for row in batch_rows:
                goal_state = states[int(row) + goal_offset]
                if hasattr(raw, '_set_goal_state'):
                    raw._set_goal_state(goal_state)
                if hasattr(raw, 'goal_pose'):
                    raw.goal_pose = np.asarray(
                        [goal_state[2], goal_state[3], goal_state[4]]
                    )
                for frame in range(num_frames):
                    raw._set_state(
                        states[int(row) + frame * action_block]
                    )
                    images.append(np.asarray(raw.render()).copy())
                action_end = int(row) + history_size * action_block
                batch_actions.append(
                    normalized_actions[int(row) : action_end].reshape(
                        history_size,
                        -1,
                    )
                )
            embedding_chunks.append(
                encode_images(
                    model,
                    images,
                    batch_size=batch_size,
                    num_frames=num_frames,
                    transform=transform,
                )
            )
            action_chunks.append(np.stack(batch_actions))
            done = min(offset + render_batch, num_windows)
            print(
                f'[{done}/{num_windows}] replay windows '
                f'elapsed={(time.time() - started) / 60:.1f}m',
                flush=True,
            )
    finally:
        world.close()

    embeddings = np.concatenate(embedding_chunks).astype(np.float16)
    actions = np.concatenate(action_chunks).astype(np.float16)
    permutation = rng.permutation(num_windows)
    num_validation = max(
        1,
        min(
            num_windows - 1,
            int(round(num_windows * validation_fraction)),
        ),
    )
    validation_indices = np.sort(permutation[:num_validation])
    train_indices = np.sort(permutation[num_validation:])
    audit = {
        'version': 1,
        'base_policy': base_policy,
        'checkpoint_path': str(checkpoint_path.resolve()),
        'checkpoint_sha256': sha256(checkpoint_path),
        'dataset': str(dataset_path.resolve()),
        'dataset_sha256': sha256(dataset_path),
        'exclude_sources': source_hashes,
        'excluded_rows': int(len(excluded)),
        'exclusion_radius': exclusion_radius,
        'seed': seed,
        'num_windows': num_windows,
        'history_size': history_size,
        'action_block': action_block,
        'goal_offset': goal_offset,
        'validation_fraction': validation_fraction,
        'embedding_dim': int(embeddings.shape[-1]),
        'max_source_row_overlap': int(
            np.isin(rows, excluded).sum()
        ),
        'elapsed_seconds': time.time() - started,
    }
    atomic_savez(
        output,
        version=np.asarray(1, dtype=np.int64),
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        rows=rows,
        episodes=episode_idx[rows].astype(np.int64),
        starts=step_idx[rows].astype(np.int64),
        embeddings=embeddings,
        actions=actions,
        train_indices=train_indices.astype(np.int64),
        validation_indices=validation_indices.astype(np.int64),
    )
    print(f'replay cache -> {output}', flush=True)
    print(
        f'shape embeddings={embeddings.shape} actions={actions.shape} '
        f'train={len(train_indices)} validation={len(validation_indices)}',
        flush=True,
    )
    print(f'elapsed={(time.time() - started) / 60:.1f} minutes', flush=True)


if __name__ == '__main__':
    run()
