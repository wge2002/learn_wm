"""Phase 8i: behavioral validation of the commitment-anchor proposer in real CEM.

Same as eval_wm.py, but if SWM_COMMIT_PROPOSER_DIR is set it monkeypatches
model.get_cost to add the learned commitment sub-goal term:

    cost = (1-lam)*terminal_MSE(pred[:,:,-1], goal)
         +    lam *MSE(pred[:,:,mid], proposer(z_grounded, goal))

The proposer (Phase 8g) is selected per planning horizon: mid = H//2, delta=mid.
Run with lam=0 (or unset dir) for the baseline, lam>0 for the commitment policy,
and compare success_rate.
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

sys.path.insert(0, str(Path(__file__).resolve().parent))


def configure_torch_threads_from_env():
    raw = os.environ.get('SWM_TORCH_THREADS')
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        pass


def img_transform(cfg, dtype=torch.float32):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(dtype, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=cfg.eval.img_size),
    ])


def get_episodes_length(dataset, episodes):
    col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    return np.array([np.max(step_idx[episode_idx == ep]) + 1 for ep in episodes])


def to_container_or_none(value):
    return None if value is None else OmegaConf.to_container(value, resolve=True)


def maybe_patch_commitment(model, cfg):
    commit_dir = os.environ.get('SWM_COMMIT_PROPOSER_DIR')
    lam = float(os.environ.get('SWM_COMMIT_LAMBDA', '0'))
    if not commit_dir or lam <= 0:
        print('[commit] no commitment patch (baseline)')
        return
    from anchor_train_phase8g import Proposer
    tag = os.environ.get('SWM_COMMIT_TAG', 'continuous')
    H = cfg.plan_config.horizon
    mid = max(1, H // 2)
    ckpt = Path(commit_dir) / f'proposer_{tag}_d{mid}.pt'
    ck = torch.load(ckpt, map_location='cuda')
    prop = Proposer(dim=ck['dim'], codebook=ck.get('codebook', 256),
                    discrete=ck['discrete']).cuda().eval()
    prop.load_state_dict(ck['state'])
    orig = model.get_cost

    def patched(info_dict, candidates):
        base = orig(info_dict, candidates)            # mutates info_dict
        pred = info_dict['predicted_emb']             # (B,S,T,D)
        g = info_dict['goal_emb']
        g = g[:, -1] if g.dim() == 3 else g           # (B,D)
        z = pred[:, 0, 0].float()                     # grounded init latent (B,D)
        with torch.no_grad():
            w, _ = prop(z, g.float())                 # (B,D)
        midv = pred[:, :, mid].float()                # (B,S,D)
        commit = ((midv - w[:, None, :]) ** 2).sum(-1)  # (B,S)
        return (1 - lam) * base + lam * commit.to(base.dtype)

    model.get_cost = patched
    print(f'[commit] PATCHED get_cost: tag={tag} lam={lam} H={H} mid={mid} ckpt={ckpt.name}')


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    configure_torch_threads_from_env()
    assert cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {'pixels': img_transform(cfg, img_dtype), 'goal': img_transform(cfg, img_dtype)}

    dataset = swm.data.load_dataset(cfg.eval.dataset_name, cache_dir=cfg.get('cache_dir', None),
                                    keys_to_cache=list(cfg.dataset.keys_to_cache))
    col_name = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    ep_indices, _ = np.unique(dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        cd = dataset.get_col_data(col)
        processor.fit(cd[~np.isnan(cd).any(axis=1)])
        process[col] = processor
        if col != 'action':
            process[f'goal_{col}'] = process[col]

    model = swm.wm.utils.load_pretrained(cfg.policy).to('cuda').eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    maybe_patch_commitment(model, cfg)

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(solver=solver, config=config, process=process, transform=transform)

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start_idx_dict[ep] for ep in dataset.get_col_data(col_name)])
    valid_indices = np.nonzero(dataset.get_col_data('step_idx') <= max_start_per_row)[0]

    g = np.random.default_rng(cfg.seed)
    idx = np.sort(valid_indices[g.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)])
    eval_episodes = dataset.get_row_data(idx)[col_name]
    eval_start_idx = dataset.get_row_data(idx)['step_idx']

    world.set_policy(policy)
    start = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=cfg.get('bf16', False)):
        metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps, eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get('callables'), resolve=True),
            options=to_container_or_none(cfg.eval.get('reset_options')), video=None)
    print(f'[8i] H={cfg.plan_config.horizon} lam={os.environ.get("SWM_COMMIT_LAMBDA","0")} '
          f'tag={os.environ.get("SWM_COMMIT_TAG","-")} seed={cfg.seed} '
          f'metrics={metrics} time={time.time()-start:.0f}s')


if __name__ == '__main__':
    run()
