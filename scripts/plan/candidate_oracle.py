"""Candidate-rank oracle (Week-1 critical path, Gates A/B).

For N dataset start states: run ONE CEM planning call with CandidateRecorder,
stratify the final candidate set, execute each selected candidate open-loop in
a cloned env (same _set_state mechanism as eval), and compare predicted vs
true terminal cost: Spearman/Kendall, pairwise inversion, top-k precision,
simple regret. The same (state, goal, candidate-bank) triples should be reused
across checkpoints via --bank to give paired comparisons.

Usage (on the GPU box):
  python scripts/plan/candidate_oracle.py policy=iter2_multistep_eval \
      +oracle.num_states=40 +oracle.per_state=24 +oracle.out=outputs/week1/oracle_K5.npz \
      [+oracle.bank=outputs/week1/bank.npz] eval.goal_offset_steps=40 ...

SMOKE-VERIFY on first run: (1) action denormalization path matches
WorldModelPolicy's executed actions; (2) env reset-to-state reproducibility;
(3) true cost definition (block pose distance) sanity vs success flag.
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

import stable_worldmodel as swm
from stable_worldmodel.solver.callbacks.candidate_recorder import (
    CandidateRecorder,
)

# reuse eval_wm's setup helpers
from eval_wm import get_dataset, img_transform  # noqa: E402


def stratify(costs: np.ndarray, per_state: int, rng) -> np.ndarray:
    """Pick candidate indices: top, near-tie band, random, spread quantiles."""
    order = np.argsort(costs)
    k = per_state
    top = order[: k // 4]
    tie = order[k // 4 : k // 2]  # near-tie band right below top
    quant = order[np.linspace(0, len(order) - 1, k // 4, dtype=int)]
    rand = rng.choice(len(costs), size=k - len(top) - len(tie) - len(quant), replace=False)
    return np.unique(np.concatenate([top, tie, quant, rand]))


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    ocfg = cfg.get('oracle', {})
    num_states = int(ocfg.get('num_states', 40))
    per_state = int(ocfg.get('per_state', 24))
    out_path = str(ocfg.get('out', 'outputs/week1/oracle.npz'))
    bank_path = ocfg.get('bank', None)

    from sklearn import preprocessing

    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)

    transform = {
        'pixels': img_transform(cfg),
        'pixels_hist': img_transform(cfg),
        'goal': img_transform(cfg),
    }
    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == 'pixels':
            continue
        proc = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        proc.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = proc
        if col != 'action':
            process[f'goal_{col}'] = proc
    process['action_hist'] = process['action']

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

    # start states: same sampling rule as eval_wm (valid start rows)
    rng = np.random.default_rng(cfg.seed)
    col = 'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    step_idx = dataset.get_col_data('step_idx')
    # rows with enough remaining steps for goal_offset
    if bank_path and os.path.exists(str(bank_path)):
        bank = np.load(str(bank_path), allow_pickle=True)
        rows = bank['rows']
    else:
        valid = np.nonzero(step_idx < step_idx.max() - cfg.eval.goal_offset_steps)[0]
        rows = np.sort(rng.choice(valid, size=num_states, replace=False))

    results = []
    block = cfg.plan_config.action_block
    for row in rows:
        rec = dataset.get_row_data(int(row))
        state = rec['state']
        goal_row = dataset.get_row_data(int(row) + cfg.eval.goal_offset_steps)
        goal_state = goal_row['state']

        # one planning call from this state (env provides obs after set_state)
        obs, _ = world.reset_to(state=state, goal_state=goal_state)  # VERIFY api
        plan = policy.plan_once(obs)  # VERIFY: exposes solver outputs incl. callbacks
        cands = recorder.history[-1][-1]['candidates'][0].float().numpy()  # (N,H,D)
        costs = recorder.history[-1][-1]['costs'][0].numpy()

        sel = stratify(costs, per_state, rng)
        true_costs = np.zeros(len(sel))
        for j, ci in enumerate(sel):
            world.reset_to(state=state, goal_state=goal_state)  # VERIFY
            acts = cands[ci].reshape(-1, cands[ci].shape[-1] // block)  # (H*block, adim)
            acts = process['action'].inverse_transform(acts)
            done = False
            for a in acts:
                if done:
                    break
                obs2, _, term, trunc, info = world.step_env(a)  # VERIFY api
                done = term or trunc
            true_costs[j] = np.linalg.norm(
                world.get_state()[:5] - goal_state[:5]  # VERIFY: block+agent pose dims
            )
        results.append({'row': int(row), 'sel': sel, 'pred': costs[sel], 'true': true_costs})

    # metrics
    from scipy.stats import kendalltau, spearmanr

    sp, kt, inv, regret = [], [], [], []
    for r in results:
        sp.append(spearmanr(r['pred'], r['true']).statistic)
        kt.append(kendalltau(r['pred'], r['true']).statistic)
        p, t = r['pred'], r['true']
        n = len(p)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        inv.append(np.mean([(p[i] < p[j]) != (t[i] < t[j]) for i, j in pairs]))
        regret.append(t[np.argmin(p)] - t.min())
    np.savez(out_path, rows=rows,
             results=np.array(results, dtype=object),
             spearman=np.array(sp), kendall=np.array(kt),
             inversion=np.array(inv), regret=np.array(regret))
    print(f'{cfg.policy}: spearman={np.nanmean(sp):.3f} kendall={np.nanmean(kt):.3f} '
          f'inversion={np.nanmean(inv):.3f} regret={np.nanmean(regret):.3f} -> {out_path}')


if __name__ == '__main__':
    run()
