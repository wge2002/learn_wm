"""Stage 1 core: does a discrete commitment beat continuous regression at a
multimodal junction (the cleanest possible test of the idea)?

Three parts on the 2-door TwoRoom data:

A. multimodality quantification: near the y~112 ridge, cross-room expert actions
   are bimodal (head to up-door vs down-door); the conditional MEAN action_y ~ 0
   (= what an L2 regressor predicts) points straight into the wall.

B. learn two policies from the SAME (agent, goal) input:
   - continuous: MLP -> action (2D), MSE  (averages the two branches => blur)
   - discrete-anchor: MLP -> categorical door choice (commit) then head to that
     door center, then to goal  (= discrete anchor + continuous execution)

C. roll both in the env from cross-room ridge starts; compare success_rate.
   Expert = oracle upper bound.

Low-dim, fast, fully controlled. This is the minimal proof that discreteness pays
where futures are multimodal (the regime PushT lacked).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_worldmodel as swm  # noqa: F401

torch.set_num_threads(1)  # tiny env ops: avoid catastrophic thread oversubscription

WALL_X = 112.0
DOORS = np.array([[112.0, 56.0], [112.0, 168.0]], dtype=np.float32)  # (x,y) centers
VV = {"door.number": 2, "door.position": [56, 168, 49], "door.size": [18, 18, 14],
      "wall.axis": 1, "wall.thickness": 10}


def door_label(agent, goal):
    """Which door the (closest-door) expert would use for a cross-room pair."""
    d0 = np.linalg.norm(agent - DOORS[0], axis=-1)
    d1 = np.linalg.norm(agent - DOORS[1], axis=-1)
    return (d1 < d0).astype(np.int64)  # 0=up(y56), 1=down(y168)


def is_cross(agent_x, goal_x):
    return (agent_x - WALL_X) * (goal_x - WALL_X) < 0


class MLP(nn.Module):
    def __init__(self, out, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def train(model, X, Y, loss_fn, epochs=60, bs=2048, lr=1e-3, device="cpu"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.tensor(X, device=device); Y = torch.tensor(Y, device=device)
    n = X.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s+bs]
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/tworoom_stage1/tworoom_2door.npz")
    ap.add_argument("--output-dir", default="outputs/tworoom_stage1")
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    d = np.load(args.data)
    agent, goal, action = d["agent"], d["goal"], d["action"]
    X = np.concatenate([agent, goal], 1).astype(np.float32)

    # ---- A. multimodality near the ridge ----
    cross = is_cross(agent[:, 0], goal[:, 0])
    ridge = cross & (np.abs(agent[:, 1] - WALL_X) < 22) & (np.abs(agent[:, 0] - WALL_X) < 45)
    ay = action[ridge, 1]
    summary = {"ridge_n": int(ridge.sum())}
    if ridge.sum() > 50:
        from sklearn.mixture import GaussianMixture
        a1 = GaussianMixture(1).fit(ay[:, None]); a2 = GaussianMixture(2).fit(ay[:, None])
        summary["ridge_action_y_mean"] = float(ay.mean())          # ~0 => continuous blur target
        summary["ridge_action_y_absmean"] = float(np.abs(ay).mean())  # actual magnitude
        summary["ridge_frac_up"] = float((ay < 0).mean())
        summary["ridge_frac_down"] = float((ay > 0).mean())
        summary["bic_1comp"] = float(a1.bic(ay[:, None]))
        summary["bic_2comp"] = float(a2.bic(ay[:, None]))
        summary["bimodal_2comp_better"] = bool(a2.bic(ay[:, None]) < a1.bic(ay[:, None]))

    # ---- B. train policies ----
    dev = args.device
    cont = train(MLP(2).to(dev), X, action, nn.MSELoss(), device=dev)
    # discrete door classifier on cross-room steps (label = closest door)
    lab = door_label(agent, goal)
    Xc, yc = X[cross], lab[cross]
    clf = train(MLP(2).to(dev), Xc, yc, lambda o, t: F.cross_entropy(o, t),
                epochs=60, device=dev) if cross.sum() > 100 else None
    # continuous policy's predicted action_y at ridge (show the blur)
    with torch.no_grad():
        pred_ridge = cont(torch.tensor(X[ridge], device=dev)).cpu().numpy()
    summary["continuous_pred_action_y_absmean_at_ridge"] = float(np.abs(pred_ridge[:, 1]).mean())

    # ---- C. rollout success from cross-room ridge starts ----
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/TwoRoom-v1")

    def reset(a_xy, t_xy):
        o, _ = env.reset(seed=int(rng.integers(1 << 30)),
                         options={"variation_values": VV, "state": a_xy, "target_state": t_xy})
        return np.asarray(o, np.float32)

    def run(policy):
        succ = 0
        for _ in range(args.eval_episodes):
            ax = rng.uniform(20, WALL_X - 20); ay_ = rng.uniform(90, 134)
            tx = rng.uniform(WALL_X + 20, 200); ty = rng.uniform(20, 200)
            if rng.random() < 0.5:
                ax, tx = tx, ax  # right->left too
            obs = reset([float(ax), float(ay_)], [float(tx), float(ty)])
            g = obs[2:4].copy()
            for t in range(args.max_steps):
                a = policy(obs[:2], g)
                obs, r, term, trunc, info = env.step(a)
                obs = np.asarray(obs, np.float32)
                if term:
                    succ += 1; break
        return 100.0 * succ / args.eval_episodes

    def cont_policy(ag, g):
        x = torch.tensor(np.concatenate([ag, g])[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            a = cont(x).cpu().numpy()[0]
        n = np.linalg.norm(a)
        return (a / n if n > 1e-6 else a).astype(np.float32)

    def door_waypoint(ag, g, door_center, tol=12.0):
        # head to door CENTER until within tol (aligned with opening), then to
        # goal (which pulls the agent through the opening); once crossed -> goal.
        crossed = (ag[0] - WALL_X) * (g[0] - WALL_X) > 0
        if crossed or np.linalg.norm(ag - door_center) < tol:
            return g
        return door_center

    def to_action(ag, wp):
        dirv = wp - ag; n = np.linalg.norm(dirv)
        return (dirv / n if n > 1e-6 else dirv).astype(np.float32)

    def anchor_policy(ag, g):
        x = torch.tensor(np.concatenate([ag, g])[None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            door = int(clf(x).argmax(-1).item())
        return to_action(ag, door_waypoint(ag, g, DOORS[door]))

    def expert_policy(ag, g):
        door = door_label(ag[None], g[None])[0]
        return to_action(ag, door_waypoint(ag, g, DOORS[door]))

    summary["success_continuous"] = run(cont_policy)
    summary["success_anchor_discrete"] = run(anchor_policy) if clf is not None else None
    summary["success_expert_oracle"] = run(expert_policy)

    (out / "tworoom_stage1_summary.json").write_text(json.dumps(summary, indent=2))
    print("=== Stage 1: TwoRoom 2-door, multimodal junction ===")
    print(f"A. ridge multimodality (n={summary['ridge_n']}):")
    print(f"   actual action_y: |mean|={summary.get('ridge_action_y_absmean',0):.2f} "
          f"(up {summary.get('ridge_frac_up',0):.2f} / down {summary.get('ridge_frac_down',0):.2f})  "
          f"=> bimodal_2comp_better={summary.get('bimodal_2comp_better')}")
    print(f"   continuous-regression target mean action_y={summary.get('ridge_action_y_mean',0):+.2f} "
          f"(~0 = blur into wall); continuous MLP |pred action_y| at ridge="
          f"{summary['continuous_pred_action_y_absmean_at_ridge']:.2f}")
    print("C. rollout success_rate from cross-room ridge starts:")
    print(f"   continuous MLP   : {summary['success_continuous']:.1f}%")
    print(f"   discrete-anchor  : {summary['success_anchor_discrete']}%")
    print(f"   expert (oracle)  : {summary['success_expert_oracle']:.1f}%")
    print(f"[stage1] wrote {out}")


if __name__ == "__main__":
    main()
