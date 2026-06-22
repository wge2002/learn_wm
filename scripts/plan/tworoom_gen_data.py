"""Stage 1 data generation: TwoRoom expert trajectories with a 2-door wall.

Produces a dataset whose futures are genuinely MULTIMODAL: with two doors (y=56,
y=168) on a vertical wall, the expert routes through the closest door, so near the
y~=112 ridge a tiny change in agent position flips the whole trajectory (up vs
down door). Aggregated over random starts, the data contains both branches.

Saves per-step: agent_xy, goal_xy, action, episode_idx, step_idx (+ door config).
Images are a deterministic function of state, so we re-render later for the WM.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import stable_worldmodel as swm  # noqa: F401  (registers envs)
from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy


DOOR_POS = [56, 168, 49]
DOOR_SIZE = [18, 18, 14]
VV = {"door.number": 2, "door.position": DOOR_POS, "door.size": DOOR_SIZE,
      "wall.axis": 1, "wall.thickness": 10}
LO, HI = 16.0, 207.0  # valid position range (inside border)
WALL_X = 112.0


def sample_pos(rng, side=None):
    x = rng.uniform(LO, HI)
    if side == "left":
        x = rng.uniform(LO, WALL_X - 18)
    elif side == "right":
        x = rng.uniform(WALL_X + 18, HI)
    y = rng.uniform(LO, HI)
    return [float(x), float(y)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/tworoom_stage1/tworoom_2door.npz")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--action-noise", type=float, default=0.05)
    ap.add_argument("--cross-room-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = gym.make("swm/TwoRoom-v1")
    pol = ExpertPolicy(action_noise=args.action_noise, seed=args.seed)
    pol.set_env(env)

    A, G, ACT, EP, ST = [], [], [], [], []
    ep = 0
    for e in range(args.episodes):
        if rng.random() < args.cross_room_frac:
            agent = sample_pos(rng, "left"); target = sample_pos(rng, "right")
            if rng.random() < 0.5:  # also right->left
                agent, target = target, agent
        else:
            agent = sample_pos(rng); target = sample_pos(rng)
        obs, info = env.reset(seed=int(rng.integers(1 << 30)),
                              options={"variation_values": VV,
                                       "state": agent, "target_state": target})
        obs = np.asarray(obs, dtype=np.float32)
        goal = obs[2:4].copy()
        for t in range(args.max_steps):
            a = pol.get_action({"state": obs[:2], "goal_state": goal})
            a = np.asarray(a, dtype=np.float32).squeeze()
            A.append(obs[:2].copy()); G.append(goal.copy()); ACT.append(a.copy())
            EP.append(ep); ST.append(t)
            obs, r, term, trunc, info = env.step(a)
            obs = np.asarray(obs, dtype=np.float32)
            if term:
                break
        ep += 1
        if (e + 1) % 500 == 0:
            print(f"[gen] {e+1}/{args.episodes} episodes, {len(A)} steps", flush=True)

    np.savez_compressed(
        out,
        agent=np.array(A, np.float32), goal=np.array(G, np.float32),
        action=np.array(ACT, np.float32),
        episode_idx=np.array(EP, np.int64), step_idx=np.array(ST, np.int64),
        door_pos=np.array(DOOR_POS[:2], np.float32),
        door_size=np.array(DOOR_SIZE[:2], np.float32), wall_x=np.float32(WALL_X),
    )
    print(f"[gen] saved {out}: {len(A)} steps, {ep} episodes")


if __name__ == "__main__":
    main()
