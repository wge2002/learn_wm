"""Stage 3 testbed: a multi-junction, long-horizon, IRREVERSIBLE corridor maze.

Why a new testbed: Exp2/Exp2b showed discrete commitment has NO planning
advantage in TwoRoom-2door, for two reasons that are NOT about the idea:
  (i)  a single junction's blur is recoverable by receding re-planning;
  (ii) the binding constraint there is door-threading (execution), not blur.

This env removes both confounds so that *commitment* becomes the binding
constraint:
  - N walls in a horizontal corridor, each with 2 wide doors (top/bottom).
    The agent must chain N commitments to reach the goal -> horizon ~ N.
  - All walls are ONE-WAY (cannot move back left through any wall plane) and the
    time budget is tight -> a blur-into-wall at junction i is UNRECOVERABLE.
  - Doors are wide -> once committed, a near-straight policy threads easily, so
    success is determined by *choosing* a door, not by *threading* it.

Falsifiable prediction that VALIDATES the testbed:
  - oracle (commits to a door each junction)      -> ~100% for all N
  - continuous BC (regresses expert action)       -> decays ~ p^N (blurs into the
    wall at each ridge with prob ~(1-p)) while oracle stays flat.
If continuous decays with N and oracle stays ~100%, the testbed has the property
we need (multimodal commitment is binding and compounds over the horizon).

Pure numpy/torch env (no swm dependency) -> runs locally, fully controllable.
Writes outputs/corridor_stage3/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

DOOR_Y = (0.25, 0.75)       # two door centers (top, bottom)
DOOR_HALF = 0.075           # half-width of a door opening (wide -> easy to thread)
SPEED = 0.035               # per-step max displacement
START_X = 0.04
GOAL_X = 0.96
GOAL_TOL = 0.05
RIDGE_BAND = 0.10           # |y-0.5|<band near a wall => ambiguous (both doors ~equal)


class CorridorMaze:
    """Continuous 2D corridor with N one-way walls, 2 wide doors each."""

    def __init__(self, n_walls, rng):
        self.N = n_walls
        self.rng = rng
        self.walls = np.array([(i + 1) / (n_walls + 1) for i in range(n_walls)], np.float32)
        self._bg = {}        # cache static wall background per pixel size

    def reset(self, start_y=None, goal_y=None):
        ay = self.rng.uniform(0.1, 0.9) if start_y is None else start_y
        gy = self.rng.uniform(0.1, 0.9) if goal_y is None else goal_y
        self.pos = np.array([START_X, ay], np.float32)
        self.goal = np.array([GOAL_X, gy], np.float32)
        return self.state()

    def state(self):
        return np.concatenate([self.pos, self.goal]).astype(np.float32)

    def crossed(self):
        return int(np.sum(self.walls < self.pos[0]))

    def step(self, a):
        a = np.clip(a, -1, 1).astype(np.float32)
        ox, oy = self.pos
        nx, ny = np.clip(self.pos + a * SPEED, 0.0, 1.0)
        # a single step (<SPEED) crosses at most one wall (spacing >= 1/(N+1))
        for wx in self.walls:
            crossing_r = ox < wx <= nx
            crossing_l = nx < wx <= ox
            if not (crossing_r or crossing_l):
                continue
            t = (wx - ox) / (nx - ox) if abs(nx - ox) > 1e-9 else 0.0
            yc = oy + t * (ny - oy)
            in_door = any(abs(yc - dy) < DOOR_HALF for dy in DOOR_Y)
            if crossing_l or not in_door:        # one-way: block ALL leftward; block rightward into wall
                nx = (wx - 0.004) if crossing_r else (wx + 0.004)
            break
        self.pos = np.array([nx, ny], np.float32)
        done = bool(np.linalg.norm(self.pos - self.goal) < GOAL_TOL)
        return self.state(), done

    def _background(self, px):
        if px not in self._bg:
            img = np.full((px, px, 3), 235, np.uint8)
            for wx in self.walls:
                xc = int(np.clip(wx, 0, 1) * (px - 1))
                for yp in range(px):
                    yv = yp / (px - 1)
                    if not any(abs(yv - dy) < DOOR_HALF for dy in DOOR_Y):
                        img[yp, max(xc - 1, 0):xc + 2] = (90, 90, 90)
            self._bg[px] = img
        return self._bg[px]

    def render(self, px=64):
        # walls are static -> copy the cached background, stamp goal + agent only
        img = self._background(px).copy()
        def P(c):
            return int(np.clip(c, 0, 1) * (px - 1))
        gy, gx = P(self.goal[1]), P(self.goal[0])
        img[max(gy-2, 0):gy+3, max(gx-2, 0):gx+3] = (40, 200, 40)
        ay, ax = P(self.pos[1]), P(self.pos[0])
        img[max(ay-2, 0):ay+3, max(ax-2, 0):ax+3] = (220, 40, 40)
        return img


def expert_action(env, chosen_doors):
    """Commit to chosen_doors[i] at wall i; approach door from left, thread, repeat."""
    ax, ay = env.pos
    c = env.crossed()
    if c >= env.N:
        target = env.goal
    else:
        wx = env.walls[c]; dy = chosen_doors[c]
        if ax > wx - 0.06 and abs(ay - dy) < DOOR_HALF * 0.8:
            target = np.array([wx + 0.05, dy], np.float32)     # aligned & close -> push through
        else:
            target = np.array([wx - 0.025, dy], np.float32)    # approach door face at right y
    d = target - env.pos; n = np.linalg.norm(d)
    return (d / n if n > 1e-6 else d).astype(np.float32)


def sample_doors(n_walls, rng, multimodal):
    """multimodal: random committed door per wall (genuine bimodality at each
    junction). unimodal control: every wall uses the SAME fixed door -> the task
    is deterministic, so a continuous BC should NOT blur and should NOT decay
    with N. Comparing the two isolates 'decay from blur' vs 'decay from length'."""
    if multimodal:
        return np.array([DOOR_Y[rng.integers(2)] for _ in range(n_walls)], np.float32)
    return np.array([DOOR_Y[0]] * n_walls, np.float32)


def gen_data(n_walls, episodes, max_steps, rng, multimodal):
    S, A, ridge_ay = [], [], []
    for _ in range(episodes):
        env = CorridorMaze(n_walls, rng); env.reset()
        doors = sample_doors(n_walls, rng, multimodal)
        for t in range(max_steps):
            a = expert_action(env, doors)
            S.append(env.state()); A.append(a)
            c = env.crossed()
            # ambiguous: just entered a segment (left of the next wall), action
            # points up or down depending on the (random) committed door
            if c < n_walls and env.pos[0] < env.walls[c] - 0.04 and abs(a[1]) > 0.2:
                ridge_ay.append(a[1])
            _, done = env.step(a)
            if done:
                break
    return np.array(S, np.float32), np.array(A, np.float32), np.array(ridge_ay, np.float32)


class MLP(nn.Module):
    def __init__(self, din, dout, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/corridor_stage3")
    ap.add_argument("--walls", default="1,2,3,4")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--bc-epochs", type=int, default=80)
    ap.add_argument("--eval-episodes", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(args.seed)
    Ns = [int(x) for x in args.walls.split(",")]
    R = {"per_N": {}}

    def train_bc(S, A):
        bc = MLP(4, 2).to(dev); opt = torch.optim.Adam(bc.parameters(), 1e-3)
        Xt = torch.tensor(S, device=dev); Yt = torch.tensor(A, device=dev); n = len(Xt)
        for ep in range(args.bc_epochs):
            pm = torch.randperm(n, device=dev)
            for s in range(0, n, 2048):
                idx = pm[s:s+2048]; loss = F.mse_loss(bc(Xt[idx]), Yt[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return bc

    def run(N, policy, seed, max_steps, multimodal):
        rg = np.random.default_rng(seed); succ = 0
        for _ in range(args.eval_episodes):
            env = CorridorMaze(N, rg); env.reset()
            doors = sample_doors(N, rg, multimodal)
            for t in range(max_steps):
                a = policy(env, doors)
                _, done = env.step(a)
                if done:
                    succ += 1; break
        return 100.0 * succ / args.eval_episodes

    def bc_policy_fn(bc):
        def f(env, doors):
            with torch.no_grad():
                a = bc(torch.tensor(env.state()[None], device=dev)).cpu().numpy()[0]
            nn_ = np.linalg.norm(a)
            return (a / nn_ if nn_ > 1e-6 else a).astype(np.float32)
        return f

    for N in Ns:
        max_steps = 40 * (N + 1)      # tight-ish budget that scales with horizon
        row = {"max_steps": max_steps}
        for tag, multimodal in [("multimodal", True), ("unimodal", False)]:
            rng = np.random.default_rng(args.seed + N + (0 if multimodal else 777))
            S, A, ridge_ay = gen_data(N, args.episodes, max_steps, rng, multimodal)
            frac_up = float((ridge_ay < 0).mean()) if len(ridge_ay) else float("nan")
            bimodal = bool(len(ridge_ay) > 50 and 0.3 < frac_up < 0.7)
            bc = train_bc(S, A)
            s_oracle = run(N, lambda e, d: expert_action(e, d), args.seed + 100 + N, max_steps, multimodal)
            s_bc = run(N, bc_policy_fn(bc), args.seed + 100 + N, max_steps, multimodal)
            row[tag] = {"oracle": s_oracle, "continuous_bc": s_bc,
                        "ridge_frac_up": frac_up, "ridge_bimodal": bimodal, "ridge_n": int(len(ridge_ay))}
        R["per_N"][N] = row
        mm, um = row["multimodal"], row["unimodal"]
        print(f"[3] N={N}: MULTI oracle={mm['oracle']:.0f}% BC={mm['continuous_bc']:.0f}% "
              f"(ridge up={mm['ridge_frac_up']:.2f} bim={mm['ridge_bimodal']}) | "
              f"UNI oracle={um['oracle']:.0f}% BC={um['continuous_bc']:.0f}%", flush=True)

    # save one render to confirm the scene
    try:
        import PIL.Image as Image
        env = CorridorMaze(max(Ns), np.random.default_rng(0)); env.reset(0.5, 0.7)
        Image.fromarray(env.render(128)).resize((256, 256), Image.NEAREST).save(out / "scene_example.png")
    except Exception as e:
        R["viz_error"] = str(e)[:200]

    (out / "corridor_stage3_summary.json").write_text(json.dumps(R, indent=2))
    lines = ["=== Stage 3 testbed: multi-junction one-way corridor maze ===",
             "validation: MULTIMODAL continuous-BC should decay with horizon N while",
             "the UNIMODAL control (fixed door) stays flat => decay is from blur, not length.",
             "",
             f"{'N':>3} | {'oracle':>7} | {'MULTI BC':>9} | {'UNI BC':>7} | ridge bimodal",
             "-" * 56]
    for N in Ns:
        mm, um = R["per_N"][N]["multimodal"], R["per_N"][N]["unimodal"]
        lines.append(f"{N:>3} | {mm['oracle']:>6.1f}% | {mm['continuous_bc']:>8.1f}% | "
                     f"{um['continuous_bc']:>6.1f}% | up={mm['ridge_frac_up']:.2f} "
                     f"{'OK' if mm['ridge_bimodal'] else '--'}")
    lines += ["",
              "PASS if: oracle~100% all N; MULTI BC decays as N grows; UNI BC stays ~flat.",
              "(=> the testbed's binding constraint is multimodal commitment, compounding",
              " over the horizon -- exactly where a discrete commitment token should pay off.)"]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[3] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
