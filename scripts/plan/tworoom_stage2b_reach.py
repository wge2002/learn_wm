"""Stage 2b (Exp 2b): decisive diagnostic for Exp2's NEGATIVE result.

Exp2 found discrete subgoals LOSE to continuous (50% vs 64%) for latent planning.
Hypothesis: a well-trained goal-conditioned policy pi(z_t, z_sg) is ROBUST to a
blurred subgoal (it re-plans every step and also sees z_t), so it absorbs the
blur and the multimodal penalty never bites. Two confounds also hurt discrete in
Exp2: (i) the door was committed ONCE at step 0 and could lock onto the FAR door;
(ii) the subgoal was 10 steps ahead.

This script removes the "too-smart policy" confound with a FAITHFUL reach
controller: train a tiny latent->(x,y) probe, decode the subgoal latent to a
target position, and drive straight there. Now:
  - continuous subgoal -> decodes to the between-doors midpoint (= wall) -> the
    agent drives into the wall and stays stuck on the ridge (blur is fatal)
  - discrete subgoal -> decodes onto a door branch -> reaches a door -> success
This is the faithful LATENT analog of Stage 1 (where the policy WAS the blurring
regressor). Also ablates discrete commit-once vs recompute-each-step.

Conditions (all under the faithful reach controller):
  continuous | discrete-recompute | discrete-commit-once | expert-oracle
Writes outputs/tworoom_stage2b/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import stable_worldmodel as swm  # noqa: F401

torch.set_num_threads(1)

WALL_X = 112.0
DOORS = np.array([[112.0, 56.0], [112.0, 168.0]], dtype=np.float32)
VV = {"door.number": 2, "door.position": [56, 168, 49], "door.size": [18, 18, 14],
      "wall.axis": 1, "wall.thickness": 10}
LO, HI = 16.0, 207.0
RIDGE_AMBIG = 30.0
TOL = 12.0
GOAL_TOL = 16.0


def closest_door(agent):
    return int(np.argmin([np.linalg.norm(agent - DOORS[0]), np.linalg.norm(agent - DOORS[1])]))


def resize64(img):
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)[None]
    return F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)[0]


def waypoint(ag, goal, door_center):
    crossed = (ag[0] - WALL_X) * (goal[0] - WALL_X) > 0
    if crossed or np.linalg.norm(ag - door_center) < TOL:
        return goal
    return door_center


def gen_images(episodes, max_steps, delta, seed):
    rng = np.random.default_rng(seed)
    env = gym.make("swm/TwoRoom-v1")
    imgs, goalimgs, agents, goals, branches, eps, steps = [], [], [], [], [], [], []
    ep = 0
    for e in range(episodes):
        ax = rng.uniform(LO, WALL_X - 18); ay = rng.uniform(LO, HI)
        tx = rng.uniform(WALL_X + 18, HI); ty = rng.uniform(LO, HI)
        if rng.random() < 0.5:
            ax, tx = tx, ax
        a0 = np.array([ax, ay], np.float32)
        assigned = int(rng.integers(2)) if abs(ay - WALL_X) < RIDGE_AMBIG else closest_door(a0)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                           options={"variation_values": VV, "state": [float(ax), float(ay)],
                                    "target_state": [float(tx), float(ty)]})
        obs = np.asarray(obs, np.float32); goal = obs[2:4].copy()
        u = env.unwrapped; gchw = u._target_img.float()
        gchw = gchw / 255.0 if float(gchw.max()) > 1.5 else gchw
        gimg = F.interpolate(gchw[None], size=(64, 64), mode="bilinear", align_corners=False)[0].cpu()
        for t in range(max_steps):
            ag = obs[:2]
            wp = waypoint(ag, goal, DOORS[assigned])
            d = wp - ag; n = np.linalg.norm(d)
            a = (d / n if n > 1e-6 else d).astype(np.float32) + rng.normal(0, 0.03, 2).astype(np.float32)
            a = np.clip(a, -1, 1).astype(np.float32)
            imgs.append(resize64(env.render().copy())); goalimgs.append(gimg)
            agents.append(ag.copy()); goals.append(goal.copy()); branches.append(assigned)
            eps.append(ep); steps.append(t)
            obs, *_ = env.step(a); obs = np.asarray(obs, np.float32)
            if np.linalg.norm(obs[:2] - goal) < GOAL_TOL:
                break
        ep += 1
        if (e + 1) % 400 == 0:
            print(f"[2b] rendered {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    imgs = torch.stack(imgs); goalimgs = torch.stack(goalimgs)
    agents = np.array(agents, np.float32); goals = np.array(goals, np.float32)
    branches = np.array(branches, np.int64); eps = np.array(eps); steps = np.array(steps)
    pairs = np.array([(i, i + delta) for i in range(len(eps) - delta)
                      if eps[i] == eps[i + delta] and steps[i + delta] == steps[i] + delta])
    print(f"[2b] {len(imgs)} frames, {len(pairs)} (t,t+{delta}) pairs", flush=True)
    return imgs, goalimgs, agents, goals, branches, steps, pairs


class ConvAE(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(), nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(), nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.Flatten(), nn.Linear(64 * 16, zdim))
        self.dec_fc = nn.Linear(zdim, 64 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(), nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(self.dec_fc(z).view(-1, 64, 4, 4))

    def forward(self, x):
        z = self.encode(x); return self.decode(z), z


class MLP(nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/tworoom_stage2b")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=70)
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=18)
    ap.add_argument("--pred-epochs", type=int, default=120)
    ap.add_argument("--probe-epochs", type=int, default=60)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = {}

    print("[2b] generating...", flush=True)
    imgs, goalimgs, agents, goals, branches, steps, pairs = gen_images(
        args.episodes, args.max_steps, args.delta, args.seed)
    R["n_frames"] = int(len(imgs)); R["n_pairs"] = int(len(pairs))

    print("[2b] training conv AE...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    N = len(imgs); bs = 256
    for ep in range(args.ae_epochs):
        perm = torch.randperm(N); tot = 0
        for s in range(0, N, bs):
            xb = imgs[perm[s:s+bs]].to(dev); rec, _ = ae(xb); loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    R["ae_recon_mse"] = float(tot / (N // bs)); ae.eval()

    with torch.no_grad():
        Z = torch.cat([ae.encode(imgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
        Zg = torch.cat([ae.encode(goalimgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
    mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Zn = (Z - mu) / sd; Zgn = (Zg - mu) / sd

    ti, tj = pairs[:, 0], pairs[:, 1]
    X = np.concatenate([Zn[ti], Zgn[ti]], 1).astype(np.float32)
    zw = Zn[tj].astype(np.float32)
    branch = branches[ti]
    cross = (agents[ti, 0] - WALL_X) * (goals[ti, 0] - WALL_X) < 0

    def train(model, Xin, Yin, loss_fn, epochs, bs=1024):
        o = torch.optim.Adam(model.parameters(), 1e-3)
        Xt = torch.tensor(Xin, device=dev); Yt = torch.tensor(Yin, device=dev); n = len(Xt)
        for ep in range(epochs):
            pm = torch.randperm(n, device=dev)
            for s in range(0, n, bs):
                idx = pm[s:s+bs]; loss = loss_fn(model(Xt[idx]), Yt[idx])
                o.zero_grad(); loss.backward(); o.step()
        return model

    print("[2b] training subgoal predictors + latent->xy probe...", flush=True)
    cont = train(MLP(2*args.zdim, args.zdim).to(dev), X, zw, nn.MSELoss(), args.pred_epochs)
    clf = train(MLP(2*args.zdim, 2).to(dev), X[cross], branch[cross],
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.pred_epochs)
    Xb = np.concatenate([X, np.eye(2, dtype=np.float32)[branch]], 1)
    head = train(MLP(2*args.zdim + 2, args.zdim).to(dev), Xb, zw, nn.MSELoss(), args.pred_epochs)
    # faithful reach controller: decode any latent -> (x,y); trained on ALL frames
    probe = train(MLP(args.zdim, 2).to(dev), Zn.astype(np.float32), agents.astype(np.float32),
                  nn.MSELoss(), args.probe_epochs)
    with torch.no_grad():
        pr = probe(torch.tensor(Zn.astype(np.float32), device=dev)).cpu().numpy()
    R["probe_xy_mae_px"] = float(np.abs(pr - agents).mean())

    # ----- faithful reach-controller rollout from cross-room ridge starts -----
    print("[2b] reach-controller rollout eval...", flush=True)
    rng = np.random.default_rng(args.seed + 7)
    env = gym.make("swm/TwoRoom-v1")

    def enc_now():
        with torch.no_grad():
            z = ae.encode(resize64(env.render().copy())[None].to(dev)).cpu().numpy()[0]
        return ((z - mu) / sd).astype(np.float32)

    def encode_goal():
        gchw = env.unwrapped._target_img.float()
        gchw = gchw / 255.0 if float(gchw.max()) > 1.5 else gchw
        g64 = F.interpolate(gchw[None], size=(64, 64), mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            z = ae.encode(g64[None].to(dev)).cpu().numpy()[0]
        return ((z - mu) / sd).astype(np.float32)

    def to_xy(z):
        with torch.no_grad():
            return probe(torch.tensor(z[None], device=dev)).cpu().numpy()[0]

    def sg_continuous(z_t, z_g, door):
        with torch.no_grad():
            return cont(torch.tensor(np.concatenate([z_t, z_g])[None], device=dev)).cpu().numpy()[0]

    def sg_discrete(z_t, z_g, door):
        x = np.concatenate([z_t, z_g, np.eye(2, dtype=np.float32)[door]])[None]
        with torch.no_grad():
            return head(torch.tensor(x, device=dev)).cpu().numpy()[0]

    def pick_door(z_t, z_g):
        with torch.no_grad():
            return int(clf(torch.tensor(np.concatenate([z_t, z_g])[None], device=dev)).argmax(-1).item())

    def run(mode):
        """mode in {continuous, discrete_recompute, discrete_once}."""
        succ = 0; wall_stuck = 0
        for _ in range(args.eval_episodes):
            ax = rng.uniform(LO + 4, WALL_X - 20); ay = rng.uniform(96, 128)
            tx = rng.uniform(WALL_X + 20, HI - 4); ty = rng.uniform(LO + 4, HI - 4)
            if rng.random() < 0.5:
                ax, tx = tx, ax
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                               options={"variation_values": VV, "state": [float(ax), float(ay)],
                                        "target_state": [float(tx), float(ty)]})
            obs = np.asarray(obs, np.float32); g = obs[2:4].copy()
            z_g = encode_goal(); door = None; done = False
            for t in range(args.eval_max_steps):
                z_t = enc_now(); cur = obs[:2]
                if mode == "continuous":
                    z_sg = sg_continuous(z_t, z_g, 0)
                else:
                    if door is None or mode == "discrete_recompute":
                        door = pick_door(z_t, z_g)
                    z_sg = sg_discrete(z_t, z_g, door)
                tgt = to_xy(z_sg)
                d = tgt - cur; n = np.linalg.norm(d)
                a = (d / n if n > 1e-6 else d).astype(np.float32)
                obs, r, term, trunc, info = env.step(a); obs = np.asarray(obs, np.float32)
                if np.linalg.norm(obs[:2] - g) < GOAL_TOL or term:
                    succ += 1; done = True; break
            if not done and abs(obs[0] - WALL_X) < 14:
                wall_stuck += 1
        return 100.0 * succ / args.eval_episodes, 100.0 * wall_stuck / args.eval_episodes

    def run_expert():
        succ = 0
        for _ in range(args.eval_episodes):
            ax = rng.uniform(LO + 4, WALL_X - 20); ay = rng.uniform(96, 128)
            tx = rng.uniform(WALL_X + 20, HI - 4); ty = rng.uniform(LO + 4, HI - 4)
            if rng.random() < 0.5:
                ax, tx = tx, ax
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                               options={"variation_values": VV, "state": [float(ax), float(ay)],
                                        "target_state": [float(tx), float(ty)]})
            obs = np.asarray(obs, np.float32); g = obs[2:4].copy()
            for t in range(args.eval_max_steps):
                ag = obs[:2]; wp = waypoint(ag, g, DOORS[closest_door(ag)])
                d = wp - ag; n = np.linalg.norm(d)
                a = (d / n if n > 1e-6 else d).astype(np.float32)
                obs, r, term, trunc, info = env.step(a); obs = np.asarray(obs, np.float32)
                if np.linalg.norm(obs[:2] - g) < GOAL_TOL or term:
                    succ += 1; break
        return 100.0 * succ / args.eval_episodes

    s_cont, w_cont = run("continuous")
    s_drec, w_drec = run("discrete_recompute")
    s_donce, w_donce = run("discrete_once")
    s_exp = run_expert()
    R.update(success_continuous=s_cont, wallstuck_continuous=w_cont,
             success_discrete_recompute=s_drec, wallstuck_discrete_recompute=w_drec,
             success_discrete_once=s_donce, wallstuck_discrete_once=w_donce,
             success_expert_oracle=s_exp,
             discrete_recompute_minus_continuous=s_drec - s_cont)

    (out / "stage2b_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== Stage 2b (Exp2b): FAITHFUL reach controller (latent->xy probe) ===",
        f"frames={R['n_frames']} pairs={R['n_pairs']} ae_recon={R['ae_recon_mse']:.4f} "
        f"probe_xy_mae={R['probe_xy_mae_px']:.1f}px delta={args.delta} eval_eps={args.eval_episodes}",
        "",
        "success_rate from cross-room RIDGE starts (subgoal decoded->xy, drive straight):",
        f"  continuous subgoal          : {s_cont:5.1f}%   (blur->wall; wall-stuck {w_cont:.0f}%)",
        f"  DISCRETE (recompute door)   : {s_drec:5.1f}%   (commit->door; wall-stuck {w_drec:.0f}%)",
        f"  discrete (commit once @t0)  : {s_donce:5.1f}%   (ablation; wall-stuck {w_donce:.0f}%)",
        f"  expert oracle               : {s_exp:5.1f}%   (ceiling)",
        "",
        f"  >>> discrete(recompute) - continuous = {s_drec - s_cont:+.1f} points",
        f"      (commit-once vs recompute = {s_donce - s_drec:+.1f}, isolates the lock-far-door confound)",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[2b] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
