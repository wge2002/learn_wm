"""Stage 2 (Exp 2): does a discrete commitment subgoal beat a continuous one for
LATENT planning? Replicate Stage 1's +27 (state space) inside the learned latent.

Pipeline (self-contained; reuses 1c/1d gen + AE + predictors):

  1. STOCHASTIC-branch expert data -> conv AE -> latent z.
  2. Latent subgoal predictors  f(z_t, z_goal) -> z_{t+d}:
       - continuous single head  (blurs to the between-doors midpoint = wall)
       - discrete anchor: door clf + per-branch head (commits onto a door branch)
  3. Latent-subgoal-conditioned policy  pi(z_t, z_subgoal) -> action  (hindsight
     goal-conditioned BC: subgoal = TRUE z_{t+d}, target = expert action a_t).
  4. Roll pi from cross-room RIDGE starts, feeding it a PREDICTED subgoal each
     step (receding). Compare success_rate when the subgoal comes from:
       - continuous prediction (blur -> points into the wall -> fail)
       - discrete commitment   (lands on a door -> success)
       - direct goal latent     (diagnostic: goal is across the wall -> fail)
       - expert oracle (state-space waypoint, the absolute ceiling)

This is the hard evidence that the discrete commitment anchor is useful for
PLANNING, not just for next-latent fidelity. Writes outputs/tworoom_stage2/.
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
    """Door-anchored expert waypoint: head to door center until aligned, then goal."""
    crossed = (ag[0] - WALL_X) * (goal[0] - WALL_X) > 0
    if crossed or np.linalg.norm(ag - door_center) < TOL:
        return goal
    return door_center


def gen_images(episodes, max_steps, delta, seed):
    """Same stochastic-branch expert as 1c/1d, but ALSO record the executed action
    at each frame (needed to train the goal-conditioned policy)."""
    rng = np.random.default_rng(seed)
    env = gym.make("swm/TwoRoom-v1")
    imgs, goalimgs, agents, goals, branches, actions, eps, steps = [], [], [], [], [], [], [], []
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
            actions.append(a); eps.append(ep); steps.append(t)
            obs, *_ = env.step(a); obs = np.asarray(obs, np.float32)
            if np.linalg.norm(obs[:2] - goal) < GOAL_TOL:
                break
        ep += 1
        if (e + 1) % 400 == 0:
            print(f"[2] rendered {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    imgs = torch.stack(imgs); goalimgs = torch.stack(goalimgs)
    agents = np.array(agents, np.float32); goals = np.array(goals, np.float32)
    branches = np.array(branches, np.int64); actions = np.array(actions, np.float32)
    eps = np.array(eps); steps = np.array(steps)
    pairs = np.array([(i, i + delta) for i in range(len(eps) - delta)
                      if eps[i] == eps[i + delta] and steps[i + delta] == steps[i] + delta])
    print(f"[2] {len(imgs)} frames, {len(pairs)} (t,t+{delta}) pairs", flush=True)
    return imgs, goalimgs, agents, goals, branches, actions, steps, pairs


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
    ap.add_argument("--output-dir", default="outputs/tworoom_stage2")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=70)
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=18)
    ap.add_argument("--pred-epochs", type=int, default=120)
    ap.add_argument("--policy-epochs", type=int, default=120)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--eval-max-steps", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = {}

    print("[2] generating stochastic-branch images (with actions)...", flush=True)
    imgs, goalimgs, agents, goals, branches, actions, steps, pairs = gen_images(
        args.episodes, args.max_steps, args.delta, args.seed)
    R["n_frames"] = int(len(imgs)); R["n_pairs"] = int(len(pairs))

    print("[2] training conv AE...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    N = len(imgs); bs = 256
    for ep in range(args.ae_epochs):
        perm = torch.randperm(N); tot = 0
        for s in range(0, N, bs):
            xb = imgs[perm[s:s+bs]].to(dev); rec, _ = ae(xb); loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 6 == 0:
            print(f"[2]  ae ep{ep} recon={tot/(N//bs):.4f}", flush=True)
    R["ae_recon_mse"] = float(tot / (N // bs)); ae.eval()

    with torch.no_grad():
        Z = torch.cat([ae.encode(imgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
        Zg = torch.cat([ae.encode(goalimgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
    mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Zn = (Z - mu) / sd; Zgn = (Zg - mu) / sd

    ti, tj = pairs[:, 0], pairs[:, 1]
    X = np.concatenate([Zn[ti], Zgn[ti]], 1).astype(np.float32)  # (z_t, z_goal)
    zw = Zn[tj].astype(np.float32)                               # z_{t+d}
    act = actions[ti].astype(np.float32)                         # expert action at t
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

    print("[2] training latent subgoal predictors (continuous + discrete anchor)...", flush=True)
    cont = train(MLP(2*args.zdim, args.zdim).to(dev), X, zw, nn.MSELoss(), args.pred_epochs)
    clf = train(MLP(2*args.zdim, 2).to(dev), X[cross], branch[cross],
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.pred_epochs)
    Xb = np.concatenate([X, np.eye(2, dtype=np.float32)[branch]], 1)
    head = train(MLP(2*args.zdim + 2, args.zdim).to(dev), Xb, zw, nn.MSELoss(), args.pred_epochs)

    print("[2] training latent-subgoal-conditioned policy (hindsight GC-BC)...", flush=True)
    # pi(z_t, z_subgoal=TRUE z_{t+d}) -> a_t   (hindsight relabeling)
    Xpi = np.concatenate([Zn[ti], zw], 1).astype(np.float32)
    policy = train(MLP(2*args.zdim, 2).to(dev), Xpi, act, nn.MSELoss(), args.policy_epochs)

    # ----- rollout eval from cross-room ridge starts -----
    print("[2] rollout eval...", flush=True)
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

    def pi_action(z_t, z_sg):
        x = torch.tensor(np.concatenate([z_t, z_sg])[None], device=dev)
        with torch.no_grad():
            a = policy(x).cpu().numpy()[0]
        n = np.linalg.norm(a)
        return (a / n if n > 1e-6 else a).astype(np.float32)

    # subgoal sources: (z_t, z_g, committed_door) -> z_subgoal
    def sg_continuous(z_t, z_g, door):
        x = torch.tensor(np.concatenate([z_t, z_g])[None], device=dev)
        with torch.no_grad():
            return cont(x).cpu().numpy()[0]

    def sg_discrete(z_t, z_g, door):
        x = torch.tensor(np.concatenate([z_t, z_g, np.eye(2, dtype=np.float32)[door]])[None], device=dev)
        with torch.no_grad():
            return head(x).cpu().numpy()[0]

    def sg_direct(z_t, z_g, door):
        return z_g

    def run_latent(subgoal_fn):
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
            z_g = encode_goal()
            door = None
            done = False
            for t in range(args.eval_max_steps):
                z_t = enc_now()
                if door is None:  # commit a branch once, at the start
                    x = torch.tensor(np.concatenate([z_t, z_g])[None], device=dev)
                    with torch.no_grad():
                        door = int(clf(x).argmax(-1).item())
                z_sg = subgoal_fn(z_t, z_g, door)
                a = pi_action(z_t, z_sg)
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

    s_cont, w_cont = run_latent(sg_continuous)
    s_disc, w_disc = run_latent(sg_discrete)
    s_dir, w_dir = run_latent(sg_direct)
    s_exp = run_expert()
    R.update(success_continuous_subgoal=s_cont, wallstuck_continuous=w_cont,
             success_discrete_subgoal=s_disc, wallstuck_discrete=w_disc,
             success_direct_goal=s_dir, wallstuck_direct=w_dir,
             success_expert_oracle=s_exp, discrete_minus_continuous=s_disc - s_cont)

    # ----- money-shot: decode the subgoal at a ridge start (blur wall vs door) -----
    try:
        import PIL.Image as Image
        rng2 = np.random.default_rng(args.seed + 99)
        cont_dec, disc_dec = [], []
        for _ in range(8):
            ax = rng2.uniform(LO + 6, WALL_X - 22); ay = rng2.uniform(104, 120)
            tx = rng2.uniform(WALL_X + 22, HI - 6); ty = rng2.uniform(LO + 6, HI - 6)
            env.reset(seed=int(rng2.integers(1 << 30)),
                      options={"variation_values": VV, "state": [float(ax), float(ay)],
                               "target_state": [float(tx), float(ty)]})
            z_t = enc_now(); z_g = encode_goal()
            x = torch.tensor(np.concatenate([z_t, z_g])[None], device=dev)
            with torch.no_grad():
                door = int(clf(x).argmax(-1).item())
            cs = sg_continuous(z_t, z_g, door); ds = sg_discrete(z_t, z_g, door)
            with torch.no_grad():
                cont_dec.append(ae.decode(torch.tensor((cs * sd + mu)[None], device=dev)).cpu().numpy()[0])
                disc_dec.append(ae.decode(torch.tensor((ds * sd + mu)[None], device=dev)).cpu().numpy()[0])
        for tag, P in [("cont_subgoal", cont_dec), ("disc_subgoal", disc_dec)]:
            row = np.concatenate([(p.transpose(1, 2, 0) * 255).astype(np.uint8) for p in P], axis=1)
            Image.fromarray(row).save(out / f"ridge_decoded_{tag}.png")
    except Exception as e:
        R["viz_error"] = str(e)[:200]

    (out / "stage2_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== Stage 2 (Exp2): discrete commitment subgoal for LATENT planning ===",
        f"frames={R['n_frames']} pairs={R['n_pairs']} ae_recon={R['ae_recon_mse']:.4f} "
        f"delta={args.delta} eval_eps={args.eval_episodes}",
        "",
        "success_rate from cross-room RIDGE starts (subgoal-conditioned policy):",
        f"  continuous-prediction subgoal : {s_cont:5.1f}%   (blur -> wall; wall-stuck {w_cont:.0f}%)",
        f"  DISCRETE-commitment subgoal   : {s_disc:5.1f}%   (commit -> door; wall-stuck {w_disc:.0f}%)",
        f"  direct-goal subgoal (diag)    : {s_dir:5.1f}%   (goal across wall; wall-stuck {w_dir:.0f}%)",
        f"  expert oracle (state waypoint): {s_exp:5.1f}%   (ceiling)",
        "",
        f"  >>> discrete - continuous = {s_disc - s_cont:+.1f} points  (the latent analog of Stage 1's +27)",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[2] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
