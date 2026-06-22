"""L0: lift the WORKING discrete commitment (D0) into a LATENT world model on the
corridor testbed. This is the bridge experiment Exp2/Exp2b failed to deliver: does
the discrete commitment recover end-task success in LATENT (render -> conv AE -> z),
not just in raw state?

Pipeline (corridor, centered starts so every junction forces a commitment):
  1. render frames -> conv AE -> latent z (z_t encodes agent+goal+walls; z_goal =
     encode a frame with the marker at the goal).
  2. goal-conditioned policies on (z_t, z_goal):
       - continuous : pi -> action (MSE, blurs at each ridge)
       - D0         : selector q -> committed-door label (supervised, frozen),
                      then pi(z_t,z_goal,onehot(c)) -> action (BC).
  3. roll from centered starts, render+encode each step, commit a code on entering
     each segment; compare success vs continuous and the state-waypoint oracle.

Decisive outcomes:
  - D0-latent recovers toward ~100% while continuous stays low  => the LATENT
    control-layer payoff finally lands; Exp2/2b's failure was the testbed.
  - D0-latent collapses                                          => latent itself
    drops the info the commitment needs (a new, separate problem).

One (N, seed) per process -> launch several in parallel. Writes outputs/corridor_L0/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corridor_stage3 import CorridorMaze, expert_action, sample_doors, DOOR_Y  # noqa: E402


def reset_centered(env, rng, center):
    if center:
        return env.reset(float(rng.uniform(0.45, 0.55)), float(rng.uniform(0.45, 0.55)))
    return env.reset()


def goal_image(env):
    saved = env.pos.copy(); env.pos = env.goal.copy()
    img = env.render(64); env.pos = saved
    return img


def gen(n_walls, episodes, max_steps, rng, center):
    """Returns uint8 frames + per-frame (goal-episode idx, action, door label)."""
    imgs, gimgs, ep_of, acts, labs = [], [], [], [], []
    for e in range(episodes):
        env = CorridorMaze(n_walls, rng); reset_centered(env, rng, center)
        doors = sample_doors(n_walls, rng, multimodal=True)
        gimgs.append(goal_image(env))
        for t in range(max_steps):
            a = expert_action(env, doors)
            c = min(env.crossed(), n_walls - 1)
            lab = 0 if doors[c] == DOOR_Y[0] else 1
            imgs.append(env.render(64)); ep_of.append(e); acts.append(a); labs.append(lab)
            _, done = env.step(a)
            if done:
                break
        if (e + 1) % 500 == 0:
            print(f"[L0] gen {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    return (np.array(imgs, np.uint8), np.array(gimgs, np.uint8), np.array(ep_of, np.int64),
            np.array(acts, np.float32), np.array(labs, np.int64))


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


def to_float(u8, dev):
    return torch.from_numpy(u8).permute(0, 3, 1, 2).float().div_(255.0).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/corridor_L0")
    ap.add_argument("--walls", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=18)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--codes", type=int, default=2)
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    N = args.walls; K = args.codes; max_steps = 40 * (N + 1)
    rng = np.random.default_rng(args.seed + N)
    R = {"N": N, "seed": args.seed, "codes": K}

    print(f"[L0] N={N} seed={args.seed} generating...", flush=True)
    imgs, gimgs, ep_of, acts, labs = gen(N, args.episodes, max_steps, rng, args.center)
    M = len(imgs); R["n_frames"] = int(M)

    print(f"[L0] training conv AE on {M} frames...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3); bs = 256
    allimg = np.concatenate([imgs, gimgs], 0)
    for ep in range(args.ae_epochs):
        perm = np.random.permutation(len(allimg)); tot = 0; nb = 0
        for s in range(0, len(allimg), bs):
            xb = to_float(allimg[perm[s:s+bs]], dev); rec, _ = ae(xb)
            loss = F.mse_loss(rec, xb); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 6 == 0:
            print(f"[L0]  ae ep{ep} recon={tot/nb:.4f}", flush=True)
    R["ae_recon_mse"] = float(tot / nb); ae.eval()

    @torch.no_grad()
    def encode_u8(u8):
        Z = []
        for s in range(0, len(u8), 512):
            Z.append(ae.encode(to_float(u8[s:s+512], dev)).cpu().numpy())
        return np.concatenate(Z, 0)

    Zt = encode_u8(imgs); Zg_ep = encode_u8(gimgs)
    mu = Zt.mean(0); sd = Zt.std(0) + 1e-6
    Zt = (Zt - mu) / sd; Zg_ep = (Zg_ep - mu) / sd
    Zg = Zg_ep[ep_of]                                  # per-frame goal latent
    X = np.concatenate([Zt, Zg], 1).astype(np.float32)

    def train(model, Xin, Yin, loss_fn, epochs):
        o = torch.optim.Adam(model.parameters(), 1e-3)
        Xtt = torch.tensor(Xin, device=dev); Ytt = torch.tensor(Yin, device=dev); n = len(Xtt)
        for ep in range(epochs):
            pm = torch.randperm(n, device=dev)
            for s in range(0, n, 2048):
                idx = pm[s:s+2048]; loss = loss_fn(model(Xtt[idx]), Ytt[idx])
                o.zero_grad(); loss.backward(); o.step()
        return model

    print("[L0] training continuous + D0 (selector frozen, then policy)...", flush=True)
    cont = train(MLP(2*args.zdim, 2).to(dev), X, acts, nn.MSELoss(), args.epochs)
    sel = train(MLP(2*args.zdim, K).to(dev), X, labs,
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.epochs)
    for p in sel.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        C = F.one_hot(sel(torch.tensor(X, device=dev)).argmax(-1), K).float().cpu().numpy()
    pol = train(MLP(2*args.zdim + K, 2).to(dev), np.concatenate([X, C], 1).astype(np.float32),
                acts, nn.MSELoss(), args.epochs)

    # ----- rollout eval (render+encode each step) -----
    print("[L0] rollout eval...", flush=True)
    rg = np.random.default_rng(args.seed + 100 + N)

    @torch.no_grad()
    def enc_one(u8):
        z = ae.encode(to_float(u8[None], dev)).cpu().numpy()[0]
        return ((z - mu) / sd).astype(np.float32)

    def run(kind):
        succ = 0
        for _ in range(args.eval_episodes):
            env = CorridorMaze(N, rg); reset_centered(env, rg, args.center)
            doors = sample_doors(N, rg, multimodal=True)
            zg = enc_one(goal_image(env)); seg = -1; c_oh = None
            for t in range(max_steps):
                if kind == "oracle":
                    a = expert_action(env, doors)
                else:
                    zt = enc_one(env.render(64)); x = np.concatenate([zt, zg]).astype(np.float32)
                    if kind == "continuous":
                        with torch.no_grad():
                            a = cont(torch.tensor(x[None], device=dev)).cpu().numpy()[0]
                    else:
                        cur = env.crossed()
                        if cur != seg:
                            seg = cur
                            with torch.no_grad():
                                c_oh = F.one_hot(sel(torch.tensor(x[None], device=dev)).argmax(-1), K).float().cpu().numpy()[0]
                        with torch.no_grad():
                            a = pol(torch.tensor(np.concatenate([x, c_oh])[None], device=dev)).cpu().numpy()[0]
                    nrm = np.linalg.norm(a); a = (a / nrm if nrm > 1e-6 else a).astype(np.float32)
                _, done = env.step(a)
                if done:
                    succ += 1; break
            # noop
        return 100.0 * succ / args.eval_episodes

    res = {k: run(k) for k in ["oracle", "continuous", "D0"]}
    R["results"] = res
    R["D0_minus_continuous"] = res["D0"] - res["continuous"]

    (out / f"L0_N{N}_seed{args.seed}.json").write_text(json.dumps(R, indent=2))
    line = (f"[L0] N={N} seed={args.seed}: oracle={res['oracle']:.0f}  "
            f"continuous={res['continuous']:.0f}  D0={res['D0']:.0f}  "
            f"(D0-cont={res['D0']-res['continuous']:+.0f}, ae_recon={R['ae_recon_mse']:.3f})")
    print(line, flush=True)
    (out / f"RESULT_N{N}_seed{args.seed}.txt").write_text(line)


if __name__ == "__main__":
    main()
