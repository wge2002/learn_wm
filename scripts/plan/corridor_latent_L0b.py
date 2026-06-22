"""L0b: diagnose WHY D0 doesn't cleanly transfer to latent (L0 found D0-cont
correlates -0.94 with AE recon, n=6). Two additions over L0:

  1. PROBES that measure what the latent actually preserves, per seed:
       - latent -> agent (x,y)   : position fidelity (MAE, reported in px on 64)
       - latent -> committed door: commitment-info fidelity (holdout accuracy)
     If D0 success tracks probe quality, the bottleneck is the REPRESENTATION,
     not the commitment idea -> connects the solution back to the diagnosis MD.
  2. STABILIZED / stronger AE (more epochs) to test the positive direction: with
     a good AE (high probe fidelity), does D0 now ROBUSTLY beat continuous?

Methods unchanged from L0: continuous, D0 (supervised-frozen selector + BC policy),
state-waypoint oracle. One (N,seed) per process. Writes outputs/corridor_L0b/.
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
    imgs, gimgs, ep_of, agents, acts, labs = [], [], [], [], [], []
    for e in range(episodes):
        env = CorridorMaze(n_walls, rng); reset_centered(env, rng, center)
        doors = sample_doors(n_walls, rng, multimodal=True)
        gimgs.append(goal_image(env))
        for t in range(max_steps):
            a = expert_action(env, doors)
            c = min(env.crossed(), n_walls - 1)
            lab = 0 if doors[c] == DOOR_Y[0] else 1
            imgs.append(env.render(64)); ep_of.append(e); agents.append(env.pos.copy())
            acts.append(a); labs.append(lab)
            _, done = env.step(a)
            if done:
                break
        if (e + 1) % 500 == 0:
            print(f"[L0b] gen {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    return (np.array(imgs, np.uint8), np.array(gimgs, np.uint8), np.array(ep_of, np.int64),
            np.array(agents, np.float32), np.array(acts, np.float32), np.array(labs, np.int64))


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
    ap.add_argument("--output-dir", default="outputs/corridor_L0b")
    ap.add_argument("--walls", type=int, default=4)
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=30)        # stabilized (was 18)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--probe-epochs", type=int, default=60)
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

    print(f"[L0b] N={N} seed={args.seed} generating...", flush=True)
    imgs, gimgs, ep_of, agents, acts, labs = gen(N, args.episodes, max_steps, rng, args.center)
    M = len(imgs); R["n_frames"] = int(M)

    print(f"[L0b] training conv AE ({args.ae_epochs} ep) on {M} frames...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3); bs = 256
    allimg = np.concatenate([imgs, gimgs], 0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.ae_epochs)
    for ep in range(args.ae_epochs):
        perm = np.random.permutation(len(allimg)); tot = 0; nb = 0
        for s in range(0, len(allimg), bs):
            xb = to_float(allimg[perm[s:s+bs]], dev); rec, _ = ae(xb)
            loss = F.mse_loss(rec, xb); opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if ep % 8 == 0:
            print(f"[L0b]  ae ep{ep} recon={tot/nb:.4f}", flush=True)
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
    Zg = Zg_ep[ep_of]
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

    # ---- PROBES: what does the latent preserve? (train on 90%, report on 10%) ----
    ntr = int(0.9 * M); idx = np.random.permutation(M); tr, te = idx[:ntr], idx[ntr:]
    print("[L0b] probes: latent->xy, latent->door...", flush=True)
    pxy = train(MLP(args.zdim, 2).to(dev), Zt[tr], agents[tr], nn.MSELoss(), args.probe_epochs)
    with torch.no_grad():
        xy_pred = pxy(torch.tensor(Zt[te], device=dev)).cpu().numpy()
    R["probe_xy_mae_px"] = float(np.abs(xy_pred - agents[te]).mean() * 64.0)   # px on 64-grid
    pdr = train(MLP(2*args.zdim, K).to(dev), X[tr], labs[tr],
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.probe_epochs)
    with torch.no_grad():
        dr_pred = pdr(torch.tensor(X[te], device=dev)).argmax(-1).cpu().numpy()
    R["probe_door_acc"] = float((dr_pred == labs[te]).mean())

    # ---- policies: continuous + D0 ----
    print("[L0b] training continuous + D0...", flush=True)
    cont = train(MLP(2*args.zdim, 2).to(dev), X, acts, nn.MSELoss(), args.epochs)
    sel = train(MLP(2*args.zdim, K).to(dev), X, labs,
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.epochs)
    for p in sel.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        C = F.one_hot(sel(torch.tensor(X, device=dev)).argmax(-1), K).float().cpu().numpy()
    pol = train(MLP(2*args.zdim + K, 2).to(dev), np.concatenate([X, C], 1).astype(np.float32),
                acts, nn.MSELoss(), args.epochs)

    # ---- rollout eval ----
    print("[L0b] rollout eval...", flush=True)
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
        return 100.0 * succ / args.eval_episodes

    res = {k: run(k) for k in ["oracle", "continuous", "D0"]}
    R["results"] = res
    R["D0_minus_continuous"] = res["D0"] - res["continuous"]

    (out / f"L0b_N{N}_seed{args.seed}.json").write_text(json.dumps(R, indent=2))
    line = (f"[L0b] N={N} seed={args.seed}: oracle={res['oracle']:.0f}  cont={res['continuous']:.0f}  "
            f"D0={res['D0']:.0f}  (D0-cont={res['D0']-res['continuous']:+.0f})  "
            f"| recon={R['ae_recon_mse']:.4f}  probe_xy={R['probe_xy_mae_px']:.2f}px  "
            f"probe_door={R['probe_door_acc']:.2f}")
    print(line, flush=True)
    (out / f"RESULT_N{N}_seed{args.seed}.txt").write_text(line)


if __name__ == "__main__":
    main()
