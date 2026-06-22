"""Stage 1d (Exp 1): UNSUPERVISED discrete mode discovery at the latent junction.

Stage 1c showed a discrete *supervised* (door-label) anchor commits while a
continuous predictor blurs. This removes the labels: a K=2 winner-take-all
(Multiple-Choice-Learning) predictor must DISCOVER the two branches on its own.

  loss = mean_samples  min_k || head_k(z_t, z_goal) - z_{t+d} ||^2

Each head specializes to one branch with no supervision. We check, on controlled
fixed-ridge configs:
  - between-ness of each MCL head  (~0 = commits onto a branch)  vs
  - continuous single-head         (~1 = stuck between branches = blur)
  - specialization: does the winning head correlate with the true door label?
    (purity >> 0.5 = the codebook genuinely discovered the branches)

Self-contained; writes outputs/tworoom_stage1d/RESULT.txt.
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


def closest_door(agent):
    return int(np.argmin([np.linalg.norm(agent - DOORS[0]), np.linalg.norm(agent - DOORS[1])]))


def resize64(img):
    t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)[None]
    return F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)[0]


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
            imgs.append(resize64(env.render().copy())); goalimgs.append(gimg)
            agents.append(ag.copy()); goals.append(goal.copy()); branches.append(assigned)
            eps.append(ep); steps.append(t)
            crossed = (ag[0] - WALL_X) * (goal[0] - WALL_X) > 0
            wp = goal if (crossed or np.linalg.norm(ag - DOORS[assigned]) < TOL) else DOORS[assigned]
            d = wp - ag; n = np.linalg.norm(d)
            a = (d / n if n > 1e-6 else d).astype(np.float32) + rng.normal(0, 0.03, 2).astype(np.float32)
            obs, *_ = env.step(np.clip(a, -1, 1)); obs = np.asarray(obs, np.float32)
            if term := (np.linalg.norm(obs[:2] - goal) < 16):
                break
        ep += 1
        if (e + 1) % 400 == 0:
            print(f"[1d] rendered {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    imgs = torch.stack(imgs); goalimgs = torch.stack(goalimgs)
    agents = np.array(agents, np.float32); goals = np.array(goals, np.float32)
    branches = np.array(branches, np.int64); eps = np.array(eps); steps = np.array(steps)
    pairs = np.array([(i, i + delta) for i in range(len(eps) - delta)
                      if eps[i] == eps[i + delta] and steps[i + delta] == steps[i] + delta])
    print(f"[1d] {len(imgs)} frames, {len(pairs)} pairs", flush=True)
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


class MCL(nn.Module):
    """K-head winner-take-all predictor: discovers modes without labels."""
    def __init__(self, din, dout, k=2, h=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(h, dout) for _ in range(k)])

    def forward(self, x):
        f = self.trunk(x)
        return torch.stack([head(f) for head in self.heads], dim=1)  # (B, K, dout)


@torch.no_grad()
def controlled_eval(ae, cont, mcl, mu, sd, delta, dev, n_configs=50, rolls=24, seed=123):
    rng = np.random.default_rng(seed)
    env = gym.make("swm/TwoRoom-v1")

    def enc(img_t):
        z = ae.encode(img_t[None].to(dev)).cpu().numpy()[0]
        return (z - mu) / sd

    cont_b, head_b, sep_l, win_purity = [], [], [], []
    for c in range(n_configs):
        ax = rng.uniform(LO + 6, WALL_X - 22); ay = float(rng.uniform(WALL_X - 6, WALL_X + 6))
        tx = rng.uniform(WALL_X + 22, HI - 6); ty = rng.uniform(LO + 6, HI - 6)
        start = [float(ax), ay]; tgt = [float(tx), float(ty)]
        env.reset(seed=int(rng.integers(1 << 30)), options={"variation_values": VV, "state": start, "target_state": tgt})
        u = env.unwrapped; gchw = u._target_img.float(); gchw = gchw / 255.0 if float(gchw.max()) > 1.5 else gchw
        zg = enc(F.interpolate(gchw[None], size=(64, 64), mode="bilinear", align_corners=False)[0])
        z0 = enc(resize64(env.render().copy()))
        zw, lab = [], []
        for k in range(rolls):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)), options={"variation_values": VV, "state": start, "target_state": tgt})
            obs = np.asarray(obs, np.float32); door = int(rng.integers(2))
            for t in range(delta):
                ag = obs[:2]; crossed = (ag[0] - WALL_X) * (tgt[0] - WALL_X) > 0
                wp = np.array(tgt, np.float32) if (crossed or np.linalg.norm(ag - DOORS[door]) < TOL) else DOORS[door]
                dd = wp - ag; n = np.linalg.norm(dd)
                a = (dd / n if n > 1e-6 else dd).astype(np.float32) + rng.normal(0, 0.04, 2).astype(np.float32)
                obs, *_ = env.step(np.clip(a, -1, 1)); obs = np.asarray(obs, np.float32)
            zw.append(enc(resize64(env.render().copy()))); lab.append(door)
        zw = np.array(zw, np.float32); lab = np.array(lab)
        if lab.sum() in (0, len(lab)):
            continue
        cen = np.stack([zw[lab == 0].mean(0), zw[lab == 1].mean(0)]); sep = np.linalg.norm(cen[0] - cen[1])
        half = sep / 2 + 1e-9
        x = torch.tensor(np.concatenate([z0, zg]).astype(np.float32)[None], device=dev)
        cp = cont(x).cpu().numpy()[0]
        mh = mcl(x).cpu().numpy()[0]  # (K, z)
        dnm = lambda p: min(np.linalg.norm(p - cen[0]), np.linalg.norm(p - cen[1]))
        cont_b.append(dnm(cp) / half)
        # committed head = the MCL head that lands closest to a real branch
        head_b.append(min(dnm(mh[kk]) / half for kk in range(mh.shape[0])))
        # specialization: for each rolled sample, which head best predicts it? correlate with door
        win = np.array([int(np.argmin([np.linalg.norm(mh[kk] - zw[j]) for kk in range(mh.shape[0])]))
                        for j in range(len(zw))])
        # head ids are arbitrary, so purity = max over the two label-to-head matchings
        agree = max((win == lab).mean(), (win != lab).mean())
        win_purity.append(agree); sep_l.append(sep)
    return {
        "n_configs_used": len(cont_b), "mean_sep": float(np.mean(sep_l)),
        "cont_between": float(np.mean(cont_b)), "mcl_head_between": float(np.mean(head_b)),
        "mcl_specialization_purity": float(np.mean(win_purity)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/tworoom_stage1d")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=70)
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=18)
    ap.add_argument("--pred-epochs", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print("[1d] generating...", flush=True)
    imgs, goalimgs, agents, goals, branches, steps, pairs = gen_images(args.episodes, args.max_steps, args.delta, args.seed)

    print("[1d] training AE...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3); N = len(imgs); bs = 256
    for ep in range(args.ae_epochs):
        perm = torch.randperm(N); tot = 0
        for s in range(0, N, bs):
            xb = imgs[perm[s:s+bs]].to(dev); rec, _ = ae(xb); loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    ae.eval()
    with torch.no_grad():
        Z = torch.cat([ae.encode(imgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
        Zg = torch.cat([ae.encode(goalimgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
    mu = Z.mean(0); sd = Z.std(0) + 1e-6; Z = (Z - mu) / sd; Zg = (Zg - mu) / sd
    ti, tj = pairs[:, 0], pairs[:, 1]
    X = np.concatenate([Z[ti], Zg[ti]], 1).astype(np.float32); zw = Z[tj].astype(np.float32)
    Xt = torch.tensor(X, device=dev); Yt = torch.tensor(zw, device=dev); n = len(Xt)

    print("[1d] training continuous + UNSUPERVISED MCL...", flush=True)
    cont = MLP(2*args.zdim, args.zdim).to(dev); oc = torch.optim.Adam(cont.parameters(), 1e-3)
    mcl = MCL(2*args.zdim, args.zdim, k=2).to(dev); om = torch.optim.Adam(mcl.parameters(), 1e-3)
    for ep in range(args.pred_epochs):
        pm = torch.randperm(n, device=dev)
        for s in range(0, n, 1024):
            idx = pm[s:s+1024]
            lc = F.mse_loss(cont(Xt[idx]), Yt[idx]); oc.zero_grad(); lc.backward(); oc.step()
            preds = mcl(Xt[idx])  # (B,K,z)
            err = ((preds - Yt[idx][:, None]) ** 2).mean(-1)  # (B,K)
            eps = 0.1  # relaxed MCL: keep both heads alive (avoid dead-head collapse)
            lm = ((1 - eps) * err.min(dim=1).values + eps * err.mean(dim=1)).mean()
            om.zero_grad(); lm.backward(); om.step()

    print("[1d] controlled eval...", flush=True)
    R = controlled_eval(ae, cont, mcl, mu, sd, args.delta, dev)
    (out / "stage1d_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== Stage 1d (Exp1): UNSUPERVISED discrete mode discovery (MCL, no labels) ===",
        f"controlled configs={R['n_configs_used']}  branch sep={R['mean_sep']:.2f}",
        "",
        "BETWEEN-NESS at ridge (dist to nearest branch / half-sep; ~1=blur, ~0=committed):",
        f"  continuous single-head     : {R['cont_between']:.2f}   (blur)",
        f"  UNSUP MCL committed head   : {R['mcl_head_between']:.2f}   (the discovered head commits onto a branch)",
        "",
        f"MCL specialization purity (winning head vs true door) = {R['mcl_specialization_purity']:.2f}",
        "  (>>0.5 => the 2 heads genuinely discovered the up/down branches with NO labels)",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines)); print("\n".join(lines), flush=True)
    print(f"[1d] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
