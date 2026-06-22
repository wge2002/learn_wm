"""Stage 1c: latent multimodal-junction test with GENUINE same-input bimodality.

Stage 1b was inconclusive: the encoder resolved the branch (input not ambiguous)
and the two branch modes overlapped. This version fixes both:

  - STOCHASTIC-branch expert: near the y~112 ridge the expert commits to a
    RANDOMLY chosen door (50/50) for the whole episode. So nearby start states
    map to BOTH futures -> genuine bimodal p(future | z_t, z_goal); a continuous
    L2 predictor must output the MEAN (between the doors = into the wall).
  - larger delta so up/down branches separate in latent (sep >> within).

Pipeline (else identical to 1b): render -> conv AE -> latent; continuous vs
discrete-anchor latent predictor; blur metric + decoded images at the ridge.
Writes outputs/tworoom_stage1c/RESULT.txt.
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


def resize64(img_hwc_uint8):
    t = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float().div_(255.0)[None]
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
        agent0 = np.array([ax, ay], np.float32)
        if abs(ay - WALL_X) < RIDGE_AMBIG:
            assigned = int(rng.integers(2))           # stochastic at the ridge
        else:
            assigned = closest_door(agent0)
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                           options={"variation_values": VV, "state": [float(ax), float(ay)],
                                    "target_state": [float(tx), float(ty)]})
        obs = np.asarray(obs, np.float32); goal = obs[2:4].copy()
        u = env.unwrapped
        gchw = u._target_img.float()
        if float(gchw.max()) > 1.5:
            gchw = gchw / 255.0
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
            obs, r, term, trunc, info = env.step(np.clip(a, -1, 1)); obs = np.asarray(obs, np.float32)
            if term:
                break
        ep += 1
        if (e + 1) % 300 == 0:
            print(f"[1c] rendered {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    imgs = torch.stack(imgs); goalimgs = torch.stack(goalimgs)
    agents = np.array(agents, np.float32); goals = np.array(goals, np.float32)
    branches = np.array(branches, np.int64); eps = np.array(eps); steps = np.array(steps)
    pairs = [(i, i + delta) for i in range(len(eps) - delta)
             if eps[i] == eps[i + delta] and steps[i + delta] == steps[i] + delta]
    pairs = np.array(pairs)
    print(f"[1c] {len(imgs)} frames, {len(pairs)} (t,t+{delta}) pairs", flush=True)
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


@torch.no_grad()
def controlled_ridge_eval(ae, cont, clf, head, mu, sd, zdim, delta, dev,
                          n_configs=40, rolls=24, seed=123):
    """Conditional bimodality: FIXED ridge (start, goal) configs, each rolled many
    times with a random door. For a fixed input the true next-latents form two
    clean clusters; a continuous L2 predictor outputs ONE value (their mean ->
    between clusters = blur); the discrete anchor commits to a cluster."""
    rng = np.random.default_rng(seed)
    env = gym.make("swm/TwoRoom-v1")

    def enc(img_t):
        z = ae.encode(img_t[None].to(dev)).cpu().numpy()[0]
        return (z - mu) / sd

    cont_blurs, disc_blurs, seps, withins = [], [], [], []
    cont_decimg, disc_decimg, true_decimg = [], [], []
    for c in range(n_configs):
        ax = rng.uniform(LO + 6, WALL_X - 22); ay = float(rng.uniform(WALL_X - 6, WALL_X + 6))
        tx = rng.uniform(WALL_X + 22, HI - 6); ty = rng.uniform(LO + 6, HI - 6)
        start = [float(ax), ay]; tgt = [float(tx), float(ty)]
        env.reset(seed=int(rng.integers(1 << 30)),
                  options={"variation_values": VV, "state": start, "target_state": tgt})
        u = env.unwrapped
        gchw = u._target_img.float()
        gchw = gchw / 255.0 if float(gchw.max()) > 1.5 else gchw
        zg = enc(F.interpolate(gchw[None], size=(64, 64), mode="bilinear", align_corners=False)[0])
        z0 = enc(resize64(env.render().copy()))
        zw_list, lab_list = [], []
        for k in range(rolls):
            obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                               options={"variation_values": VV, "state": start, "target_state": tgt})
            obs = np.asarray(obs, np.float32); door = int(rng.integers(2))
            for t in range(delta):
                ag = obs[:2]
                crossed = (ag[0] - WALL_X) * (tgt[0] - WALL_X) > 0
                wp = np.array(tgt, np.float32) if (crossed or np.linalg.norm(ag - DOORS[door]) < TOL) else DOORS[door]
                dd = wp - ag; n = np.linalg.norm(dd)
                a = (dd / n if n > 1e-6 else dd).astype(np.float32) + rng.normal(0, 0.04, 2).astype(np.float32)
                obs, *_ = env.step(np.clip(a, -1, 1)); obs = np.asarray(obs, np.float32)
            zw_list.append(enc(resize64(env.render().copy()))); lab_list.append(door)
        zw = np.array(zw_list, np.float32); lab = np.array(lab_list)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        cen = np.stack([zw[lab == 0].mean(0), zw[lab == 1].mean(0)])
        within = np.mean([np.linalg.norm(zw[lab == b] - cen[b], axis=1).mean() for b in (0, 1)])
        sep = np.linalg.norm(cen[0] - cen[1])
        x = np.concatenate([z0, zg]).astype(np.float32)[None]
        cp = cont(torch.tensor(x, device=dev)).cpu().numpy()[0]
        bsel = int(clf(torch.tensor(x, device=dev)).argmax(-1).item())
        dp = head(torch.tensor(np.concatenate([x[0], np.eye(2, dtype=np.float32)[bsel]])[None], device=dev)).cpu().numpy()[0]
        # "between-ness": dist to nearest branch as a fraction of half-separation.
        # ~1.0 = exactly between the two branches (blur); ~0 = sitting on a branch.
        half = sep / 2 + 1e-9
        dnm = lambda p: min(np.linalg.norm(p - cen[0]), np.linalg.norm(p - cen[1]))
        cont_blurs.append(dnm(cp) / half); disc_blurs.append(dnm(dp) / half)
        seps.append(sep); withins.append(within)
        if len(cont_decimg) < 8:
            cont_decimg.append(cp); disc_decimg.append(dp); true_decimg.append(cen[lab[0]])
    return {
        "n_configs_used": len(cont_blurs),
        "mean_sep": float(np.mean(seps)), "mean_within": float(np.mean(withins)),
        "sep_over_within": float(np.mean(seps) / (np.mean(withins) + 1e-9)),
        "cont_blur": float(np.mean(cont_blurs)), "disc_blur": float(np.mean(disc_blurs)),
    }, (np.array(cont_decimg), np.array(disc_decimg), np.array(true_decimg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/tworoom_stage1c")
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--max-steps", type=int, default=70)
    ap.add_argument("--delta", type=int, default=10)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=18)
    ap.add_argument("--pred-epochs", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = {}

    print("[1c] generating stochastic-branch images...", flush=True)
    imgs, goalimgs, agents, goals, branches, steps, pairs = gen_images(
        args.episodes, args.max_steps, args.delta, args.seed)
    R["n_frames"] = int(len(imgs)); R["n_pairs"] = int(len(pairs))

    print("[1c] training conv AE...", flush=True)
    ae = ConvAE(args.zdim).to(dev); opt = torch.optim.Adam(ae.parameters(), 1e-3)
    N = len(imgs); bs = 256
    for ep in range(args.ae_epochs):
        perm = torch.randperm(N); tot = 0
        for s in range(0, N, bs):
            xb = imgs[perm[s:s+bs]].to(dev); rec, _ = ae(xb); loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 6 == 0:
            print(f"[1c]  ae ep{ep} recon={tot/(N//bs):.4f}", flush=True)
    R["ae_recon_mse"] = float(tot / (N // bs))

    ae.eval()
    with torch.no_grad():
        Z = torch.cat([ae.encode(imgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
        Zg = torch.cat([ae.encode(goalimgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)]).numpy()
    mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Z = (Z - mu) / sd; Zg = (Zg - mu) / sd

    ti, tj = pairs[:, 0], pairs[:, 1]
    zt, zg, zw = Z[ti], Zg[ti], Z[tj]
    X = np.concatenate([zt, zg], 1).astype(np.float32)
    branch = branches[ti]
    cross = (agents[ti, 0] - WALL_X) * (goals[ti, 0] - WALL_X) < 0

    def train(model, Xin, Yin, loss_fn, epochs):
        o = torch.optim.Adam(model.parameters(), 1e-3)
        Xt = torch.tensor(Xin, device=dev); Yt = torch.tensor(Yin, device=dev); n = len(Xt)
        for ep in range(epochs):
            pm = torch.randperm(n, device=dev)
            for s in range(0, n, 1024):
                idx = pm[s:s+1024]; loss = loss_fn(model(Xt[idx]), Yt[idx])
                o.zero_grad(); loss.backward(); o.step()
        return model

    print("[1c] training latent predictors...", flush=True)
    cont = train(MLP(2*args.zdim, args.zdim).to(dev), X, zw, nn.MSELoss(), args.pred_epochs)
    clf = train(MLP(2*args.zdim, 2).to(dev), X[cross], branch[cross],
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.pred_epochs)
    Xb = np.concatenate([X, np.eye(2, dtype=np.float32)[branch]], 1)
    head = train(MLP(2*args.zdim + 2, args.zdim).to(dev), Xb, zw, nn.MSELoss(), args.pred_epochs)

    # genuine ambiguity only at the pre-commitment START state: once the agent
    # moves toward its (randomly) assigned door, its position reveals the branch.
    ridge = cross & (np.abs(agents[ti, 1] - WALL_X) < 12) & (steps[ti] < 2)
    R["ridge_pairs"] = int(ridge.sum())
    rb = branch[ridge]; rzw = zw[ridge]
    cen = np.stack([rzw[rb == 0].mean(0), rzw[rb == 1].mean(0)])
    with torch.no_grad():
        Xr = torch.tensor(X[ridge], device=dev)
        cont_pred = cont(Xr).cpu().numpy()
        bsel = clf(Xr).argmax(-1).cpu().numpy()
        disc_pred = head(torch.tensor(np.concatenate([X[ridge], np.eye(2, dtype=np.float32)[bsel]], 1),
                                      device=dev)).cpu().numpy()

    def dnm(P):
        return np.minimum(np.linalg.norm(P - cen[0], axis=1), np.linalg.norm(P - cen[1], axis=1))
    sep = float(np.linalg.norm(cen[0] - cen[1]))
    within = float(np.mean([np.linalg.norm(rzw[rb == b] - cen[b], axis=1).mean()
                            for b in (0, 1) if (rb == b).sum() > 0]))
    R.update(mode_separation=sep, within_branch_spread=within,
             cont_pred_err_to_true=float(np.mean(np.linalg.norm(cont_pred - rzw, axis=1))),
             disc_pred_err_to_true=float(np.mean(np.linalg.norm(disc_pred - rzw, axis=1))),
             cont_blur=float(np.mean(dnm(cont_pred))) / (within + 1e-9),
             disc_blur=float(np.mean(dnm(disc_pred))) / (within + 1e-9),
             branch_clf_acc_at_ridge=float((bsel == rb).mean()))

    # --- controlled conditional eval (fixed ridge configs, repeated rolls) ---
    print("[1c] controlled conditional ridge eval...", flush=True)
    cres, (cimg, dimg, timg) = controlled_ridge_eval(
        ae, cont, clf, head, mu, sd, args.zdim, args.delta, dev)
    R["controlled"] = cres

    try:
        import PIL.Image as Image
        with torch.no_grad():
            for tag, P in [("cont", cimg), ("disc", dimg), ("true", timg)]:
                if len(P) == 0:
                    continue
                dec = ae.decode(torch.tensor(P, device=dev)).cpu().numpy()
                row = np.concatenate([(dec[k].transpose(1, 2, 0) * 255).astype(np.uint8) for k in range(len(dec))], axis=1)
                Image.fromarray(row).save(out / f"ridge_decoded_{tag}.png")
    except Exception as e:
        R["viz_error"] = str(e)[:200]

    (out / "stage1c_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== Stage 1c: TwoRoom LATENT junction, GENUINE bimodal (stochastic branch) ===",
        f"frames={R['n_frames']} pairs={R['n_pairs']} ridge={R['ridge_pairs']} ae_recon={R['ae_recon_mse']:.4f} delta={args.delta}",
        f"branch clf acc at ridge = {R['branch_clf_acc_at_ridge']:.2f} (~0.5 expected = genuinely ambiguous input)",
        f"latent mode separation = {sep:.2f}, within-branch spread = {within:.2f}, sep/within = {sep/within:.2f}",
        "",
        "prediction error to TRUE next-latent at ridge (lower=better):",
        f"  continuous     : {R['cont_pred_err_to_true']:.3f}",
        f"  discrete-anchor: {R['disc_pred_err_to_true']:.3f}",
        "",
        "BLUR aggregate (dist to nearest branch / within-spread; ~1=mode, >>1=blur):",
        f"  continuous     : {R['cont_blur']:.2f}",
        f"  discrete-anchor: {R['disc_blur']:.2f}",
        "",
        f"=== CONTROLLED conditional eval ({cres['n_configs_used']} fixed ridge configs x repeated rolls) ===",
        f"branch separation sep={cres['mean_sep']:.2f}, within-branch spread={cres['mean_within']:.2f}, sep/within={cres['sep_over_within']:.1f}",
        "BETWEEN-NESS = dist to nearest branch / (sep/2)  [THE clean test]:",
        f"  continuous     : {cres['cont_blur']:.2f}   (~1.0 = stuck exactly between the two branches = blur)",
        f"  discrete-anchor: {cres['disc_blur']:.2f}   (~0.0 = committed onto a branch)",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[1c] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
