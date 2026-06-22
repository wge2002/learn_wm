"""Stage 1b: lift the multimodal-junction demo to a LATENT world model.

Stage 1 proved in state space that at a multimodal junction continuous
regression mean-blurs (into the wall) while a discrete commitment recovers the
oracle. This script asks the same question in a learned LATENT space (the real
world-model setting):

  1. render TwoRoom images, train a small conv auto-encoder -> latent z.
  2. continuous latent predictor : (z_t, z_goal) -> z_{t+d}     (L2)
  3. discrete-anchor predictor   : (z_t, z_goal) -> branch (commit) -> z_{t+d}
  4. at junction (ridge) transitions, the true z_{t+d} forms TWO clusters
     (up-door vs down-door). Measure whether the continuous prediction lands
     BETWEEN the clusters (off-manifold blur) and whether the discrete anchor
     commits to a cluster. Decode predictions to images for a visual.

Fully self-contained; writes outputs/tworoom_stage1b/RESULT.txt at the end.
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


def door_label(agent, goal):
    d0 = np.linalg.norm(agent - DOORS[0], axis=-1)
    d1 = np.linalg.norm(agent - DOORS[1], axis=-1)
    return (d1 < d0).astype(np.int64)


def resize64(img_hwc_uint8):
    t = torch.from_numpy(img_hwc_uint8).permute(2, 0, 1).float().div_(255.0)[None]
    return F.interpolate(t, size=(64, 64), mode="bilinear", align_corners=False)[0]


def gen_images(episodes, max_steps, delta, seed, log):
    from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy
    rng = np.random.default_rng(seed)
    env = gym.make("swm/TwoRoom-v1")
    pol = ExpertPolicy(action_noise=0.05, seed=seed); pol.set_env(env)
    imgs, goalimgs, agents, goals, eps, steps = [], [], [], [], [], []
    ep = 0
    for e in range(episodes):
        ax = rng.uniform(LO, WALL_X - 18); ay = rng.uniform(LO, HI)
        tx = rng.uniform(WALL_X + 18, HI); ty = rng.uniform(LO, HI)
        if rng.random() < 0.5:
            ax, tx = tx, ax
        obs, _ = env.reset(seed=int(rng.integers(1 << 30)),
                           options={"variation_values": VV, "state": [float(ax), float(ay)],
                                    "target_state": [float(tx), float(ty)]})
        obs = np.asarray(obs, np.float32); goal = obs[2:4].copy()
        # goal image = agent rendered at target (env caches it as _target_img, CHW)
        u = env.unwrapped
        gchw = u._target_img.float()
        if float(gchw.max()) > 1.5:
            gchw = gchw / 255.0
        gimg = F.interpolate(gchw[None], size=(64, 64), mode="bilinear",
                             align_corners=False)[0].cpu()
        for t in range(max_steps):
            img = resize64(env.render().copy())
            imgs.append(img); goalimgs.append(gimg); agents.append(obs[:2].copy())
            goals.append(goal.copy()); eps.append(ep); steps.append(t)
            a = np.asarray(pol.get_action({"state": obs[:2], "goal_state": goal}), np.float32).squeeze()
            obs, r, term, trunc, info = env.step(a); obs = np.asarray(obs, np.float32)
            if term:
                break
        ep += 1
        if (e + 1) % 200 == 0:
            print(f"[1b] rendered {e+1}/{episodes} eps, {len(imgs)} frames", flush=True)
    imgs = torch.stack(imgs); goalimgs = torch.stack(goalimgs)
    agents = np.array(agents, np.float32); goals = np.array(goals, np.float32)
    eps = np.array(eps); steps = np.array(steps)
    # build (t, t+delta) pairs within episode
    pairs = []
    for i in range(len(eps) - delta):
        if eps[i] == eps[i + delta] and steps[i + delta] == steps[i] + delta:
            pairs.append((i, i + delta))
    pairs = np.array(pairs)
    print(f"[1b] {len(imgs)} frames, {len(pairs)} (t,t+{delta}) pairs", flush=True)
    return imgs, goalimgs, agents, goals, pairs


class ConvAE(nn.Module):
    def __init__(self, zdim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),   # 32
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),  # 16
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),  # 8
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),  # 4
            nn.Flatten(), nn.Linear(64 * 16, zdim))
        self.dec_fc = nn.Linear(zdim, 64 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(self.dec_fc(z).view(-1, 64, 4, 4))

    def forward(self, x):
        z = self.encode(x); return self.decode(z), z


class MLP(nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h),
                                 nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/tworoom_stage1b")
    ap.add_argument("--episodes", type=int, default=2500)
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--delta", type=int, default=3)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--ae-epochs", type=int, default=15)
    ap.add_argument("--pred-epochs", type=int, default=80)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    R = {}

    print("[1b] generating images...", flush=True)
    imgs, goalimgs, agents, goals, pairs = gen_images(
        args.episodes, args.max_steps, args.delta, args.seed, out)
    R["n_frames"] = int(len(imgs)); R["n_pairs"] = int(len(pairs))

    # --- train conv AE ---
    print("[1b] training conv AE...", flush=True)
    ae = ConvAE(args.zdim).to(dev)
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    N = len(imgs); bs = 256
    for ep in range(args.ae_epochs):
        perm = torch.randperm(N)
        tot = 0
        for s in range(0, N, bs):
            xb = imgs[perm[s:s+bs]].to(dev)
            rec, _ = ae(xb); loss = F.mse_loss(rec, xb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if ep % 5 == 0:
            print(f"[1b]  ae ep{ep} recon={tot/(N//bs):.4f}", flush=True)
    R["ae_recon_mse"] = float(tot / (N // bs))

    # --- encode all ---
    ae.eval()
    with torch.no_grad():
        Z = torch.cat([ae.encode(imgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)])
        Zg = torch.cat([ae.encode(goalimgs[s:s+512].to(dev)).cpu() for s in range(0, N, 512)])
    Z = Z.numpy(); Zg = Zg.numpy()
    # standardize latents (per-dim) so distances are in consistent units
    mu = Z.mean(0); sd = Z.std(0) + 1e-6
    Z = (Z - mu) / sd; Zg = (Zg - mu) / sd

    ti, tj = pairs[:, 0], pairs[:, 1]
    zt, zg, zw = Z[ti], Zg[ti], Z[tj]
    X = np.concatenate([zt, zg], 1).astype(np.float32)
    branch = door_label(agents[ti], goals[ti])  # 0/1 supervision for discrete anchor
    cross = (agents[ti, 0] - WALL_X) * (goals[ti, 0] - WALL_X) < 0

    def train(model, Xin, Yin, loss_fn, epochs):
        o = torch.optim.Adam(model.parameters(), 1e-3)
        Xt = torch.tensor(Xin, device=dev); Yt = torch.tensor(Yin, device=dev)
        n = len(Xt)
        for ep in range(epochs):
            pm = torch.randperm(n, device=dev)
            for s in range(0, n, 1024):
                idx = pm[s:s+1024]
                loss = loss_fn(model(Xt[idx]), Yt[idx])
                o.zero_grad(); loss.backward(); o.step()
        return model

    print("[1b] training latent predictors...", flush=True)
    cont = train(MLP(2*args.zdim, args.zdim).to(dev), X, zw, nn.MSELoss(), args.pred_epochs)
    # discrete anchor: classifier on cross-room + per-branch latent head
    clf = train(MLP(2*args.zdim, 2).to(dev), X[cross], branch[cross],
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=dev).long()), args.pred_epochs)
    # per-branch waypoint head: input (z_t,z_goal,branch_onehot) -> z_w
    Xb = np.concatenate([X, np.eye(2, dtype=np.float32)[branch]], 1)
    head = train(MLP(2*args.zdim + 2, args.zdim).to(dev), Xb, zw, nn.MSELoss(), args.pred_epochs)

    # --- eval at ridge ---
    ridge = cross & (np.abs(agents[ti, 1] - WALL_X) < 10)
    R["ridge_pairs"] = int(ridge.sum())
    # two true branch cluster centroids at ridge (in latent), per branch label
    rb = branch[ridge]; rzw = zw[ridge]
    cen = np.stack([rzw[rb == 0].mean(0), rzw[rb == 1].mean(0)])  # (2, z)
    with torch.no_grad():
        Xr = torch.tensor(X[ridge], device=dev)
        cont_pred = cont(Xr).cpu().numpy()
        bsel = clf(Xr).argmax(-1).cpu().numpy()
        Xrb = np.concatenate([X[ridge], np.eye(2, dtype=np.float32)[bsel]], 1)
        disc_pred = head(torch.tensor(Xrb, device=dev)).cpu().numpy()

    def dist_to_nearest_mode(P):
        d0 = np.linalg.norm(P - cen[0], axis=1); d1 = np.linalg.norm(P - cen[1], axis=1)
        return np.minimum(d0, d1)
    sep = float(np.linalg.norm(cen[0] - cen[1]))  # mode separation
    # within-branch spread = typical distance of true samples to their own centroid
    within = float(np.mean([
        np.linalg.norm(rzw[rb == b] - cen[b], axis=1).mean()
        for b in (0, 1) if (rb == b).sum() > 0]))
    R["mode_separation"] = sep
    R["within_branch_spread"] = within
    R["cont_pred_err_to_true"] = float(np.mean(np.linalg.norm(cont_pred - rzw, axis=1)))
    R["disc_pred_err_to_true"] = float(np.mean(np.linalg.norm(disc_pred - rzw, axis=1)))
    R["cont_dist_to_nearest_mode"] = float(np.mean(dist_to_nearest_mode(cont_pred)))
    R["disc_dist_to_nearest_mode"] = float(np.mean(dist_to_nearest_mode(disc_pred)))
    # blur = how far the prediction sits from the nearest real branch, in units of
    # within-branch spread. ~1 = lands on a mode; >>1 = stuck between modes (blur).
    R["cont_blur"] = R["cont_dist_to_nearest_mode"] / (within + 1e-9)
    R["disc_blur"] = R["disc_dist_to_nearest_mode"] / (within + 1e-9)
    R["branch_clf_acc"] = float((bsel == rb).mean())

    # --- decode a few predictions for a visual ---
    try:
        import PIL.Image as Image
        with torch.no_grad():
            sampleidx = np.where(ridge)[0][:8]
            for tag, P in [("cont", cont_pred[:8]), ("disc", disc_pred[:8]),
                           ("true", rzw[:8])]:
                dec = ae.decode(torch.tensor(P, device=dev)).cpu().numpy()
                row = np.concatenate([(dec[k].transpose(1, 2, 0) * 255).astype(np.uint8)
                                      for k in range(len(dec))], axis=1)
                Image.fromarray(row).save(out / f"ridge_decoded_{tag}.png")
    except Exception as e:
        R["viz_error"] = str(e)[:200]

    (out / "stage1b_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== Stage 1b: TwoRoom LATENT multimodal junction ===",
        f"frames={R['n_frames']} pairs={R['n_pairs']} ridge_pairs={R['ridge_pairs']} ae_recon={R['ae_recon_mse']:.4f}",
        f"latent mode separation (up vs down branch) = {sep:.3f}",
        f"branch classifier acc at ridge = {R['branch_clf_acc']:.2f}",
        "",
        "prediction error to TRUE next-latent at ridge (lower=better):",
        f"  continuous     : {R['cont_pred_err_to_true']:.3f}",
        f"  discrete-anchor: {R['disc_pred_err_to_true']:.3f}",
        "",
        f"BLUR test - dist to nearest real branch / within-branch spread "
        f"(sep={sep:.2f} within={within:.2f}; ~1=on a mode, >>1=between modes=blur):",
        f"  continuous     : {R['cont_blur']:.2f}",
        f"  discrete-anchor: {R['disc_blur']:.2f}",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[1b] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
