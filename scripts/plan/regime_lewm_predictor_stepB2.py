"""Step B iteration 2 (faithful): regime-MoE inside LeWM's REAL Transformer predictor.

Closes the Step B caveats that the earlier MLP testbed left open:
  - architecture: uses LeWM's own `Predictor` (Transformer depth6/heads16) + `Embedder`
    action encoder + `pred_proj`, NOT a toy MLP. "mono" here IS LeWM's predictor stage.
  - param scale: ~LeWM level (~11M predictor) instead of 1.6M.
  - training objective: supports single-step teacher-forced (LeWM-native, num_preds=1)
    AND multi-step open-loop unroll, so we test whether our multi-step loss was the
    confound.

Encoder stays FROZEN (we train only the predictor stage on precomputed LeWM latents) —
this isolates the predictor-form question. End-to-end encoder co-adaptation is iteration 2.

Arms: mono | moe | oracle. MoE = gate(latent) -> K param-matched Transformer experts;
oracle replaces the gate with the true contact bin. Metrics: single-step pred MSE,
multi-step rollout mse@k + slope, gate->contact NMI (teacher-forced AND rollout).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from stable_worldmodel.wm.lewm.module import Predictor, Embedder, MLP  # noqa: E402


def cfg_threads():
    raw = os.environ.get("SWM_TORCH_THREADS")
    if raw:
        torch.set_num_threads(max(1, int(raw)))


def make_predictor(dim, depth, heads=16, mlp_dim=2048, dim_head=64, hs=3):
    return Predictor(num_frames=hs, input_dim=dim, hidden_dim=dim, output_dim=dim,
                     depth=depth, heads=heads, mlp_dim=mlp_dim, dim_head=dim_head,
                     dropout=0.0, emb_dropout=0.0)


class MonoLeWM(nn.Module):
    """LeWM's real predictor stage: Embedder(action) + Predictor(Transformer) + pred_proj."""

    def __init__(self, dim, adim, hs, depth=6):
        super().__init__()
        self.act_enc = Embedder(input_dim=adim, smoothed_dim=adim, emb_dim=dim)
        self.predictor = make_predictor(dim, depth, hs=hs)
        self.pred_proj = MLP(dim, hidden_dim=2048, output_dim=dim,
                             norm_fn=lambda d: nn.BatchNorm1d(d))

    def forward(self, ctx, a_win, **kw):
        """ctx (B,T,D) latents, a_win (B,T,adim) raw actions. Returns pred (B,T,D), {}."""
        c = self.act_enc(a_win)
        p = self.predictor(ctx, c)
        p = self.pred_proj(p.reshape(-1, p.size(-1))).reshape_as(p)
        return p, {}


class MoELeWM(nn.Module):
    """gate(latent) -> K param-matched Transformer experts (each a LeWM Predictor)."""

    def __init__(self, dim, adim, hs, K=2, expert_depth=3, gate_hidden=256):
        super().__init__()
        self.K = K
        self.act_enc = Embedder(input_dim=adim, smoothed_dim=adim, emb_dim=dim)
        self.gate = nn.Sequential(nn.Linear(dim, gate_hidden), nn.GELU(),
                                  nn.Linear(gate_hidden, K))
        self.experts = nn.ModuleList(
            [make_predictor(dim, expert_depth, hs=hs) for _ in range(K)])
        self.pred_proj = MLP(dim, hidden_dim=2048, output_dim=dim,
                             norm_fn=lambda d: nn.BatchNorm1d(d))

    def forward(self, ctx, a_win, tau=1.0, hard=False, route=None, soft_eval=False):
        c = self.act_enc(a_win)
        logits = self.gate(ctx)                       # (B,T,K) gate on latent (state)
        if route is not None:
            p = route                                  # (B,T,K) one-hot oracle
        elif self.training and not hard:
            p = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        elif soft_eval:
            p = F.softmax(logits / tau, dim=-1)
        else:
            p = F.one_hot(logits.argmax(-1), self.K).float()
        outs = torch.stack([e(ctx, c) for e in self.experts], dim=2)  # (B,T,K,D)
        out = (p.unsqueeze(-1) * outs).sum(dim=2)                      # (B,T,D)
        out = self.pred_proj(out.reshape(-1, out.size(-1))).reshape_as(out)
        return out, {"logits": logits}


# --------------------------------------------------------------------------- #
# rollout: 3-frame causal windows (LeWM-style). Predicting frame e+1 uses the
# last `hs` latents ending at e and actions aligned to them.
# --------------------------------------------------------------------------- #
def unroll(model, z, a, t0, steps, hs, tau=1.0, hard=False, route_full=None,
           soft_eval=False, want_logits=False):
    """Open-loop: feed predictions back. Returns preds (B,steps,D)[, logits]."""
    hist = list(z[:, t0:t0 + hs].unbind(dim=1))      # hs true seed frames
    preds, logit_list = [], []
    for s in range(steps):
        e = t0 + hs - 1 + s                           # current last true/pred index
        ctx = torch.stack(hist[-hs:], dim=1)          # (B,hs,D)
        a_win = a[:, e - hs + 1:e + 1]                # actions aligned to ctx
        route = None
        if route_full is not None:
            route = route_full[:, e - hs + 1:e + 1]   # (B,hs,K)
        out, info = model(ctx, a_win, tau=tau, hard=hard, route=route,
                          soft_eval=soft_eval)
        nxt = out[:, -1]                              # prediction for e+1
        preds.append(nxt); hist.append(nxt)
        if want_logits and "logits" in info:
            logit_list.append(info["logits"][:, -1])
    preds = torch.stack(preds, dim=1)
    if want_logits:
        return preds, (torch.stack(logit_list, dim=1) if logit_list else None)
    return preds


def single_step_loss(model, z, a, hs, lab=None, gate_sup=0.0, route_full=None,
                     tau=1.0):
    """Teacher-forced: every length-hs window predicts the next frame (LeWM-native)."""
    B, L, D = z.shape
    losses, ces = [], []
    for t in range(0, L - hs):
        ctx = z[:, t:t + hs]
        a_win = a[:, t:t + hs]
        tgt = z[:, t + 1:t + hs + 1]
        route = route_full[:, t:t + hs] if route_full is not None else None
        out, info = model(ctx, a_win, tau=tau, hard=False, route=route)
        losses.append(F.mse_loss(out, tgt))
        if gate_sup > 0 and "logits" in info and lab is not None:
            lg = info["logits"].reshape(-1, info["logits"].size(-1))
            ces.append(F.cross_entropy(lg, lab[:, t + 1:t + hs + 1].reshape(-1)))
    loss = torch.stack(losses).mean()
    if ces:
        loss = loss + gate_sup * torch.stack(ces).mean()
    return loss


@torch.no_grad()
def routing_nmi(model, z, a, lab, hs, rollout):
    """gate code vs contact; rollout=False teacher-forced, True open-loop drifted."""
    model.eval()
    if not isinstance(model, MoELeWM):
        return None
    codes, labels = [], []
    if not rollout:
        for t in range(0, z.size(1) - hs):
            _, info = model(z[:, t:t + hs], a[:, t:t + hs], hard=True)
            codes.append(info["logits"][:, -1].argmax(-1).cpu().numpy())
            labels.append(lab[:, t + hs].cpu().numpy())
    else:
        full = z.size(1) - hs
        _, lg = unroll(model, z, a, 0, full, hs, hard=True, want_logits=True)
        codes_arr = lg.argmax(-1).cpu().numpy()           # (B,full)
        labels_arr = lab[:, hs:hs + full].cpu().numpy()
        codes = [codes_arr.flatten()]; labels = [labels_arr.flatten()]
    codes = np.concatenate(codes); labels = np.concatenate(labels)
    nmi = float(normalized_mutual_info_score(labels, codes))
    acc = (labels == codes).mean(); acc = float(max(acc, 1 - acc))
    return {"nmi": nmi, "acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/regime_stepB/train_contact.npz")
    ap.add_argument("--arm", choices=["mono", "moe", "oracle"], required=True)
    ap.add_argument("--train-mode", choices=["single", "multi"], default="multi")
    ap.add_argument("--gate-sup", type=float, default=0.0)
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--mono-depth", type=int, default=6)
    ap.add_argument("--expert-depth", type=int, default=3)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg_threads()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    d = np.load(args.data)
    z = torch.from_numpy(d["z"].astype(np.float32))
    a = torch.from_numpy(d["a"].astype(np.float32))
    lab_np = (d["contact_frac"] > 0.0).astype(np.int64)
    lab = torch.from_numpy(lab_np)
    N, L, D = z.shape; adim = a.shape[2]; hs = args.hist; K = args.experts
    U = min(args.unroll, L - hs)

    route_all = None
    if args.arm == "oracle":
        r = np.zeros((*lab_np.shape, K), np.float32)
        np.put_along_axis(r, lab_np[..., None], 1.0, axis=-1)
        route_all = torch.from_numpy(r)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g); nv = max(1, int(N * 0.1))
    tr, va = perm[nv:], perm[:nv]
    z_tr, a_tr, lab_tr = z[tr].to(device), a[tr].to(device), lab[tr].to(device)
    z_va, a_va, lab_va = z[va].to(device), a[va].to(device), lab[va].to(device)
    route_tr = route_all[tr].to(device) if route_all is not None else None
    # eval uses learned gate for moe; oracle uses true route at eval too
    route_va = route_all[va].to(device) if route_all is not None else None

    if args.arm == "mono":
        model = MonoLeWM(D, adim, hs, depth=args.mono_depth).to(device)
    else:
        model = MoELeWM(D, adim, hs, K=K, expert_depth=args.expert_depth).to(device)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    ntr = z_tr.size(0); max_t0 = (L - hs) - U
    print(f"[{args.arm}/{args.train_mode}] N={N} D={D} adim={adim} K={K} "
          f"params={nparams:.2f}M U={U}", flush=True)

    for ep in range(args.epochs):
        model.train()
        tau = max(0.5, 1.0 * (1 - ep / args.epochs))
        order = torch.randperm(ntr, device=device)
        tot, nb = 0.0, 0
        for s in range(0, ntr, args.batch_size):
            bi = order[s:s + args.batch_size]
            zb, ab = z_tr[bi], a_tr[bi]
            rb = route_tr[bi] if route_tr is not None else None
            lb = lab_tr[bi]
            if args.train_mode == "single":
                loss = single_step_loss(model, zb, ab, hs, lab=lb,
                                        gate_sup=args.gate_sup, route_full=rb, tau=tau)
            else:
                tt = int(torch.randint(0, max_t0 + 1, (1,)).item()) if max_t0 > 0 else 0
                want = args.gate_sup > 0
                res = unroll(model, zb, ab, tt, U, hs, tau=tau, hard=False,
                             route_full=rb, want_logits=want)
                preds, logits = res if want else (res, None)
                tgt = zb[:, tt + hs:tt + hs + U]
                loss = F.mse_loss(preds, tgt)
                if want and logits is not None:
                    lt = lb[:, tt + hs:tt + hs + U]
                    loss = loss + args.gate_sup * F.cross_entropy(
                        logits.reshape(-1, K), lt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} loss={tot/max(nb,1):.4f}", flush=True)

    # ---- eval ----
    model.eval()
    with torch.no_grad():
        full = L - hs
        preds = unroll(model, z_va, a_va, 0, full, hs, hard=True, route_full=route_va)
        per_k = ((preds - z_va[:, hs:hs + full]) ** 2).mean(dim=(0, 2)).cpu().numpy()
        # single-step (teacher-forced) MSE
        ss = []
        for t in range(0, L - hs):
            o, _ = model(z_va[:, t:t + hs], a_va[:, t:t + hs], hard=True,
                         route=(route_va[:, t:t + hs] if route_va is not None else None))
            ss.append(((o[:, -1] - z_va[:, t + hs]) ** 2).mean().item())
        ss_mse = float(np.mean(ss))
    x = np.arange(1, len(per_k) + 1)
    slope = float(np.polyfit(x, per_k, 1)[0])
    tf = routing_nmi(model, z_va, a_va, lab_va, hs, rollout=False)
    ro = routing_nmi(model, z_va, a_va, lab_va, hs, rollout=True)

    result = dict(arm=args.arm, train_mode=args.train_mode, gate_sup=args.gate_sup,
                  experts=K, params_M=nparams, unroll=U, hist=hs, seed=args.seed,
                  single_step_mse=ss_mse, rollout_mse_vs_k=per_k.tolist(),
                  mse_at_full=float(per_k[-1]), slope=slope,
                  routing_tf=tf, routing_rollout=ro,
                  elapsed_sec=time.time() - t0, data=str(args.data), N=int(N))
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[{args.arm}/{args.train_mode} gs{args.gate_sup}] params={nparams:.2f}M "
          f"ss_mse={ss_mse:.4f} mse@{full}={per_k[-1]:.4f} slope={slope:.4f} "
          f"tf={tf} ro={ro} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
