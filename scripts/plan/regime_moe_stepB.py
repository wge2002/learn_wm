"""Step B (natural emergence / readout) for the dynamic-discrete-regime direction.

Direction doc: docs/knowledge/direction_discrete_regime_from_lewm.md

Step A established (GREEN): a discrete regime lives in the trained LeWM transition f
(its local Jacobian), aligned to contact, while the state z is structureless; and the
clean regime signal is in the *operator/state*, NOT the action-confounded residual.

Step B turns that into a usable latent dynamics model and asks whether regime-
conditioning helps:

  mono : z_{t+1} = z_t + MLP([hist, a])                          (monolithic baseline)
  moe  : z_{t+1} = z_t + sum_k g_k(hist_state) * f_k([hist, a])  (piecewise / MoE)
         gate g_k conditions on the STATE history only (per Step A); experts f_k are
         full MLPs that see the action. Gumbel-softmax during training, hard at eval.

Trained with a multi-step UNROLLED latent-MSE rollout loss (the LeWM objective; NO
z-clustering loss added -- regime must emerge from dynamics fit alone).

Go/kill (from the doc):
  moe flattens the mse@k slope (anti-drift) AND its gate aligns with contact
  unsupervised (purity / NMI up vs a contact-blind baseline) -> Step B lands.
  Slope not flattened -> method dies.

Output: <out>/result.json with rollout-MSE-vs-k (mono needs its own run), slope,
gate usage, and gate-vs-contact purity/NMI on the contact-labeled eval set.
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


def configure_torch_threads_from_env() -> None:
    raw = os.environ.get("SWM_TORCH_THREADS")
    if not raw:
        return
    n = max(1, int(raw))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class History(nn.Module):
    def __init__(self, dim, adim, hs):
        super().__init__()
        self.dim, self.adim, self.hs = dim, adim, hs
        self.in_dim = hs * dim + adim
        self.state_dim = hs * dim

    def make(self, hist, a):
        return torch.cat([hist.reshape(hist.size(0), -1), a], dim=-1)

    def state(self, hist):
        return hist.reshape(hist.size(0), -1)


class MonoPredictor(nn.Module):
    def __init__(self, dim, adim, hs, hidden=512):
        super().__init__()
        self.h = History(dim, adim, hs)
        self.net = nn.Sequential(
            nn.Linear(self.h.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def step(self, hist, a, tau=1.0, hard=False, route=None):
        z = hist[:, -1]
        return z + self.net(self.h.make(hist, a)), {}


class MoEPredictor(nn.Module):
    """z_{t+1} = z_t + sum_k g_k(state_hist) * f_k([hist, a]).

    Gate sees STATE history only (Step A: regime is a state/operator property, the
    residual is action-confounded). Each expert is a full MLP that sees the action.
    """

    def __init__(self, dim, adim, hs, K, hidden=512, gate_input="state"):
        super().__init__()
        self.dim, self.K, self.gate_input = dim, K, gate_input
        self.h = History(dim, adim, hs)
        gate_in = self.h.state_dim if gate_input == "state" else self.h.in_dim
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.h.in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, dim),
            ) for _ in range(K)
        ])

    def step(self, hist, a, tau=1.0, hard=False, route=None):
        """route: optional (B,K) one-hot to override the gate (oracle regime)."""
        z = hist[:, -1]
        gin = self.h.state(hist) if self.gate_input == "state" else self.h.make(hist, a)
        logits = self.gate(gin)
        if route is not None:
            p = route
        elif self.training and not hard:
            p = F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            p = F.one_hot(logits.argmax(-1), self.K).float()
        feat = self.h.make(hist, a)
        outs = torch.stack([e(feat) for e in self.experts], dim=1)  # (B,K,D)
        delta = (p.unsqueeze(-1) * outs).sum(dim=1)
        return z + delta, {"logits": logits, "p": p}


def build_model(arm, dim, adim, hs, K, hidden, gate_input="state"):
    if arm == "mono":
        return MonoPredictor(dim, adim, hs, hidden=hidden)
    if arm in ("moe", "oracle"):
        # oracle reuses the MoE experts but is routed by ground-truth contact
        return MoEPredictor(dim, adim, hs, K, hidden=hidden, gate_input=gate_input)
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
# rollout helpers (open-loop unroll, predictions fed back)
# --------------------------------------------------------------------------- #
def seed_history(z_seq, t0, hs):
    idx = [max(0, t0 - hs + 1 + j) for j in range(hs)]
    return z_seq[:, idx, :]


def unroll(model, z_seq, a_seq, t0, steps, hs, tau=1.0, hard=False,
           route_seq=None):
    """route_seq: optional (B, K1, K) one-hot oracle routing per timestep; step s
    predicting time t+1 is routed by route_seq[:, t+1]."""
    hist = list(seed_history(z_seq, t0, hs).unbind(dim=1))
    preds = []
    for s in range(steps):
        t = t0 + s
        h = torch.stack(hist[-hs:], dim=1)
        route = route_seq[:, t + 1] if route_seq is not None else None
        z_next, _ = model.step(h, a_seq[:, t], tau=tau, hard=hard, route=route)
        preds.append(z_next)
        hist.append(z_next)
    return torch.stack(preds, dim=1)


def linfit_slope(y):
    x = np.arange(1, len(y) + 1, dtype=np.float64)
    return float(np.polyfit(x, np.asarray(y, dtype=np.float64), 1)[0])


# --------------------------------------------------------------------------- #
# gate-vs-contact purity on the contact-labeled eval set
# --------------------------------------------------------------------------- #
@torch.no_grad()
def gate_contact_alignment(model, z_ev, a_ev, contact, hs, thresh):
    """For each interior transition, hard gate assignment vs contact label.
    Returns purity, NMI, codes_used, per-code contact-rate spread."""
    model.eval()
    K1 = z_ev.size(1)
    labels, codes = [], []
    for k in range(hs - 1, a_ev.size(1)):
        h = z_ev[:, k - hs + 1:k + 1]
        _, info = model.step(h, a_ev[:, k], hard=True)
        codes.append(info["logits"].argmax(-1).cpu().numpy())
        labels.append((contact[:, k + 1] > thresh).astype(int))
    codes = np.concatenate(codes)
    labels = np.concatenate(labels)
    purity = 0.0
    crate = []
    for c in np.unique(codes):
        m = codes == c
        vals, cnts = np.unique(labels[m], return_counts=True)
        purity += cnts.max()
        crate.append(float(labels[m].mean()))
    purity = float(purity / len(codes))
    nmi = float(normalized_mutual_info_score(labels, codes))
    return {
        "gate_contact_purity": purity,
        "gate_contact_nmi": nmi,
        "codes_used": int(len(np.unique(codes))),
        "contact_rate": float(labels.mean()),
        "code_contact_rate_min": float(min(crate)),
        "code_contact_rate_max": float(max(crate)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="train npz: z (N,K1,D), a (N,K,A)")
    ap.add_argument("--eval-contact", default=None,
                    help="npz with z,a,contact_frac for gate-contact alignment")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arm", choices=["mono", "moe", "oracle"], required=True)
    ap.add_argument("--experts", type=int, default=3, help="K (moe/oracle)")
    ap.add_argument("--gate-input", choices=["state", "both"], default="state",
                    help="moe gate sees state-history only, or state+action")
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--contact-thresh", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    configure_torch_threads_from_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    d = np.load(args.data)
    z = torch.from_numpy(d["z"].astype(np.float32))
    a = torch.from_numpy(d["a"].astype(np.float32))
    N, K1, D = z.shape
    adim = a.shape[2]
    hs = args.hist
    U = min(args.unroll, K1 - 1)
    # oracle routing needs ground-truth contact bins on train + val (binary, K=2)
    route_all = None
    if args.arm == "oracle":
        assert args.experts == 2, "oracle uses binary contact regime (K=2)"
        assert "contact_frac" in d, "oracle needs contact_frac in --data"
        lab = (d["contact_frac"] > args.contact_thresh).astype(np.int64)  # (N,K1)
        r = np.zeros((*lab.shape, args.experts), dtype=np.float32)
        np.put_along_axis(r, lab[..., None], 1.0, axis=-1)
        route_all = torch.from_numpy(r)

    if "z_val" in d:
        z_tr, a_tr = z.to(device), a.to(device)
        z_va = torch.from_numpy(d["z_val"].astype(np.float32)).to(device)
        a_va = torch.from_numpy(d["a_val"].astype(np.float32)).to(device)
        route_tr = route_va = None
        if route_all is not None:  # only single-set oracle supported here
            raise ValueError("oracle expects a single contact-labeled --data (no z_val)")
    else:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(N, generator=g)
        nv = max(1, int(N * 0.1))
        tr_i, va_i = perm[nv:], perm[:nv]
        z_tr, a_tr = z[tr_i].to(device), a[tr_i].to(device)
        z_va, a_va = z[va_i].to(device), a[va_i].to(device)
        route_tr = route_all[tr_i].to(device) if route_all is not None else None
        route_va = route_all[va_i].to(device) if route_all is not None else None

    model = build_model(args.arm, D, adim, hs, args.experts, args.hidden,
                        gate_input=args.gate_input).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ntr = z_tr.size(0)
    max_t0 = (K1 - 1) - U
    print(f"[{args.arm}] N={N} D={D} A={adim} K={args.experts} U={U} "
          f"params={n_params/1e6:.2f}M", flush=True)

    for ep in range(args.epochs):
        model.train()
        tau = max(0.5, 1.0 * (1 - ep / args.epochs))
        order = torch.randperm(ntr, device=device)
        ep_loss, nb = 0.0, 0
        for s in range(0, ntr, args.batch_size):
            bidx = order[s:s + args.batch_size]
            zb, ab = z_tr[bidx], a_tr[bidx]
            t0 = int(torch.randint(0, max_t0 + 1, (1,)).item()) if max_t0 > 0 else 0
            rb = route_tr[bidx] if route_tr is not None else None
            preds = unroll(model, zb, ab, t0, U, hs, tau=tau, hard=False, route_seq=rb)
            loss = F.mse_loss(preds, zb[:, t0 + 1: t0 + 1 + U])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item(); nb += 1
        if ep % 15 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} loss={ep_loss/max(nb,1):.4f} tau={tau:.2f}", flush=True)

    # ---- eval: open-loop rollout mse vs k ----
    model.eval()
    with torch.no_grad():
        full = K1 - 1
        preds = unroll(model, z_va, a_va, 0, full, hs, hard=True, route_seq=route_va)
        per_k = ((preds - z_va[:, 1:1 + full]) ** 2).mean(dim=(0, 2)).cpu().numpy()
        spread = float(z_va.var(dim=0).mean().item())
        usage = None
        if args.arm == "moe":
            h0 = seed_history(z_va, 0, hs)
            _, info = model.step(h0, a_va[:, 0], hard=True)
            idx = info["logits"].argmax(-1).cpu().numpy()
            usage = {"codes_used": int(len(np.unique(idx))), "K": args.experts,
                     "hist": np.bincount(idx, minlength=args.experts).tolist()}

    align = None
    if args.eval_contact and args.arm == "moe":
        de = np.load(args.eval_contact)
        z_ev = torch.from_numpy(de["z"].astype(np.float32)).to(device)
        a_ev = torch.from_numpy(de["a"].astype(np.float32)).to(device)
        contact = de["contact_frac"]
        align = gate_contact_alignment(model, z_ev, a_ev, contact, hs,
                                       args.contact_thresh)

    result = {
        "arm": args.arm, "experts": args.experts, "gate_input": args.gate_input,
        "unroll": U, "hist": hs,
        "seed": args.seed, "n_params_M": n_params / 1e6,
        "rollout_mse_vs_k": per_k.tolist(),
        "rollout_mse_slope": linfit_slope(per_k),
        "natural_spread": spread,
        "rollout_mse_over_spread": (per_k / max(spread, 1e-9)).tolist(),
        "gate_usage": usage,
        "gate_contact_alignment": align,
        "elapsed_sec": time.time() - t_start,
        "data": str(args.data),
        "N": int(N),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    tag = f"{args.arm}_K{args.experts}_U{U}_s{args.seed}"
    print(f"[{tag}] mse k1->{full}: {per_k[0]:.4f}->{per_k[-1]:.4f} "
          f"slope={result['rollout_mse_slope']:.4f} (spread={spread:.3f})", flush=True)
    if align:
        print(f"[{tag}] gate-contact purity={align['gate_contact_purity']:.3f} "
              f"nmi={align['gate_contact_nmi']:.3f} codes={align['codes_used']}/"
              f"{args.experts} crate[{align['code_contact_rate_min']:.2f},"
              f"{align['code_contact_rate_max']:.2f}]", flush=True)
    print(f"[{tag}] wrote {out/'result.json'}", flush=True)


if __name__ == "__main__":
    main()
