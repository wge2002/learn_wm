"""Discrete-operator dynamics vs continuous baseline on the frozen LeWM latent.

Hypothesis (user direction, "solid = anti-diffusion"): putting the DISCRETENESS in
the transition OPERATOR (not in the state -> no latent quantization), with each
autonomous operator spectrally constrained to be non-expansive, makes the open-loop
latent rollout resist the on-manifold isotropic diffusion diagnosed in Phase 6/7.
Continuous map compounds error (analog noise accumulation); a finite set of
contractive operators with a selector behaves like a digital repeater chain --
per-step quantization-ish floor but the error stops compounding.

This is NOT the previously-falsified avenues:
  - not post-hoc projection / snap-back of a frozen latent (Phase 7, dead)
  - not a discrete commitment subgoal added to CEM cost (Phase 8 / overnight, dead)
  - not quantizing the precision-carrying state (the user's explicit constraint)
It is the explicitly-untested "end-to-end discrete DYNAMICS" lever.

Arms (matched param budget, all conditioned on HS-frame latent history + action):
  cont   : z_{t+1} = z_t + MLP([hist, a])
  disc   : z_{t+1} = z_t + sum_c p_c (A_c z_t) + W_a a       (p_c = Gumbel-softmax)
  disc_c : same, each A_c spectral-norm <= 1 (contraction -> non-compounding)

Trains with a multi-step UNROLLED rollout loss (open-loop, predictions fed back) so
"resist drift" is directly in the objective. Evaluates open-loop rollout MSE vs k.

Output: <out>/result.json  with the rollout-MSE-vs-k curve, op spectral radii,
selector usage, and param counts.
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
    """Flatten last HS latents + current action into a conditioning vector."""

    def __init__(self, dim, adim, hs):
        super().__init__()
        self.dim, self.adim, self.hs = dim, adim, hs
        self.in_dim = hs * dim + adim

    def make(self, hist, a):
        # hist: (B, HS, D)  a: (B, A) -> (B, HS*D + A)
        return torch.cat([hist.reshape(hist.size(0), -1), a], dim=-1)


class ContPredictor(nn.Module):
    def __init__(self, dim, adim, hs, hidden=512):
        super().__init__()
        self.h = History(dim, adim, hs)
        self.net = nn.Sequential(
            nn.Linear(self.h.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def step(self, hist, a, tau=1.0, hard=False):
        z = hist[:, -1]
        return z + self.net(self.h.make(hist, a)), {}


class DiscOperatorPredictor(nn.Module):
    """z_{t+1} = z_t + sum_c p_c (A_c z_t) + W_a a.

    Discreteness lives in the autonomous operator A_c (the "how it evolves").
    State stays continuous; action drive W_a a stays continuous (precision).
    contract=True spectral-normalizes each A_c so ||A_c|| <= 1 (anti-compounding).
    """

    def __init__(self, dim, adim, hs, K, hidden=512, contract=False):
        super().__init__()
        self.dim, self.K, self.contract = dim, K, contract
        self.h = History(dim, adim, hs)
        self.selector = nn.Sequential(
            nn.Linear(self.h.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, K),
        )
        # K autonomous operators on z_t; small init around 0 (near-identity delta).
        ops = []
        for _ in range(K):
            lin = nn.Linear(dim, dim, bias=True)
            nn.init.normal_(lin.weight, std=1.0 / (dim ** 0.5) * 0.5)
            nn.init.zeros_(lin.bias)
            if contract:
                lin = nn.utils.parametrizations.spectral_norm(lin, n_power_iterations=1)
            ops.append(lin)
        self.ops = nn.ModuleList(ops)
        self.act_drive = nn.Linear(adim, dim, bias=False)

    def step(self, hist, a, tau=1.0, hard=False):
        z = hist[:, -1]
        logits = self.selector(self.h.make(hist, a))
        if self.training and not hard:
            p = F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            idx = logits.argmax(-1)
            p = F.one_hot(idx, self.K).float()
        # stack operator outputs: (B, K, D)
        outs = torch.stack([op(z) for op in self.ops], dim=1)
        delta = (p.unsqueeze(-1) * outs).sum(dim=1)
        z_next = z + delta + self.act_drive(a)
        return z_next, {"logits": logits, "p": p}


def build_model(arm, dim, adim, hs, K, hidden):
    if arm == "cont":
        return ContPredictor(dim, adim, hs, hidden=hidden)
    if arm == "disc":
        return DiscOperatorPredictor(dim, adim, hs, K, hidden=hidden, contract=False)
    if arm == "disc_c":
        return DiscOperatorPredictor(dim, adim, hs, K, hidden=hidden, contract=True)
    raise ValueError(arm)


# --------------------------------------------------------------------------- #
# rollout helpers
# --------------------------------------------------------------------------- #
def seed_history(z_seq, t0, hs):
    """history of true latents ending at t0, left-padded by repeating z[0]."""
    B = z_seq.size(0)
    idx = [max(0, t0 - hs + 1 + j) for j in range(hs)]
    return z_seq[:, idx, :]  # (B, HS, D)


def unroll(model, z_seq, a_seq, t0, steps, hs, tau=1.0, hard=False, teacher_hist=True):
    """Open-loop predict steps from t0; predictions fed back into history.

    Returns preds (B, steps, D) for z_{t0+1..t0+steps}.
    """
    B, K1, D = z_seq.shape
    hist = list(seed_history(z_seq, t0, hs).unbind(dim=1))  # HS tensors (B, D)
    preds = []
    for s in range(steps):
        t = t0 + s
        a = a_seq[:, t]  # (B, A)
        h = torch.stack(hist[-hs:], dim=1)  # (B, HS, D)
        z_next, _ = model.step(h, a, tau=tau, hard=hard)
        preds.append(z_next)
        hist.append(z_next)
    return torch.stack(preds, dim=1)  # (B, steps, D)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz with z (N,K1,D) and a (N,K,A)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--arm", choices=["cont", "disc", "disc_c"], required=True)
    ap.add_argument("--codebook", type=int, default=8, help="K operators (disc only)")
    ap.add_argument("--unroll", type=int, default=5, help="train rollout length U")
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    configure_torch_threads_from_env()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    d = np.load(args.data)
    z = torch.from_numpy(d["z"].astype(np.float32))  # (N, K1, D)
    a = torch.from_numpy(d["a"].astype(np.float32))  # (N, K,  A)
    N, K1, D = z.shape
    Kh = a.shape[1]
    adim = a.shape[2]
    assert K1 == Kh + 1, f"expect z len = a len + 1, got {K1} vs {Kh}"
    hs = args.hist
    U = min(args.unroll, K1 - 1)

    if "z_val" in d:  # gendata produced an explicit held-out split
        z_tr, a_tr = z.to(device), a.to(device)
        z_va = torch.from_numpy(d["z_val"].astype(np.float32)).to(device)
        a_va = torch.from_numpy(d["a_val"].astype(np.float32)).to(device)
    else:
        g = torch.Generator().manual_seed(args.seed)
        perm = torch.randperm(N, generator=g)
        n_val = max(1, int(N * args.val_frac))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        z_tr, a_tr = z[tr_idx].to(device), a[tr_idx].to(device)
        z_va, a_va = z[val_idx].to(device), a[val_idx].to(device)

    model = build_model(args.arm, D, adim, hs, args.codebook, args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ntr = z_tr.size(0)
    max_t0 = (K1 - 1) - U  # latest start that allows U full steps

    print(f"[{args.arm}] N={N} D={D} A={adim} K={args.codebook} U={U} "
          f"params={n_params/1e6:.2f}M", flush=True)

    for ep in range(args.epochs):
        model.train()
        tau = max(0.5, 1.0 * (1 - ep / args.epochs))
        order = torch.randperm(ntr, device=device)
        ep_loss = 0.0
        nb = 0
        for s in range(0, ntr, args.batch_size):
            bidx = order[s:s + args.batch_size]
            zb, ab = z_tr[bidx], a_tr[bidx]
            # random start so all horizons get trained
            t0 = int(torch.randint(0, max_t0 + 1, (1,)).item()) if max_t0 > 0 else 0
            preds = unroll(model, zb, ab, t0, U, hs, tau=tau, hard=False)
            tgt = zb[:, t0 + 1: t0 + 1 + U]
            loss = F.mse_loss(preds, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} train_loss={ep_loss/max(nb,1):.4f} tau={tau:.2f}",
                  flush=True)

    # ---- eval: open-loop rollout MSE vs k from t0=0 ----
    model.eval()
    with torch.no_grad():
        full = K1 - 1
        preds = unroll(model, z_va, a_va, 0, full, hs, hard=True)  # (Nv, full, D)
        tgt = z_va[:, 1:1 + full]
        per_k_mse = ((preds - tgt) ** 2).mean(dim=(0, 2)).cpu().numpy()  # (full,)
        # natural spread for normalization
        natural_spread = float(z_va.var(dim=0).mean().item())

        # selector usage + operator spectral radius (disc arms)
        usage = None
        spec_radii = None
        if args.arm in ("disc", "disc_c"):
            h0 = seed_history(z_va, 0, hs)
            _, info = model.step(h0, a_va[:, 0], hard=True)
            idx = info["logits"].argmax(-1).cpu().numpy()
            counts = np.bincount(idx, minlength=args.codebook)
            usage = {"codes_used": int((counts > 0).sum()),
                     "K": args.codebook,
                     "hist": counts.tolist()}
            radii = []
            for op in model.ops:
                W = op.weight.detach() if hasattr(op, "weight") else \
                    op.parametrizations.weight.original.detach()
                # spectral radius via top singular value
                sv = torch.linalg.svdvals(W.float())
                radii.append(float(sv.max().item()))
            spec_radii = radii

    result = {
        "arm": args.arm,
        "codebook": args.codebook,
        "unroll": U,
        "hist": hs,
        "seed": args.seed,
        "n_params_M": n_params / 1e6,
        "rollout_mse_vs_k": per_k_mse.tolist(),  # k=1..full
        "natural_spread": natural_spread,
        "rollout_mse_over_spread": (per_k_mse / max(natural_spread, 1e-9)).tolist(),
        "selector_usage": usage,
        "operator_spectral_radii": spec_radii,
        "elapsed_sec": time.time() - t_start,
        "data": str(args.data),
        "N": int(N),
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    tag = f"{args.arm}_K{args.codebook}_U{U}_s{args.seed}"
    print(f"[{tag}] rollout MSE k=1->{full}: "
          f"{per_k_mse[0]:.4f} -> {per_k_mse[-1]:.4f}  "
          f"(spread={natural_spread:.3f})", flush=True)
    if spec_radii is not None:
        print(f"[{tag}] op spectral radii: "
              f"min={min(spec_radii):.2f} max={max(spec_radii):.2f} "
              f"codes_used={usage['codes_used']}/{args.codebook}", flush=True)
    print(f"[{tag}] wrote {out/'result.json'}", flush=True)


if __name__ == "__main__":
    main()
