"""Phase 8g: train the discrete commitment-anchor proposer.

proposer:  (z, g) -> discrete code c -> waypoint  w = z + E[c]
where E is a learned codebook of latent DISPLACEMENTS ("commitment moves").
The code is a dedicated low-info anchor, not a copy of the latent.

Trains on the Phase 8f expert triples (grounded inputs only). Reports val
waypoint MSE to the oracle target against baselines on the SAME metric:
  identity      : w = z                (no move)
  mean_move     : w = z + mean_train(w - z)
  retrieval     : nearest train (z,g) -> its w   (= Phase 8d, intrinsic form)
  continuous    : w = z + MLP(z,g)     (no codebook; upper bound)
  discrete Cxxx : w = z + E[argmax]    (the method)

Saves the trained proposer for the downstream action-quality eval (Phase 8h).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Proposer(nn.Module):
    def __init__(self, dim=192, hidden=512, codebook=256, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.enc = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        if discrete:
            self.to_logits = nn.Linear(hidden, codebook)
            self.codebook = nn.Parameter(torch.randn(codebook, dim) * 0.1)
        else:
            self.to_move = nn.Linear(hidden, dim)

    def forward(self, z, g, tau=1.0, hard=False):
        h = self.enc(torch.cat([z, g], dim=-1))
        if not self.discrete:
            return z + self.to_move(h), None
        logits = self.to_logits(h)
        if self.training and not hard:
            probs = F.gumbel_softmax(logits, tau=tau, hard=False)
        else:
            idx = logits.argmax(-1)
            probs = F.one_hot(idx, logits.size(-1)).float()
        move = probs @ self.codebook
        return z + move, logits


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def retrieval_baseline(z_tr, g_tr, w_tr, z_va, g_va, block=512):
    key_tr = np.concatenate([z_tr, g_tr], 1).astype(np.float32)
    key_va = np.concatenate([z_va, g_va], 1).astype(np.float32)
    ksq = (key_tr * key_tr).sum(1)
    out = np.empty_like(w_tr[:z_va.shape[0]])
    for s in range(0, key_va.shape[0], block):
        q = key_va[s:s + block]
        d2 = (q * q).sum(1)[:, None] - 2 * q @ key_tr.T + ksq[None, :]
        out[s:s + block] = w_tr[d2.argmin(1)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/lghl_phase8f_anchor_data/anchor_triples.npz")
    ap.add_argument("--output-dir", default="outputs/lghl_phase8g_proposer")
    ap.add_argument("--delta", type=int, default=2, choices=[1, 2])
    ap.add_argument("--codebooks", default="64,256,1024")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ent-reg", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    d = np.load(args.data)
    wkey = f"w{args.delta}"
    z_tr, g_tr, w_tr = d["z_train"], d["g_train"], d[f"{wkey}_train"]
    z_va, g_va, w_va = d["z_val"], d["g_val"], d[f"{wkey}_val"]
    dim = z_tr.shape[1]
    print(f"[8g] delta={args.delta} train={z_tr.shape} val={z_va.shape}", flush=True)

    # baselines on val (same metric)
    results = {
        "identity": mse(z_va, w_va),
        "mean_move": mse(z_va + (w_tr - z_tr).mean(0, keepdims=True), w_va),
        "retrieval": mse(retrieval_baseline(z_tr, g_tr, w_tr, z_va, g_va), w_va),
    }
    oracle_mse = 0.0  # oracle == target

    zt = torch.from_numpy(z_tr).to(device); gt = torch.from_numpy(g_tr).to(device)
    wt = torch.from_numpy(w_tr).to(device)
    zv = torch.from_numpy(z_va).to(device); gv = torch.from_numpy(g_va).to(device)
    n = zt.shape[0]

    def train_one(discrete, C):
        model = Proposer(dim=dim, codebook=C, discrete=discrete).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(n, device=device)
            tau = max(0.5, 1.0 * (1 - ep / args.epochs))
            for s in range(0, n, args.batch_size):
                idx = perm[s:s + args.batch_size]
                w_pred, logits = model(zt[idx], gt[idx], tau=tau)
                loss = F.mse_loss(w_pred, wt[idx])
                if discrete and args.ent_reg > 0:
                    p = F.softmax(logits, -1).mean(0)
                    ent = -(p * (p + 1e-9).log()).sum()
                    loss = loss - args.ent_reg * ent  # encourage codebook usage
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            w_pred, logits = model(zv, gv, hard=True)
            val_mse = F.mse_loss(w_pred, torch.from_numpy(w_va).to(device)).item()
            usage = None
            if discrete:
                idx = logits.argmax(-1).cpu().numpy()
                usage = int(len(np.unique(idx)))
        return model, val_mse, usage

    cont_model, cont_mse, _ = train_one(False, 0)
    results["continuous"] = cont_mse
    torch.save({"state": cont_model.state_dict(), "discrete": False, "dim": dim},
               out_dir / f"proposer_continuous_d{args.delta}.pt")

    for C in [int(x) for x in args.codebooks.split(",") if x]:
        m, vmse, usage = train_one(True, C)
        results[f"discrete_C{C}"] = vmse
        torch.save({"state": m.state_dict(), "discrete": True, "codebook": C, "dim": dim},
                   out_dir / f"proposer_discrete_C{C}_d{args.delta}.pt")
        print(f"[8g] discrete C={C}: val_mse={vmse:.4f} codes_used={usage}/{C}", flush=True)

    # recovery vs oracle on the waypoint-MSE metric: 100*(baseline_id - method)/(baseline_id - oracle)
    base = results["identity"]
    rec = {k: 100 * (base - v) / (base - oracle_mse) if base != oracle_mse else float("nan")
           for k, v in results.items()}
    summary = {"delta": args.delta, "val_waypoint_mse": results,
               "pct_closure_to_oracle": rec}
    (out_dir / f"phase8g_proposer_d{args.delta}.json").write_text(json.dumps(summary, indent=2))

    print(f"\n[8g] val waypoint MSE to oracle (delta={args.delta}, lower=better):")
    for k, v in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {k:16s} mse={v:.4f}  closes {rec[k]:5.1f}% of identity->oracle gap")
    print(f"[8g] wrote {out_dir}")


if __name__ == "__main__":
    main()
