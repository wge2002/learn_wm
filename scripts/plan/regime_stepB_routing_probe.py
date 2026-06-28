"""Step B routing probe — close two holes raised in review.

(1) Mechanism check: our verdict claims specialized experts fail because the
    gate misroutes on DRIFTED rollout states. We only ever measured gate->contact
    alignment on TRUE (teacher-forced) windows (NMI up to 0.55). Here we measure
    routing on the OPEN-LOOP rollout (gate sees its own drifted predictions) and
    compare. If rollout-routing NMI << teacher-forced NMI, the mechanism holds.

(2) Train-objective faithfulness: LeWM trains single-step teacher-forced
    (num_preds=1). We trained multi-step open-loop (U=5), which may itself make
    the gate hard to learn. We re-run with U=1 (LeWM-like) to see if the gate
    learns better / MoE helps.

Reuses the Step B trainer's model + unroll. One seed, ~11s/config; this is a
diagnostic, not a full sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regime_moe_stepB as B  # noqa: E402


def best_perm_acc(labels, codes, K):
    """K=2 routing accuracy under the better of identity / flip code<->label map."""
    acc = (labels == codes).mean()
    return float(max(acc, 1.0 - acc))


@torch.no_grad()
def teacher_forced_routing(model, z, a, lab, hs):
    """gate code on TRUE windows vs contact label (no drift)."""
    model.eval()
    codes, labels = [], []
    for k in range(hs - 1, a.size(1)):
        h = z[:, k - hs + 1:k + 1]
        _, info = model.step(h, a[:, k], hard=True)
        codes.append(info["logits"].argmax(-1).cpu().numpy())
        labels.append(lab[:, k + 1].cpu().numpy())
    codes = np.concatenate(codes); labels = np.concatenate(labels)
    return (float(normalized_mutual_info_score(labels, codes)),
            best_perm_acc(labels, codes, model.K))


@torch.no_grad()
def rollout_routing(model, z, a, lab, hs):
    """gate code along the OPEN-LOOP rollout (gate sees drifted predictions)."""
    model.eval()
    full = z.size(1) - 1
    hist = list(B.seed_history(z, 0, hs).unbind(dim=1))
    codes = []
    for s in range(full):
        h = torch.stack(hist[-hs:], dim=1)
        z_next, info = model.step(h, a[:, s], hard=True)
        codes.append(info["logits"].argmax(-1).cpu().numpy())
        hist.append(z_next)
    codes = np.stack(codes, axis=1)              # (n, full), routing on drifted hist
    labels = lab[:, 1:full + 1].cpu().numpy()    # contact at the target step
    return (float(normalized_mutual_info_score(labels.flatten(), codes.flatten())),
            best_perm_acc(labels.flatten(), codes.flatten(), model.K))


def train_one(z_tr, a_tr, lab_tr, D, adim, hs, U, gate_sup, epochs, lr, bs, device, K=2):
    model = B.MoEPredictor(D, adim, hs, K, hidden=512, gate_input="state").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ntr = z_tr.size(0)
    max_t0 = (z_tr.size(1) - 1) - U
    for ep in range(epochs):
        model.train()
        tau = max(0.5, 1.0 * (1 - ep / epochs))
        order = torch.randperm(ntr, device=device)
        for s in range(0, ntr, bs):
            bidx = order[s:s + bs]
            zb, ab = z_tr[bidx], a_tr[bidx]
            t0 = int(torch.randint(0, max_t0 + 1, (1,)).item()) if max_t0 > 0 else 0
            if gate_sup > 0:
                preds, logits = B.unroll(model, zb, ab, t0, U, hs, tau=tau,
                                         hard=False, want_logits=True)
            else:
                preds, logits = B.unroll(model, zb, ab, t0, U, hs, tau=tau,
                                         hard=False), None
            loss = F.mse_loss(preds, zb[:, t0 + 1:t0 + 1 + U])
            if gate_sup > 0 and logits is not None:
                tgt = lab_tr[bidx][:, t0 + 1:t0 + 1 + U]
                loss = loss + gate_sup * F.cross_entropy(
                    logits.reshape(-1, K), tgt.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_mse10(model, z_va, a_va, hs):
    model.eval()
    full = z_va.size(1) - 1
    preds = B.unroll(model, z_va, a_va, 0, full, hs, hard=True)
    per_k = ((preds - z_va[:, 1:1 + full]) ** 2).mean(dim=(0, 2)).cpu().numpy()
    return float(per_k[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/regime_stepB/train_contact.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="outputs/regime_stepB/routing_probe.json")
    args = ap.parse_args()

    B.configure_torch_threads_from_env()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    d = np.load(args.data)
    z = torch.from_numpy(d["z"].astype(np.float32))
    a = torch.from_numpy(d["a"].astype(np.float32))
    lab = torch.from_numpy((d["contact_frac"] > 0.0).astype(np.int64))
    N, K1, D = z.shape; adim = a.shape[2]; hs = 3

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(N, generator=g)
    nv = max(1, int(N * 0.1)); tr_i, va_i = perm[nv:], perm[:nv]
    z_tr, a_tr, lab_tr = z[tr_i].to(device), a[tr_i].to(device), lab[tr_i].to(device)
    z_va, a_va, lab_va = z[va_i].to(device), a[va_i].to(device), lab[va_i].to(device)

    configs = [
        ("blind  U=5", 5, 0.0),
        ("gate-sup U=5", 5, 1.0),
        ("gate-sup U=1 (LeWM-like)", 1, 1.0),
        ("blind  U=1 (LeWM-like)", 1, 0.0),
    ]
    rows = []
    for name, U, gs in configs:
        torch.manual_seed(args.seed)
        model = train_one(z_tr, a_tr, lab_tr, D, adim, hs, U, gs,
                          args.epochs, 1e-3, 512, device)
        tf_nmi, tf_acc = teacher_forced_routing(model, z_va, a_va, lab_va, hs)
        ro_nmi, ro_acc = rollout_routing(model, z_va, a_va, lab_va, hs)
        mse10 = eval_mse10(model, z_va, a_va, hs)
        rows.append(dict(config=name, U=U, gate_sup=gs, mse10=mse10,
                         tf_nmi=tf_nmi, tf_acc=tf_acc, ro_nmi=ro_nmi, ro_acc=ro_acc))
        print(f"{name:<26} mse@10={mse10:.3f} | teacher-forced NMI={tf_nmi:.3f} "
              f"acc={tf_acc:.3f} | ROLLOUT NMI={ro_nmi:.3f} acc={ro_acc:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(rows, indent=2))
    print(f"\ncontact_rate(val) = {(lab_va[:,1:]>0).float().mean().item():.3f} "
          f"(routing-acc trivial floor = max(rate,1-rate))")
    print(f"[wrote] {args.output}")


if __name__ == "__main__":
    main()
