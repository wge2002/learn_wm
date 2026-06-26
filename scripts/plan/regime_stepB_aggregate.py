"""Aggregate Step B decisive sweep: mono ladder vs MoE(state/both) vs oracle.

Keys runs by (arm, gate_input, hidden) so the two mono widths (h512 ~0.66M,
h1024 ~1.85M) do NOT collide. Computes per-config mean/std of mse@k, slope,
and gate-contact NMI; runs the decisive oracle-vs-mono-wide significance test;
writes a summary figure (mse@k curves + gate-NMI bars w/ Step A reference).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats


def load_runs(root: Path):
    runs = []
    for rj in sorted(root.glob("*/result.json")):
        d = json.loads(rj.read_text())
        d["_dir"] = rj.parent.name
        runs.append(d)
    return runs


def cfg_key(d):
    # round params to 2 decimals (M) to separate mono widths cleanly
    return (d["arm"], d.get("gate_input", "-"), round(d["n_params_M"], 2))


def cfg_label(arm, gin, p):
    if arm == "mono":
        return f"mono-h{'512' if p < 1.0 else '1024'} ({p:.2f}M)"
    if arm == "oracle":
        return f"oracle ({p:.2f}M)"
    return f"moe-{gin} ({p:.2f}M)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/regime_stepB/decisive_20260625_1913")
    ap.add_argument("--out-fig", default="docs/knowledge/regime_stepA_figures/stepB_decisive.png")
    args = ap.parse_args()
    root = Path(args.root)

    runs = load_runs(root)
    groups = defaultdict(list)
    for d in runs:
        groups[cfg_key(d)].append(d)

    # collect per-config stats
    rows = []
    for key, ds in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][2])):
        arm, gin, p = key
        mse_k = np.array([d["rollout_mse_vs_k"] for d in ds])  # (seeds, 10)
        slopes = np.array([d["rollout_mse_slope"] for d in ds])
        nmis = [d["gate_contact_alignment"]["gate_contact_nmi"]
                for d in ds if d.get("gate_contact_alignment")]
        rows.append(dict(
            arm=arm, gin=gin, p=p, label=cfg_label(arm, gin, p),
            n=len(ds),
            mse_mean=mse_k.mean(0), mse_std=mse_k.std(0),
            mse10=mse_k[:, 9], mse1=mse_k[:, 0],
            slope_mean=slopes.mean(), slope_std=slopes.std(),
            nmi=(float(np.mean(nmis)) if nmis else None),
        ))

    # print table
    print(f"\nStep B decisive sweep — {root.name}  (3 seeds each)\n")
    print(f"{'config':<22}{'mse@1':>8}{'mse@10':>16}{'slope':>9}{'gate-NMI':>10}")
    for r in rows:
        nmi = f"{r['nmi']:.3f}" if r["nmi"] is not None else "   -"
        print(f"{r['label']:<22}{r['mse1'].mean():>8.4f}"
              f"{r['mse10'].mean():>10.4f}±{r['mse10'].std():.3f}"
              f"{r['slope_mean']:>9.4f}{nmi:>10}")

    # decisive test: oracle vs mono-wide (the param-matched continuous baseline)
    def find(arm, pmin=None, pmax=None):
        for r in rows:
            if r["arm"] == arm and (pmin is None or r["p"] >= pmin) and (pmax is None or r["p"] <= pmax):
                return r
        return None

    oracle = find("oracle")
    mono_wide = find("mono", pmin=1.0)   # ~1.85M
    mono_narrow = find("mono", pmax=1.0)  # ~0.66M
    moe_state = next((r for r in rows if r["arm"] == "moe" and r["gin"] == "state"), None)
    moe_both = next((r for r in rows if r["arm"] == "moe" and r["gin"] == "both"), None)

    print("\n--- decisive: ORACLE vs param-matched continuous (mono-wide 1.85M) ---")
    if oracle and mono_wide:
        t, pv = stats.ttest_ind(mono_wide["mse10"], oracle["mse10"])
        dlt = oracle["mse10"].mean() - mono_wide["mse10"].mean()
        print(f"mse@10  mono-wide {mono_wide['mse10'].mean():.4f} vs oracle "
              f"{oracle['mse10'].mean():.4f}  Δ={dlt:+.4f}  t={t:.2f} p={pv:.3f}")
        ts, ps = stats.ttest_ind(mono_wide["mse10"], moe_state["mse10"]) if moe_state else (0, 1)
        print(f"slope   mono-wide {mono_wide['slope_mean']:.4f} vs oracle "
              f"{oracle['slope_mean']:.4f}")
        if moe_state:
            print(f"\nlearned gate vs oracle: moe-state mse@10 {moe_state['mse10'].mean():.4f} "
                  f"(p vs mono-wide={ps:.3f}) — gate-NMI {moe_state['nmi']:.3f} (≈0 ⇒ gate blind)")

    # ---- figure ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    ks = np.arange(1, 11)
    order = [mono_narrow, mono_wide, moe_state, moe_both, oracle]
    colors = {"mono-h512": "#9aa0a6", "mono-h1024": "#5f6368",
              "moe-state": "#1a73e8", "moe-both": "#7b1fa2", "oracle": "#188038"}
    for r in order:
        if r is None:
            continue
        ckey = r["label"].split(" (")[0]
        c = colors.get(ckey, "#000")
        axL.plot(ks, r["mse_mean"], "-o", ms=4, color=c, label=r["label"])
        axL.fill_between(ks, r["mse_mean"] - r["mse_std"], r["mse_mean"] + r["mse_std"],
                         color=c, alpha=0.12)
    axL.set_xlabel("rollout step k"); axL.set_ylabel("latent MSE@k")
    axL.set_title("Open-loop drift: oracle routing flattens drift\n(learned MoE does not)")
    axL.legend(fontsize=8, loc="upper left"); axL.grid(alpha=0.3)

    # right: gate-NMI bars vs Step A Jacobian reference + trivial floor
    bars = [("moe-state", moe_state["nmi"] if moe_state else 0, "#1a73e8"),
            ("moe-both", moe_both["nmi"] if moe_both else 0, "#7b1fa2")]
    xb = np.arange(len(bars))
    axR.bar(xb, [b[1] for b in bars], color=[b[2] for b in bars], width=0.5)
    axR.axhline(0.298, color="#188038", ls="--", lw=2,
                label="Step A Jacobian→contact NMI 0.30 (achievable)")
    axR.axhline(0.0, color="#999", ls=":", lw=1)
    axR.set_xticks(xb); axR.set_xticklabels([b[0] for b in bars])
    axR.set_ylabel("learned gate → contact NMI")
    axR.set_ylim(-0.02, 0.34)
    axR.set_title("Learned gate is contact-blind\n(bottleneck = gate discovery, not regime value)")
    axR.legend(fontsize=8); axR.grid(alpha=0.3, axis="y")

    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(args.out_fig, dpi=130)
    print(f"\n[fig] {args.out_fig}")


if __name__ == "__main__":
    main()
