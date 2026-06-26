"""Step B round A summary: regime is real (oracle) but no realizable learned gate
captures it. Bar of mse@10 across the variant ladder vs mono-wide / oracle refs.

Pulls means/seeds straight from the run dirs; no hardcoded numbers.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEC = Path("outputs/regime_stepB/decisive_20260625_1913")
GS = Path("outputs/regime_stepB/gatesup_20260626_1615")
TRGT = Path("outputs/regime_stepB/trgt_20260626_1619")
SOFT = Path("outputs/regime_stepB/softeval_20260626_1622")


def mse10(dirs):
    vals = [json.loads(p.read_text())["rollout_mse_vs_k"][9]
            for d in dirs for p in [d / "result.json"] if p.exists()]
    return np.array(vals)


def pick(root, pred):
    return [d for d in sorted(root.glob("*/")) if (d / "result.json").exists()
            and pred(json.loads((d / "result.json").read_text()))]


# realizable variants (no test-time GT label) + the oracle upper bound
bars = [
    ("oracle\n(test-time GT)", mse10(pick(DEC, lambda r: r["arm"] == "oracle")), "#188038"),
    ("mono-wide\n(continuous)", mse10(pick(DEC, lambda r: r["arm"] == "mono" and r["n_params_M"] > 1.0)), "#5f6368"),
    ("blind MoE\n(no sup)", mse10(pick(DEC, lambda r: r["arm"] == "moe" and r["gate_input"] == "state")), "#9aa0a6"),
    ("clean-experts\n+gate, soft", mse10(pick(SOFT, lambda r: True) and [d for d in sorted(SOFT.glob("trgt_soft_*/"))]), "#7b1fa2"),
    ("clean-experts\n+gate, hard", mse10(sorted(TRGT.glob("trgt_gs1.0_*/"))), "#9c27b0"),
    ("sup-gate\nsoft", mse10(sorted(SOFT.glob("sup_soft_*/"))), "#1a73e8"),
    ("sup-gate\nhard", mse10(pick(GS, lambda r: r["gate_sup"] == 1.0 and r["gate_input"] == "state")), "#4285f4"),
]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(bars))
means = [b[1].mean() for b in bars]
errs = [b[1].std() for b in bars]
ax.bar(x, means, yerr=errs, color=[b[2] for b in bars], width=0.62, capsize=4)
mono = bars[1][1].mean()
ax.axhline(mono, color="#5f6368", ls="--", lw=1.5, label=f"mono-wide baseline {mono:.3f}")
ax.axhline(bars[0][1].mean(), color="#188038", ls=":", lw=1.5,
           label=f"oracle upper bound {bars[0][1].mean():.3f}")
for xi, m in zip(x, means):
    ax.text(xi, m + 0.012, f"{m:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
ax.set_ylabel("open-loop drift  mse@10  (lower = better)")
ax.set_ylim(0, 0.58)
ax.set_title("Step B round A — regime is real (oracle wins) but NO realizable learned gate captures it\n"
             "every realizable MoE variant is WORSE than the plain continuous predictor")
ax.legend(loc="upper left"); ax.grid(alpha=0.3, axis="y")
out = "docs/knowledge/regime_stepA_figures/stepB_roundA.png"
fig.tight_layout(); fig.savefig(out, dpi=130)
print("[fig]", out)
for b in bars:
    print(f"  {b[0].replace(chr(10),' '):<28} {b[1].mean():.4f} ± {b[1].std():.4f}  (n={len(b[1])})")
