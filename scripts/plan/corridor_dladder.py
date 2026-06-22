"""D-ladder on the Stage 3 corridor testbed: HOW you train the discrete commitment
is the variable. All methods are goal-conditioned state->action policies trained
by BC on the SAME multimodal expert data, evaluated on the SAME corridor, swept
over horizon N. Apples-to-apples.

Methods:
  continuous : pi(s,g) -> a, MSE. Blurs to the mean at each ridge (baseline).
  oracle     : hand-coded commit-to-a-door expert (ceiling).
  D0         : frozen-reconstruction discrete. A selector q(s,g) is trained
               SEPARATELY to predict the committed-door label (a reconstruction
               signal, NO task gradient), frozen; then pi(s,g,onehot(c)) is BC'd.
  D2         : end-to-end discrete. selector q + policy pi trained JOINTLY; the
               code c ~ Gumbel-softmax(straight-through) so q gets gradient from
               the BC/control loss -> the codebook is shaped by task utility.

Commitment semantics at eval: the code is (re)selected once when the agent enters
a new corridor segment and HELD while crossing it.

Hypothesis: continuous decays with N; D0 ~ continuous (or weak); D2 stays high and
beats continuous at large N. Which of these holds is the empirical question that
tells us whether "discrete doesn't help" or "naive-trained discrete doesn't help".

Reuses the env from corridor_stage3.py. Writes outputs/corridor_dladder/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from corridor_stage3 import CorridorMaze, expert_action, sample_doors, DOOR_Y  # noqa: E402


def _reset(env, rng, center):
    """center: start/goal y near the mid-line (0.5) -> every junction forces a
    genuine up/down commitment (the multimodal decision is unavoidable), which
    cuts the config-difficulty variance that made the single-seed sweep noisy."""
    if center:
        return env.reset(float(rng.uniform(0.45, 0.55)), float(rng.uniform(0.45, 0.55)))
    return env.reset()


def gen_with_labels(n_walls, episodes, max_steps, rng, center):
    """Multimodal stochastic-branch expert; also record the committed-door label
    (0=top,1=bottom) of the segment the agent is currently heading into."""
    S, A, L = [], [], []
    for _ in range(episodes):
        env = CorridorMaze(n_walls, rng); _reset(env, rng, center)
        doors = sample_doors(n_walls, rng, multimodal=True)
        for t in range(max_steps):
            a = expert_action(env, doors)
            c = min(env.crossed(), n_walls - 1)
            lab = 0 if doors[c] == DOOR_Y[0] else 1
            S.append(env.state()); A.append(a); L.append(lab)
            _, done = env.step(a)
            if done:
                break
    return np.array(S, np.float32), np.array(A, np.float32), np.array(L, np.int64)


class MLP(nn.Module):
    def __init__(self, din, dout, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/corridor_dladder")
    ap.add_argument("--walls", default="1,2,3,4,5,6")
    ap.add_argument("--episodes", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--eval-episodes", type=int, default=400)
    ap.add_argument("--codes", type=int, default=2)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--center", action="store_true",
                    help="center start/goal y so every junction forces a commitment")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    Ns = [int(x) for x in args.walls.split(",")]
    K = args.codes
    R = {"per_N": {}, "codes": K}

    def batches(n, bs=2048):
        pm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            yield pm[s:s+bs]

    for N in Ns:
        rng = np.random.default_rng(args.seed + N)
        max_steps = 40 * (N + 1)
        S, A, L = gen_with_labels(N, args.episodes, max_steps, rng, args.center)
        Xt = torch.tensor(S, device=dev); Yt = torch.tensor(A, device=dev)
        Lt = torch.tensor(L, device=dev); n = len(Xt)

        # --- continuous baseline ---
        cont = MLP(4, 2).to(dev); oc = torch.optim.Adam(cont.parameters(), 1e-3)
        for _ in range(args.epochs):
            for idx in batches(n):
                loss = F.mse_loss(cont(Xt[idx]), Yt[idx]); oc.zero_grad(); loss.backward(); oc.step()

        # --- D0: frozen-reconstruction selector (predict door label), then BC ---
        sel0 = MLP(4, K).to(dev); os0 = torch.optim.Adam(sel0.parameters(), 1e-3)
        for _ in range(args.epochs):                     # selector: predict committed door (no task grad)
            for idx in batches(n):
                loss = F.cross_entropy(sel0(Xt[idx]), Lt[idx].clamp(max=K-1))
                os0.zero_grad(); loss.backward(); os0.step()
        for p in sel0.parameters():
            p.requires_grad_(False)
        pol0 = MLP(4 + K, 2).to(dev); op0 = torch.optim.Adam(pol0.parameters(), 1e-3)
        for _ in range(args.epochs):
            for idx in batches(n):
                with torch.no_grad():
                    c = F.one_hot(sel0(Xt[idx]).argmax(-1), K).float()
                loss = F.mse_loss(pol0(torch.cat([Xt[idx], c], -1)), Yt[idx])
                op0.zero_grad(); loss.backward(); op0.step()

        # --- D2: end-to-end selector + policy, Gumbel straight-through ---
        # tau annealing (soft->hard) + load-balancing aux loss prevent the classic
        # codebook collapse (selector ignoring state -> single code -> no commit).
        sel2 = MLP(4, K).to(dev); pol2 = MLP(4 + K, 2).to(dev)
        o2 = torch.optim.Adam(list(sel2.parameters()) + list(pol2.parameters()), 1e-3)
        for ep in range(args.epochs):
            tau = max(0.5, args.tau * (1.0 - ep / max(args.epochs - 1, 1)) + 0.5)
            for idx in batches(n):
                lg = sel2(Xt[idx]); p = lg.softmax(-1)
                c = F.gumbel_softmax(lg, tau=tau, hard=True)
                mse = F.mse_loss(pol2(torch.cat([Xt[idx], c], -1)), Yt[idx])
                mean_p = p.mean(0)                                  # batch code usage
                lb = (mean_p * (mean_p * K + 1e-9).log()).sum()    # KL(mean_p || uniform)
                loss = mse + 0.05 * lb
                o2.zero_grad(); loss.backward(); o2.step()

        # --- eval with per-segment commitment ---
        def run(kind):
            rg = np.random.default_rng(args.seed + 100 + N); succ = 0
            for _ in range(args.eval_episodes):
                env = CorridorMaze(N, rg); _reset(env, rg, args.center)
                doors = sample_doors(N, rg, multimodal=True)
                seg = -1; c_oh = None
                for t in range(max_steps):
                    s = env.state()
                    if kind == "oracle":
                        a = expert_action(env, doors)
                    else:
                        st = torch.tensor(s[None], device=dev)
                        if kind == "continuous":
                            with torch.no_grad():
                                a = cont(st).cpu().numpy()[0]
                        else:
                            cur = env.crossed()
                            if cur != seg:                 # commit a code on entering a new segment
                                seg = cur
                                sel = sel0 if kind == "D0" else sel2
                                with torch.no_grad():
                                    c_oh = F.one_hot(sel(st).argmax(-1), K).float()
                            pol = pol0 if kind == "D0" else pol2
                            with torch.no_grad():
                                a = pol(torch.cat([st, c_oh], -1)).cpu().numpy()[0]
                        nrm = np.linalg.norm(a); a = (a / nrm if nrm > 1e-6 else a).astype(np.float32)
                    _, done = env.step(a)
                    if done:
                        succ += 1; break
            return 100.0 * succ / args.eval_episodes

        res = {k: run(k) for k in ["oracle", "continuous", "D0", "D2"]}
        R["per_N"][N] = res
        print(f"[D] N={N}: oracle={res['oracle']:.0f}  cont={res['continuous']:.0f}  "
              f"D0={res['D0']:.0f}  D2={res['D2']:.0f}  "
              f"(D2-cont={res['D2']-res['continuous']:+.0f}, D0-cont={res['D0']-res['continuous']:+.0f})",
              flush=True)

    (out / "corridor_dladder_summary.json").write_text(json.dumps(R, indent=2))
    lines = ["=== D-ladder on corridor testbed: how to train discrete commitment ===",
             f"codes K={K}, multimodal expert, per-segment commitment at eval",
             "",
             f"{'N':>3} | {'oracle':>6} | {'cont':>5} | {'D0':>5} | {'D2':>5} | {'D2-cont':>7} | {'D0-cont':>7}",
             "-" * 60]
    for N in Ns:
        d = R["per_N"][N]
        lines.append(f"{N:>3} | {d['oracle']:>5.0f}% | {d['continuous']:>4.0f}% | {d['D0']:>4.0f}% | "
                     f"{d['D2']:>4.0f}% | {d['D2']-d['continuous']:>+6.0f} | {d['D0']-d['continuous']:>+6.0f}")
    lines += ["",
              "read: continuous should decay with N; the question is whether D0 and/or D2",
              "recover toward oracle. D2>>continuous => training signal (task gradient) is",
              "the key; D0>>continuous => even naive discrete pays off on the right testbed."]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[D] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
