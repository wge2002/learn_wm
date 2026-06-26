"""Step C (control layer) — regime-triggered re-grounding vs budget-matched uniform.

Direction doc: docs/knowledge/direction_discrete_regime_from_lewm.md (Step C iii).

Step B killed the regime-as-MoE-predictor form: specializing experts makes the model
brittle to routing error. The regime's surviving use (doc) is as a *monitoring signal*:
"regime boundary = when to re-ground". This does NOT route a brittle predictor; it only
decides WHEN to replace the drifting latent with a fresh true encoding.

Hypothesis: contact is piecewise dynamics, so prediction error spikes at regime
boundaries (contact onset/release). Spending a fixed re-ground budget AT those
boundaries should beat spreading it uniformly -> lower rollout MSE at equal budget.

This script tests the ORACLE version first (true contact boundaries) as a go/no-go,
exactly mirroring Step B's oracle-ceiling approach. If oracle-timed re-grounding does
not beat budget-matched uniform, the monitoring idea is dead; if it does, a realizable
test-time trigger is the next round.

Re-grounding mechanism is identical to phase3.regrounded_rollout (reseed from a single
true frame, history rebuilds); only the schedule of reseed points differs between arms,
so the comparison is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402
import regime_existence_stepA as stepA  # noqa: E402


# --------------------------------------------------------------------------- #
# scheduled re-grounding rollout: arbitrary per-trajectory reseed points,
# batched by segment horizon. Uses the SAME model.rollout as phase3 so the
# only difference between arms is WHERE the reseeds are.
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def scheduled_rollout(*, model, frames, true_emb, model_actions, reseed_mask,
                      batch_size, device):
    """reseed_mask: (n, num_k) bool; reseed_mask[:,0] is forced True. At each True
    index k the rollout reseeds from the true frame k and rolls open-loop until the
    next True index. Returns pred (n, num_k, D)."""
    dtype = next(model.parameters()).dtype
    n, num_k = true_emb.shape[:2]
    max_k = num_k - 1
    pred = np.empty_like(true_emb)
    pred[:, 0] = true_emb[:, 0]

    mask = reseed_mask.copy()
    mask[:, 0] = True
    # per-trajectory ordered anchors and the segments [anchor, next_anchor)
    seg_start = [[] for _ in range(n)]
    seg_hor = [[] for _ in range(n)]
    for i in range(n):
        anchors = np.flatnonzero(mask[i])
        anchors = anchors[anchors < max_k]  # last frame has no successor to predict
        for j, a in enumerate(anchors):
            nxt = anchors[j + 1] if j + 1 < len(anchors) else max_k
            seg_start[i].append(int(a))
            seg_hor[i].append(int(nxt - a))
    max_slots = max((len(s) for s in seg_start), default=0)

    for slot in range(max_slots):
        # trajectories that have a `slot`-th segment, grouped by horizon
        rows = np.array([i for i in range(n) if slot < len(seg_start[i])])
        if rows.size == 0:
            continue
        starts = np.array([seg_start[i][slot] for i in rows])
        hors = np.array([seg_hor[i][slot] for i in rows])
        for h in np.unique(hors):
            grp = rows[hors == h]
            st = starts[hors == h]
            for b0 in range(0, grp.size, batch_size):
                bi = grp[b0:b0 + batch_size]
                bs = st[b0:b0 + batch_size]
                seed = frames[bi, bs]  # (B, H, W, 3)
                pixels = p3.images_to_tensor(seed[:, None]).to(device=device, dtype=dtype)
                # gather action slices [start : start+h] per trajectory
                aidx = bs[:, None] + np.arange(h)[None, :]
                acts = torch.from_numpy(
                    model_actions[bi[:, None], aidx]).to(device=device, dtype=dtype)
                out = model.rollout({"pixels": pixels[:, None]}, acts[:, None])
                seg = out["predicted_emb"][:, 0, 1:h + 1].float().cpu().numpy()
                for r, (i, s) in enumerate(zip(bi, bs)):
                    pred[i, s + 1:s + h + 1] = seg[r]
    return pred


# --------------------------------------------------------------------------- #
# reseed schedules
# --------------------------------------------------------------------------- #
def regime_schedule(contact_bin):
    """reseed at contact-regime boundaries: k where the binary contact label differs
    from k-1 (onset or release). Plus forced k=0."""
    n, num_k = contact_bin.shape
    mask = np.zeros((n, num_k), bool)
    mask[:, 0] = True
    flips = contact_bin[:, 1:] != contact_bin[:, :-1]  # (n, num_k-1), flip AT k
    mask[:, 1:] |= flips
    return mask


def regime_pre_schedule(contact_bin):
    """reseed just BEFORE each contact-regime boundary (at k-1), giving the model a
    fresh true anchor right before the hard transition. Plus forced k=0."""
    n, num_k = contact_bin.shape
    mask = np.zeros((n, num_k), bool)
    mask[:, 0] = True
    flips = contact_bin[:, 1:] != contact_bin[:, :-1]  # flip AT k (index k-1 in flips)
    # flip at k -> reseed at k-1
    mask[:, :-1] |= flips
    return mask


def budget_matched_uniform(budget, num_k):
    """uniform schedule with exactly `budget_i` reseeds per trajectory (incl k=0),
    evenly spaced over [0, num_k-1)."""
    n = len(budget)
    mask = np.zeros((n, num_k), bool)
    mask[:, 0] = True
    for i in range(n):
        b = int(budget[i])
        if b <= 1:
            continue
        pos = np.unique(np.round(np.linspace(0, num_k - 1, b, endpoint=False)).astype(int))
        mask[i, pos] = True
    return mask


def fixed_interval_schedule(interval, n, num_k):
    mask = np.zeros((n, num_k), bool)
    mask[:, ::interval] = True
    mask[:, 0] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name",
                    default="/home/jovyan/.stable_worldmodel/pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--contact-thresh", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="outputs/regime_stepC/run")
    args = ap.parse_args()

    p3.configure_torch_threads_from_env()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t0 = time.time()

    dataset = p3.swm.data.load_dataset(
        args.dataset_name,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    batch = p3.build_window_batch(
        dataset, num_samples=args.num_samples, max_k=args.max_k,
        goal_offset=args.goal_offset, action_block=args.action_block, seed=args.seed)

    print("[stepC] replay for contact + frames", flush=True)
    frames, contact_max, contact_frac, _states, _term = stepA.replay_windows(
        env_name=args.env_name, init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        action_block=args.action_block, img_size=args.img_size, seed=args.seed)

    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    print(f"[stepC] encoding true frames {frames.shape}", flush=True)
    true_emb = p3.encode_frames(model=model, frames=frames,
                                batch_size=args.batch_size, device=device)

    contact_bin = (contact_frac > args.contact_thresh).astype(int)  # (n, num_k)
    ma = batch.model_actions
    n, num_k = true_emb.shape[:2]

    # schedules
    reg_mask = regime_schedule(contact_bin)
    pre_mask = regime_pre_schedule(contact_bin)
    budget = reg_mask.sum(axis=1)  # per-trajectory reseed count (incl k=0)
    budget_pre = pre_mask.sum(axis=1)
    uni_mask = budget_matched_uniform(budget, num_k)
    uni_pre_mask = budget_matched_uniform(budget_pre, num_k)
    open_mask = np.zeros((n, num_k), bool); open_mask[:, 0] = True

    def run(mask, name):
        pred = scheduled_rollout(model=model, frames=frames, true_emb=true_emb,
                                 model_actions=ma, reseed_mask=mask,
                                 batch_size=args.batch_size, device=device)
        mse = p3.metric_arrays(pred, true_emb)["mse"]  # (n, num_k)
        return mse

    print(f"[stepC] budget/traj: mean={budget.mean():.2f} "
          f"(min{budget.min()}/max{budget.max()}); contact_rate="
          f"{(contact_bin[:,1:]>0).mean():.3f}", flush=True)

    mse_open = run(open_mask, "open")
    mse_reg = run(reg_mask, "regime")
    mse_pre = run(pre_mask, "regime_pre")
    mse_uni = run(uni_mask, "uniform")
    mse_uni_pre = run(uni_pre_mask, "uniform_pre")
    fixed = {iv: run(fixed_interval_schedule(iv, n, num_k), f"fixed{iv}")
             for iv in (1, 2, 3, 5)}

    # area metric = mean MSE over interior steps k=1..max_k (lower=better drift ctrl)
    def area(m):
        return m[:, 1:].mean(axis=1)
    a_open, a_reg, a_uni = area(mse_open), area(mse_reg), area(mse_uni)
    a_pre, a_uni_pre = area(mse_pre), area(mse_uni_pre)

    # paired: regime vs budget-matched uniform. Restrict to traj where schedules
    # actually differ (budget>=2 AND regime!=uniform mask), report both.
    differ = (budget >= 2) & np.any(reg_mask != uni_mask, axis=1)
    t_all, p_all = stats.ttest_rel(a_uni, a_reg)
    t_d, p_d = stats.ttest_rel(a_uni[differ], a_reg[differ]) if differ.sum() > 1 else (0, 1)
    differ_pre = (budget_pre >= 2) & np.any(pre_mask != uni_pre_mask, axis=1)
    t_pre, p_pre = (stats.ttest_rel(a_uni_pre[differ_pre], a_pre[differ_pre])
                    if differ_pre.sum() > 1 else (0, 1))

    result = {
        "n": int(n), "num_k": int(num_k),
        "budget_mean": float(budget.mean()),
        "contact_rate": float((contact_bin[:, 1:] > 0).mean()),
        "n_differ": int(differ.sum()),
        "area_mse": {
            "open_loop": float(a_open.mean()),
            "regime": float(a_reg.mean()),
            "regime_pre": float(a_pre.mean()),
            "uniform_budget_matched": float(a_uni.mean()),
            "uniform_pre_matched": float(a_uni_pre.mean()),
            **{f"fixed{iv}": float(area(m).mean()) for iv, m in fixed.items()},
        },
        "regime_pre_vs_uniform_differ": {
            "delta": float(a_pre[differ_pre].mean() - a_uni_pre[differ_pre].mean()),
            "t": float(t_pre), "p": float(p_pre),
            "uniform": float(a_uni_pre[differ_pre].mean()),
            "regime_pre": float(a_pre[differ_pre].mean()),
            "n_differ": int(differ_pre.sum())},
        "regime_vs_uniform_all": {
            "delta": float(a_reg.mean() - a_uni.mean()),
            "t": float(t_all), "p": float(p_all)},
        "regime_vs_uniform_differ": {
            "delta": float(a_reg[differ].mean() - a_uni[differ].mean()),
            "t": float(t_d), "p": float(p_d),
            "uniform": float(a_uni[differ].mean()),
            "regime": float(a_reg[differ].mean())},
        "mse_curves": {
            "open_loop": mse_open.mean(0).tolist(),
            "regime": mse_reg.mean(0).tolist(),
            "uniform": mse_uni.mean(0).tolist(),
            **{f"fixed{iv}": m.mean(0).tolist() for iv, m in fixed.items()},
        },
        "elapsed_sec": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))

    # figure
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    ks = np.arange(num_k)
    axL.plot(ks, mse_open.mean(0), "-o", ms=3, color="#d93025", label="open-loop (0 reground)")
    for iv, m in fixed.items():
        axL.plot(ks, m.mean(0), "-", lw=1, alpha=0.5, color="#9aa0a6",
                 label=f"fixed every {iv}")
    axL.plot(ks, mse_uni.mean(0), "-s", ms=3, color="#5f6368",
             label=f"uniform (budget-matched ~{budget.mean():.1f})")
    axL.plot(ks, mse_reg.mean(0), "-o", ms=4, color="#188038",
             label="regime-triggered (at boundary)")
    axL.plot(ks, mse_pre.mean(0), "-^", ms=4, color="#1a73e8",
             label="regime-triggered (before boundary)")
    axL.set_xlabel("rollout step k"); axL.set_ylabel("latent MSE@k")
    axL.set_title("Re-grounding schedules (lower=better)")
    axL.legend(fontsize=8); axL.grid(alpha=0.3)

    labels = ["open", "uniform\n(matched)", "regime\n(oracle)"]
    vals = [a_open.mean(), a_uni.mean(), a_reg.mean()]
    cols = ["#d93025", "#5f6368", "#188038"]
    axR.bar(range(3), vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        axR.text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    axR.set_xticks(range(3)); axR.set_xticklabels(labels)
    axR.set_ylabel("area = mean MSE over k=1..max")
    axR.set_title(f"Equal budget: regime vs uniform\n"
                  f"Δ={result['regime_vs_uniform_differ']['delta']:+.4f} "
                  f"p={result['regime_vs_uniform_differ']['p']:.3f} "
                  f"(n_differ={int(differ.sum())})")
    axR.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(out / "stepC_reground.png", dpi=130)

    print(f"\n[stepC] area-MSE  open={a_open.mean():.4f}  "
          f"uniform={a_uni.mean():.4f}  regime={a_reg.mean():.4f}")
    print(f"[stepC] regime@boundary vs uniform (differ n={int(differ.sum())}): "
          f"Δ={result['regime_vs_uniform_differ']['delta']:+.4f} "
          f"p={result['regime_vs_uniform_differ']['p']:.3f}")
    print(f"[stepC] regime-pre vs uniform (differ n={int(differ_pre.sum())}): "
          f"Δ={result['regime_pre_vs_uniform_differ']['delta']:+.4f} "
          f"p={result['regime_pre_vs_uniform_differ']['p']:.3f}  "
          f"(pre={a_pre.mean():.4f} uni_pre={a_uni_pre.mean():.4f})")
    print(f"[stepC] fixed ladder: "
          + "  ".join(f"{iv}:{area(m).mean():.4f}" for iv, m in fixed.items()))
    print(f"[stepC] wrote {out}/result.json + stepC_reground.png "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
