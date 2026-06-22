"""LeWM two-goal CONTROL test (step 2): does a DISCRETE commitment proposer beat a
CONTINUOUS one for planning on the REAL LeWM model?

Two-goal PushT: from a start the planner may reach goal A OR B (cost = min(A,B)).
A mid-horizon commitment sub-goal is added to the planning cost (Phase-8i style):

    cost(candidate) = min(termMSE_A, termMSE_B)  +  lam * MSE(pred[mid], waypoint)

  - continuous proposer: f(z0, gA, gB) -> waypoint, trained on bimodal targets
    (z_wA or z_wB) with L2 -> predicts the BETWEEN-goals midpoint (invalid).
  - discrete proposer: selector(z0,gA,gB) -> commit A|B, head -> that branch's
    waypoint (committed, valid).

We run random-shooting two-goal planning under three cost configs (baseline lam=0,
+continuous, +discrete), pick each config's best candidate, and measure how close
that plan actually gets to EITHER goal in imagination (terminal min-cost, lower =
better; and "reach rate" vs the baseline median). discrete << continuous terminal
=> the discrete commitment helps planning on real LeWM latent.

(Imagination/planner-decision level; closed-loop env success is the follow-up.)
Reuses latent_drift_phase3. Writes outputs/lewm_twogoal_control/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3  # noqa: E402


@torch.inference_mode()
def encode_goal_emb(model, goal_frames, device):
    dtype = next(model.parameters()).dtype
    px = p3.images_to_tensor(goal_frames[:, None]).to(device=device, dtype=dtype)  # (B,1,C,H,W)
    return model.encode({"pixels": px})["emb"][:, 0].float()                        # (B,D)


@torch.inference_mode()
def roll(model, start_frames, acts, device, bchunk=16):
    """Roll S action candidates from a single grounding frame, CHUNKED over starts
    to bound GPU memory (rolling B*S sequences at once OOMs for large B*S).
    acts: (B,S,T,madim) tensor on device. Returns predicted_emb (B,S,T+1,D)."""
    dtype = next(model.parameters()).dtype
    B = start_frames.shape[0]
    S = acts.shape[1]
    outs = []
    for s in range(0, B, bchunk):
        sf = start_frames[s:s + bchunk]; ac = acts[s:s + bchunk]
        b = sf.shape[0]
        px1 = p3.images_to_tensor(sf[:, None]).to(device=device, dtype=dtype)        # (b,1,C,H,W)
        pixels = px1[:, None].expand(b, S, 1, *px1.shape[2:]).contiguous()           # (b,S,1,C,H,W)
        info = model.rollout({"pixels": pixels}, ac)
        outs.append(info["predicted_emb"].float())
    return torch.cat(outs, 0)                                                        # (B,S,T+1,D)


def term_cost(pred_final, g_emb):
    """sum-sq distance of terminal latent to a goal emb. pred_final (B,S,D), g (B,D)."""
    return ((pred_final - g_emb[:, None, :]) ** 2).sum(-1)                           # (B,S)


class MLP(nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="quentinll/lewm-pusht")
    ap.add_argument("--dataset-name", default="pusht_expert_train.h5")
    ap.add_argument("--env-name", default="swm/PushT-v1")
    ap.add_argument("--output-dir", default="outputs/lewm_twogoal_control")
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--n-samples", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--goal-pairing", default="far", choices=["far", "shuffle"])
    ap.add_argument("--pred-epochs", type=int, default=200)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    p3.configure_torch_threads_from_env()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    dataset = p3.swm.data.load_dataset(
        args.dataset_name, cache_dir=args.cache_dir,
        keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    model = p3.swm.wm.utils.load_pretrained(args.policy).to(device).eval()
    model.requires_grad_(False); model.interpolate_pos_encoding = True
    action_dim = int(np.asarray(dataset.get_col_data("action")).reshape(
        len(np.asarray(dataset.get_col_data("state"))), -1).shape[1])
    madim = args.action_block * action_dim
    mid = max(1, args.horizon // 2)

    def make_set(n, seed):
        batch = p3.build_window_batch(dataset, num_samples=n, max_k=2, goal_offset=args.goal_offset,
                                      action_block=args.action_block, seed=seed)
        starts = batch.states_by_k[:, 0]; gA = batch.goal_states
        if args.goal_pairing == "far":
            # pair each start's goalA with the batch goal whose BLOCK pose (dims 2,3,4)
            # is farthest -> genuinely distinct A/B (random shuffle can pair near-dups)
            bp = gA[:, 2:5]
            dmat = np.linalg.norm(bp[:, None, :] - bp[None, :, :], axis=-1)
            gB = gA[dmat.argmax(1)]
        else:
            gB = gA[rng.permutation(len(gA))]
        sf = p3.render_state_sequence(env_name=args.env_name, states_by_k=starts[:, None],
                                      goal_states=gA, variations=(), img_size=args.img_size, seed=seed)[:, 0]
        gAf = p3.render_state_sequence(env_name=args.env_name, states_by_k=gA[:, None],
                                       goal_states=gA, variations=(), img_size=args.img_size, seed=seed)[:, 0]
        gBf = p3.render_state_sequence(env_name=args.env_name, states_by_k=gB[:, None],
                                       goal_states=gB, variations=(), img_size=args.img_size, seed=seed)[:, 0]
        return sf, gAf, gBf

    def plan_pack(sf, gAf, gBf):
        """sample candidates, roll, return pred (B,S,T+1,D), costA, costB, embs."""
        B = len(sf)
        acts = torch.from_numpy(rng.uniform(-1, 1, (B, args.n_samples, args.horizon, madim)).astype(np.float32)
                                ).to(device, next(model.parameters()).dtype)
        pred = roll(model, sf, acts, device)                       # (B,S,T+1,D)
        gA = encode_goal_emb(model, gAf, device); gB = encode_goal_emb(model, gBf, device)
        z0 = encode_goal_emb(model, sf, device)                    # grounded latent of start
        cA = term_cost(pred[:, :, -1], gA); cB = term_cost(pred[:, :, -1], gB)
        return pred, cA, cB, z0, gA, gB

    # ---- training data: bimodal waypoints toward A / toward B ----
    print(f"[ctrl] gen train n={args.n_train} (madim={madim} mid={mid})", flush=True)
    sf, gAf, gBf = make_set(args.n_train, args.seed)
    pred, cA, cB, z0, gA, gB = plan_pack(sf, gAf, gBf)
    bestA = cA.argmin(1); bestB = cB.argmin(1)
    wA = torch.gather(pred[:, :, mid], 1, bestA[:, None, None].expand(-1, 1, pred.shape[-1]))[:, 0]
    wB = torch.gather(pred[:, :, mid], 1, bestB[:, None, None].expand(-1, 1, pred.shape[-1]))[:, 0]
    X = torch.cat([z0, gA, gB], -1).cpu().numpy()
    wA, wB = wA.cpu().numpy(), wB.cpu().numpy()
    # z-score latent space (fit on all latents involved)
    allz = np.concatenate([z0.cpu().numpy(), gA.cpu().numpy(), gB.cpu().numpy(), wA, wB], 0)
    mu, sd = allz.mean(0), allz.std(0) + 1e-6
    def nz(a): return (a - mu) / sd
    Xn = np.concatenate([nz(z0.cpu().numpy()), nz(gA.cpu().numpy()), nz(gB.cpu().numpy())], -1).astype(np.float32)
    wAn, wBn = nz(wA).astype(np.float32), nz(wB).astype(np.float32)
    D = wAn.shape[1]

    # pooled bimodal targets: half A, half B
    pick = rng.integers(2, size=len(Xn))
    Ytgt = np.where(pick[:, None] == 0, wAn, wBn).astype(np.float32)
    lab = pick.astype(np.int64)

    def train(model_, Xin, Yin, loss_fn, epochs):
        o = torch.optim.Adam(model_.parameters(), 1e-3)
        Xt = torch.tensor(Xin, device=device); Yt = torch.tensor(Yin, device=device); n = len(Xt)
        for ep in range(epochs):
            pm = torch.randperm(n, device=device)
            for s in range(0, n, 512):
                idx = pm[s:s+512]; loss = loss_fn(model_(Xt[idx]), Yt[idx])
                o.zero_grad(); loss.backward(); o.step()
        return model_

    print("[ctrl] train continuous + discrete proposers...", flush=True)
    cont = train(MLP(3*D, D).to(device), Xn, Ytgt, nn.MSELoss(), args.pred_epochs)
    sel = train(MLP(3*D, 2).to(device), Xn, lab,
                lambda o, t: F.cross_entropy(o, torch.as_tensor(t, device=device).long()), args.pred_epochs)
    Xb = np.concatenate([Xn, np.eye(2, dtype=np.float32)[lab]], -1)
    head = train(MLP(3*D + 2, D).to(device), Xb, np.where(lab[:, None] == 0, wAn, wBn).astype(np.float32),
                 nn.MSELoss(), args.pred_epochs)
    # SEPARATE per-branch heads: architecturally CANNOT ignore the branch (different
    # nets) -> discrete's best shot (rules out "onehot conditioning too weak").
    headA = train(MLP(3*D, D).to(device), Xn, wAn, nn.MSELoss(), args.pred_epochs)
    headB = train(MLP(3*D, D).to(device), Xn, wBn, nn.MSELoss(), args.pred_epochs)

    # waypoint blur check: continuous vs discrete between-ness on held-out branches
    with torch.no_grad():
        cw = cont(torch.tensor(Xn, device=device)).cpu().numpy()
        bsel = sel(torch.tensor(Xn, device=device)).argmax(-1).cpu().numpy()
        dw = head(torch.tensor(np.concatenate([Xn, np.eye(2, dtype=np.float32)[bsel]], -1), device=device)).cpu().numpy()
        # I3 diagnostic: does the head actually USE the branch onehot? force A vs B.
        oh0 = np.tile([1, 0], (len(Xn), 1)).astype(np.float32)
        oh1 = np.tile([0, 1], (len(Xn), 1)).astype(np.float32)
        h0 = head(torch.tensor(np.concatenate([Xn, oh0], -1), device=device)).cpu().numpy()
        h1 = head(torch.tensor(np.concatenate([Xn, oh1], -1), device=device)).cpu().numpy()
    onehot_response = float(np.linalg.norm(h0 - h1, axis=1).mean() /
                            (np.linalg.norm(wAn - wBn, axis=1).mean() + 1e-9))  # ~0 ignores branch, ~1 uses it
    wp_sep = np.linalg.norm(wAn - wBn, axis=1)                     # mid-waypoint A/B separation
    ok = wp_sep > np.median(wp_sep) * 0.25 + 1e-6                  # only score where branches separate
    half = wp_sep[ok] / 2 + 1e-6
    cont_btw = float((np.minimum(np.linalg.norm((cw - wAn)[ok], axis=1),
                                 np.linalg.norm((cw - wBn)[ok], axis=1)) / half).mean()) if ok.any() else float("nan")
    disc_btw = float((np.minimum(np.linalg.norm((dw - wAn)[ok], axis=1),
                                 np.linalg.norm((dw - wBn)[ok], axis=1)) / half).mean()) if ok.any() else float("nan")
    R_wpsep = float(wp_sep.mean())

    # ---- control eval: plan under baseline / +continuous / +discrete ----
    print(f"[ctrl] eval n={args.n_eval} lam={args.lam}", flush=True)
    sfe, gAfe, gBfe = make_set(args.n_eval, args.seed + 999)
    pe, cAe, cBe, z0e, gAe, gBe = plan_pack(sfe, gAfe, gBfe)
    base_term = torch.minimum(cAe, cBe)                         # (B,S)
    midz = pe[:, :, mid]                                        # (B,S,D)
    Xe = np.concatenate([nz(z0e.cpu().numpy()), nz(gAe.cpu().numpy()), nz(gBe.cpu().numpy())], -1).astype(np.float32)
    with torch.no_grad():
        wc = cont(torch.tensor(Xe, device=device)).cpu().numpy() * sd + mu          # de-norm to latent
        be = sel(torch.tensor(Xe, device=device)).argmax(-1).cpu().numpy()
        wd = head(torch.tensor(np.concatenate([Xe, np.eye(2, dtype=np.float32)[be]], -1), device=device)).cpu().numpy() * sd + mu
        hA = headA(torch.tensor(Xe, device=device)).cpu().numpy()
        hB = headB(torch.tensor(Xe, device=device)).cpu().numpy()
        wsep = np.where(be[:, None] == 0, hA, hB) * sd + mu        # separate-head discrete
    wc_t = torch.tensor(wc, device=device, dtype=midz.dtype); wd_t = torch.tensor(wd, device=device, dtype=midz.dtype)
    commit_c = ((midz - wc_t[:, None]) ** 2).sum(-1)           # (B,S)
    commit_d = ((midz - wd_t[:, None]) ** 2).sum(-1)
    # ORACLE commit: the TRUE committed-branch mid-waypoint (best candidate toward the
    # achievable goal). Upper bound on what ANY proposer could give. If even this
    # doesn't beat baseline, the commitment cost term is moot for this min(A,B) CEM.
    bestAe = cAe.argmin(1); bestBe = cBe.argmin(1)
    wAe = torch.gather(midz, 1, bestAe[:, None, None].expand(-1, 1, midz.shape[-1]))[:, 0]
    wBe = torch.gather(midz, 1, bestBe[:, None, None].expand(-1, 1, midz.shape[-1]))[:, 0]
    pickA = (cAe.min(1).values <= cBe.min(1).values)          # commit to the achievable branch
    wO = torch.where(pickA[:, None], wAe, wBe)
    commit_o = ((midz - wO[:, None]) ** 2).sum(-1)

    def best_terminal(total_cost):
        idx = total_cost.argmin(1)
        return torch.gather(base_term, 1, idx[:, None])[:, 0]   # the REAL min-goal-dist of chosen plan

    wsep_t = torch.tensor(wsep, device=device, dtype=midz.dtype)
    commit_sep = ((midz - wsep_t[:, None]) ** 2).sum(-1)
    # separate-head between-ness on train (does it actually commit?)
    with torch.no_grad():
        hAn = headA(torch.tensor(Xn, device=device)).cpu().numpy()
        hBn = headB(torch.tensor(Xn, device=device)).cpu().numpy()
    wsep_n = np.where(bsel[:, None] == 0, hAn, hBn)
    sep_btw = float((np.minimum(np.linalg.norm((wsep_n - wAn)[ok], axis=1),
                                np.linalg.norm((wsep_n - wBn)[ok], axis=1)) / half).mean()) if ok.any() else float("nan")

    term_base = best_terminal(base_term)
    term_cont = best_terminal(base_term + args.lam * commit_c)
    term_disc = best_terminal(base_term + args.lam * commit_d)
    term_sep = best_terminal(base_term + args.lam * commit_sep)
    term_oracle = best_terminal(base_term + args.lam * commit_o)
    thr = float(term_base.median())                            # "reach" = at least as close as baseline median
    R = {
        "n_train": args.n_train, "n_eval": args.n_eval, "n_samples": args.n_samples,
        "horizon": args.horizon, "mid": mid, "lam": args.lam, "latent_dim": int(D),
        "mid_waypoint_sep_AB": R_wpsep,
        "head_onehot_response": onehot_response,
        "waypoint_betweenness_continuous": float(cont_btw),
        "waypoint_betweenness_discrete": float(disc_btw),
        "waypoint_betweenness_discrete_sephead": sep_btw,
        "term_discrete_sephead": float(term_sep.mean()),
        "reach_discrete_sephead": float((term_sep <= thr).float().mean()),
        "term_base": float(term_base.mean()), "term_continuous": float(term_cont.mean()),
        "term_discrete": float(term_disc.mean()), "term_oracle_commit": float(term_oracle.mean()),
        "reach_base": float((term_base <= thr).float().mean()),
        "reach_continuous": float((term_cont <= thr).float().mean()),
        "reach_discrete": float((term_disc <= thr).float().mean()),
        "reach_oracle_commit": float((term_oracle <= thr).float().mean()),
        "oracle_minus_base_term": float((term_oracle - term_base).mean()),
        "discrete_minus_continuous_term": float((term_disc - term_cont).mean()),
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "twogoal_control_summary.json").write_text(json.dumps(R, indent=2))
    lines = [
        "=== LeWM two-goal CONTROL: discrete vs continuous commitment proposer ===",
        f"n_train={R['n_train']} n_eval={R['n_eval']} S={R['n_samples']} H={R['horizon']} "
        f"lam={R['lam']} ({R['elapsed_sec']}s)",
        "",
        f"head onehot-response={R['head_onehot_response']:.2f} (~0 head IGNORES branch / ~1 uses it)",
        f"mid-waypoint A/B sep={R['mid_waypoint_sep_AB']:.2f}  | between-ness "
        f"continuous={R['waypoint_betweenness_continuous']:.2f} discrete={R['waypoint_betweenness_discrete']:.2f} (~1 blur/~0 commit)",
        "",
        "chosen-plan terminal min-dist to EITHER goal (lower=better plan):",
        f"  baseline (lam=0)      : {R['term_base']:.2f}",
        f"  + continuous commit   : {R['term_continuous']:.2f}",
        f"  + DISCRETE commit     : {R['term_discrete']:.2f}",
        f"  + DISCRETE sep-head   : {R['term_discrete_sephead']:.2f}  (between-ness {R['waypoint_betweenness_discrete_sephead']:.2f}; discrete's best shot)",
        f"  + ORACLE commit (UB)  : {R['term_oracle_commit']:.2f}  (true committed waypoint; oracle-base={R['oracle_minus_base_term']:+.2f})",
        f"  discrete - continuous : {R['discrete_minus_continuous_term']:+.2f} (negative = discrete better)",
        "",
        f"reach rate (<= baseline median): base {R['reach_base']:.2f} / cont {R['reach_continuous']:.2f} / disc {R['reach_discrete']:.2f} / oracle {R['reach_oracle_commit']:.2f}",
    ]
    (out / "RESULT.txt").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    print(f"[ctrl] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
