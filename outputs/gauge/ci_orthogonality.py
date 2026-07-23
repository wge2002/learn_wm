"""CI-GWM veto gate: is the action-response subspace orthogonal to the top
error-amplification subspace of the latent transition?

CI-GWM (Control-Innovation factorized Gaussian WM) posits a transition
  z_{t+1} = g(z_t) + U(z_t) psi(a_t) + L_perp(z_t) eps
with a low-rank action tangent U orthogonal to a diagonal innovation channel.
That factorization is only a correct inductive bias if, in the ALREADY TRAINED
models, the directions actions push (col-space of dz'/da) are approximately
orthogonal to the directions that amplify state error (top singular vectors of
dz'/dz). This script measures that overlap on existing checkpoints — zero
training. Kill CI-GWM if median squared overlap > 0.3 (pre-registered).

For each window along the true trajectory:
  Ja = d predict / d act_emb  (D x A)   -> action-response span (rank <= A)
  Jz = d predict / d z_last   (D x D)   -> top-r left singular vectors = amp span
  overlap = || P_amp @ orth(Ja) ||_F^2 / rank(Ja)   in [0,1]
Also reports a random-subspace baseline (expected overlap r/D) so the number
is interpretable, and the action-channel amplification vs innovation-channel
amplification (does error really live off the action span?).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_worldmodel as swm  # noqa: E402

HS, STEPS = 3, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--num', type=int, default=256)
    ap.add_argument('--rank', type=int, default=0,
                    help='top-r amplification dirs; 0 => match action rank')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = swm.wm.utils.load_pretrained(args.policy).to(dev).eval()
    model.requires_grad_(False)
    d = np.load(args.data)
    z = torch.from_numpy(d['z'][: args.num].astype(np.float32)).to(dev)
    a = torch.from_numpy(d['a'][: args.num].astype(np.float32)).to(dev)
    N, L, D = z.shape
    act_emb = model.action_encoder(a)
    Araw = a.shape[-1]  # PHYSICAL action dim (2 for PushT); CI-GWM's low-rank U

    def jacs(ctx, araw_win, ae):
        # Jz: D x D wrt last latent; Ja: D x Araw wrt last PHYSICAL action
        # (differentiated through the action encoder, so the action tangent has
        # rank <= physical action dim, which is what CI-GWM's low-rank U means).
        zl = ctx[-1].detach().requires_grad_(True)
        araw = araw_win[-1].detach().requires_grad_(True)

        def g_z(x):
            c = torch.cat([ctx[:-1], x.unsqueeze(0)], 0).unsqueeze(0)
            return model.predict(c, ae.unsqueeze(0))[0, -1]

        def g_a(x):
            ae_last = model.action_encoder(x.reshape(1, 1, -1))[0, 0]
            aw = torch.cat([ae[:-1], ae_last.unsqueeze(0)], 0).unsqueeze(0)
            return model.predict(ctx.unsqueeze(0), aw)[0, -1]

        Jz = torch.autograd.functional.jacobian(g_z, zl, vectorize=True)
        Ja = torch.autograd.functional.jacobian(g_a, araw, vectorize=True)
        return Jz, Ja

    overlaps, act_amp, inn_amp, aranks = [], [], [], []
    for i in range(N):
        t = HS - 1 + (STEPS // 2)  # a mid-trajectory window
        Jz, Ja = jacs(z[i, t - 2 : t + 1], a[i, t - 2 : t + 1], act_emb[i, t - 2 : t + 1])
        # action-response orthonormal basis
        Ua, sa, _ = torch.linalg.svd(Ja, full_matrices=False)
        ar = int((sa > 1e-4 * sa[0]).sum())
        if ar == 0:
            continue
        Ua = Ua[:, :ar]  # D x ar
        aranks.append(ar)
        r = args.rank or ar
        # top-r amplification directions = top left singular vecs of Jz
        Uz, sz, _ = torch.linalg.svd(Jz, full_matrices=False)
        Uamp = Uz[:, :r]  # D x r
        # squared overlap: how much of action span lies in amp span
        M = Uamp.T @ Ua  # r x ar
        ov = float((M**2).sum() / ar)
        overlaps.append(ov)
        # amplification felt along action span vs orthogonal complement
        proj_a = Ua @ (Ua.T)  # D x D projector onto action span
        act_amp.append(float(torch.linalg.matrix_norm(proj_a @ Jz, ord=2)))
        inn_amp.append(float(torch.linalg.matrix_norm((torch.eye(D, device=dev) - proj_a) @ Jz, ord=2)))

    overlaps = np.array(overlaps)
    ar_med = float(np.median(aranks))
    res = {
        'policy': args.policy,
        'n': int(len(overlaps)),
        'action_rank_med': ar_med,
        'overlap_med': float(np.median(overlaps)),
        'overlap_p25': float(np.percentile(overlaps, 25)),
        'overlap_p75': float(np.percentile(overlaps, 75)),
        'random_baseline': ar_med / D,  # expected overlap of random subspaces
        'amp_on_action_med': float(np.median(act_amp)),
        'amp_off_action_med': float(np.median(inn_amp)),
        'verdict_gate': 'PASS' if np.median(overlaps) <= 0.3 else 'KILL',
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(args.policy,
          f"overlap={res['overlap_med']:.3f} (rand {res['random_baseline']:.3f}) "
          f"amp_on_act={res['amp_on_action_med']:.2f} amp_off={res['amp_off_action_med']:.2f} "
          f"arank={ar_med:.0f} -> {res['verdict_gate']}")


if __name__ == '__main__':
    main()
