"""Direction B veto gate: does PushT have modelable stochasticity, and does it
live along the error-amplification directions?

Generative-LeWM (Direction B) posits z' = f(z,a) + Sigma(z)^{1/2} eps with an
innovation whose covariance is shaped by the amplification law. That is only a
model, not a prior-on-noise, if (1) the true transition has nonzero conditional
variance given (z,a), and (2) that variance is anisotropic and aligns with the
top error-amplification directions (else there is nothing structured to model).

We approximate the conditional variance by grouping near-duplicate (state,
action) pairs from the probe rollouts and measuring the spread of their next
latents. PushT expert data may be near-deterministic; this gate says so cheaply.

Kill Direction B if median conditional std / latent scale < 0.02 (effectively
deterministic) OR the conditional covariance is near-isotropic (top/mean
eigenvalue ratio < 2, nothing anisotropic to allocate).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import stable_worldmodel as swm  # noqa: E402

HS = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--num', type=int, default=4000)
    ap.add_argument('--knn', type=int, default=12, help='neighbors per (state,action) group')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = swm.wm.utils.load_pretrained(args.policy).to(dev).eval()
    model.requires_grad_(False)
    d = np.load(args.data)
    z = d['z'][: args.num].astype(np.float32)  # (N,L,D)
    a = d['a'][: args.num].astype(np.float32)  # (N,L,Araw)
    s = d['states'][: args.num].astype(np.float32)  # (N,L,S) true env state
    N, L, D = z.shape

    # transition tuples at a mid frame: (true_state, action) -> next latent
    t = HS - 1 + 2
    S_t = s[:, t]                       # (N,S)
    A_t = a[:, t]                       # (N,Araw)
    Znext = z[:, t + 1]                 # (N,D)
    key = np.concatenate([S_t, A_t], axis=1)  # (N, S+Araw)
    key = (key - key.mean(0)) / (key.std(0) + 1e-6)

    # group by nearest neighbors in (state,action): spread of next latent = cond var
    from scipy.spatial import cKDTree
    tree = cKDTree(key)
    dists, idx = tree.query(key, k=args.knn + 1)  # includes self
    # only keep groups whose neighbors are genuinely close in (s,a)
    close = dists[:, -1] < np.percentile(dists[:, -1], 40)  # tight 40% groups
    znext_scale = float(np.linalg.norm(Znext - Znext.mean(0), axis=1).mean())

    cond_covs = []
    cond_stds = []
    for i in np.nonzero(close)[0]:
        grp = Znext[idx[i]]            # (knn+1, D)
        c = np.cov(grp.T)             # (D,D)
        cond_covs.append(c)
        cond_stds.append(np.sqrt(np.trace(c)))
    cond_covs = np.array(cond_covs)
    mean_cov = cond_covs.mean(0)
    ev = np.linalg.eigvalsh(mean_cov)[::-1].clip(min=0)
    aniso = float(ev[0] / (ev.mean() + 1e-9))
    cond_std = float(np.median(cond_stds))

    # alignment with amplification dirs: leading cond-cov eigenvector vs top
    # singular vector of a latent-transition Jacobian (reuse mid-frame autograd)
    zt = torch.from_numpy(z[:64, t - 2 : t + 1]).to(dev)
    at = torch.from_numpy(a[:64, t - 2 : t + 1]).to(dev)
    aet = model.action_encoder(at)
    amp_overlap = []
    top_condvec = np.linalg.eigh(mean_cov)[1][:, -1]  # (D,)
    tv = torch.from_numpy(top_condvec.astype(np.float32)).to(dev)
    for i in range(zt.shape[0]):
        zl = zt[i, -1].detach().requires_grad_(True)

        def g(x):
            c = torch.cat([zt[i, :-1], x.unsqueeze(0)], 0).unsqueeze(0)
            return model.predict(c, aet[i].unsqueeze(0))[0, -1]

        J = torch.autograd.functional.jacobian(g, zl, vectorize=True)
        U, _, _ = torch.linalg.svd(J, full_matrices=False)
        amp_overlap.append(float((U[:, :3].T @ tv).pow(2).sum()))  # cond-var dir in top-3 amp span

    ratio = cond_std / (znext_scale + 1e-9)
    verdict = 'PASS' if (ratio >= 0.02 and aniso >= 2.0) else 'KILL'
    res = {
        'policy': args.policy,
        'n_groups': int(close.sum()),
        'cond_std_med': cond_std,
        'znext_scale': znext_scale,
        'cond_std_ratio': ratio,
        'cond_cov_anisotropy': aniso,
        'condvar_in_amp_span_med': float(np.median(amp_overlap)),
        'verdict_gate': verdict,
        'reason': ('deterministic' if ratio < 0.02 else
                   'isotropic' if aniso < 2.0 else 'structured stochasticity'),
    }
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(args.policy,
          f"cond_std/scale={ratio:.4f} aniso={aniso:.2f} "
          f"condvar_in_amp={res['condvar_in_amp_span_med']:.3f} -> {verdict} ({res['reason']})")


if __name__ == '__main__':
    main()
