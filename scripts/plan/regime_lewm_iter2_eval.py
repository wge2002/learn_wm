"""Step B iteration 2 eval — does end-to-end MoE training make the regime routable?

Loads the from-scratch baseline + MoE LeWM checkpoints, re-encodes contact-labeled
expert windows with each model's OWN (newly trained) encoder, and measures:
  - MoE gate -> contact NMI/acc  (the key question: did encoder co-adaptation make the
    contact regime emerge in the gate, unlike the frozen-encoder iteration 1?)
  - open-loop drift mse@k (normalized by latent variance) for baseline vs MoE.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import latent_drift_phase3 as p3
import regime_existence_stepA as stepA
import json as _json
from hydra.utils import instantiate
from omegaconf import OmegaConf

_CKPT = Path("/home/jovyan/.stable_worldmodel/checkpoints")


def load_ckpt(sub, epoch):
    """Manual loader: config.json holds the FULL train cfg, so instantiate cfg['model']
    (the LeWM subtree) and load the weights state_dict."""
    base = _CKPT / sub
    cfg = _json.load(open(base / "config.json"))
    model = instantiate(OmegaConf.create(cfg["model"]))
    sd = torch.load(base / f"weights_epoch_{epoch}.pt", map_location="cpu")
    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--num-samples", type=int, default=1500)
    ap.add_argument("--max-k", type=int, default=10)
    ap.add_argument("--action-block", type=int, default=5)
    ap.add_argument("--goal-offset", type=int, default=50)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dataset-name",
                    default="/home/jovyan/.stable_worldmodel/pusht_expert_train.h5")
    ap.add_argument("--models", default="baseline:iter2_baseline,moe:iter2_moe",
                    help="comma-separated name:checkpoint_subdir")
    ap.add_argument("--output", default="outputs/regime_stepB2/iter2_eval.json")
    args = ap.parse_args()
    model_list = [tuple(m.split(":")) for m in args.models.split(",")]

    p3.configure_torch_threads_from_env()
    device = torch.device(args.device)
    hs = 3

    dataset = p3.swm.data.load_dataset(
        args.dataset_name, keys_to_load=["action", "state", "episode_idx", "step_idx"],
        keys_to_cache=["action", "state"])
    batch = p3.build_window_batch(dataset, num_samples=args.num_samples, max_k=args.max_k,
                                  goal_offset=args.goal_offset,
                                  action_block=args.action_block, seed=args.seed)
    print("[iter2-eval] replay for frames + contact", flush=True)
    frames, _cmax, contact_frac, _s, _t = stepA.replay_windows(
        env_name="swm/PushT-v1", init_states=batch.init_states,
        goal_states=batch.goal_states, raw_actions=batch.raw_actions,
        action_block=args.action_block, img_size=args.img_size, seed=args.seed)
    contact_bin = (contact_frac > 0.0).astype(int)            # (N, K+1)
    actions = batch.model_actions.astype(np.float32)          # (N, K, adim)

    results = {}
    for name, sub in model_list:
        print(f"[iter2-eval] loading {sub}/weights_epoch_{args.epoch}.pt", flush=True)
        model = load_ckpt(sub, args.epoch).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

        z_true = p3.encode_frames(model=model, frames=frames,
                                  batch_size=args.batch_size, device=device)  # (N,K+1,D)
        # 3-frame-seed open-loop latent rollout (matches both models' training seed;
        # model.rollout's 1-frame seed is unfair to the multi-step model).
        zt = torch.from_numpy(z_true).to(device)
        at = torch.from_numpy(actions).to(device)
        full = zt.size(1) - hs
        with torch.no_grad():
            histl = list(zt[:, :hs].unbind(dim=1))
            preds = []
            for s in range(full):
                e = hs - 1 + s
                ctx = torch.stack(histl[-hs:], dim=1)
                ae = model.action_encoder(at[:, e - hs + 1:e + 1])
                nxt = model.predict(ctx, ae)[:, -1]
                preds.append(nxt); histl.append(nxt)
            predt = torch.stack(preds, dim=1)                # (N, full, D)
            tgtt = zt[:, hs:hs + full]
            mse = ((predt - tgtt) ** 2).mean(dim=(0, 2)).cpu().numpy()  # (full,)
        var = float(z_true.var(axis=0).mean())
        norm = (mse / max(var, 1e-9))
        entry = {"mse_vs_k": mse.tolist(), "latent_var": var,
                 "norm_mse_vs_k": norm.tolist(), "norm_mse_at_k": float(norm[-1])}

        # MoE gate -> contact alignment on TRUE windows (best-case, teacher-forced)
        is_moe = hasattr(model.predictor, "num_experts")
        if is_moe:
            zt = torch.from_numpy(z_true).to(device)
            at = torch.from_numpy(actions).to(device)
            codes, labels = [], []
            for k in range(hs - 1, actions.shape[1]):
                emb_w = zt[:, k - hs + 1:k + 1]
                ae = model.action_encoder(at[:, k - hs + 1:k + 1])
                _ = model.predict(emb_w, ae)
                lg = model.predictor.last_gate_logits          # (N,hs,K)
                codes.append(lg[:, -1].argmax(-1).cpu().numpy())
                labels.append(contact_bin[:, k + 1])
            codes = np.concatenate(codes); labels = np.concatenate(labels)
            nmi = float(normalized_mutual_info_score(labels, codes))
            acc = (labels == codes).mean(); acc = float(max(acc, 1 - acc))
            _, cnt = np.unique(codes, return_counts=True)
            entry["gate_contact_nmi"] = nmi
            entry["gate_contact_acc"] = acc
            entry["code_usage"] = cnt.tolist()
            entry["contact_rate"] = float(labels.mean())
        results[name] = entry
        msg = (f"[{name}] norm_mse@{args.max_k}={norm[-1]:.3f} (var={var:.3f})")
        if is_moe:
            msg += f"  gate->contact NMI={entry['gate_contact_nmi']:.3f} acc={entry['gate_contact_acc']:.3f}"
        print(msg, flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("\n[iter2-eval] summary (norm mse@k, lower=better):")
    for name, e in results.items():
        extra = (f"  gate->contact NMI={e['gate_contact_nmi']:.3f}"
                 if "gate_contact_nmi" in e else "")
        print(f"  {name:<12} {e['norm_mse_at_k']:.3f}{extra}")
    print(f"[iter2-eval] wrote {args.output}")


if __name__ == "__main__":
    main()
