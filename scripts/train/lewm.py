import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    unroll = int(cfg.wm.get('unroll', 0) or 0)
    unroll_sg = int(cfg.wm.get('unroll_sg', 0) or 0)
    if unroll_sg > 1:
        # L_new (theory_sufficiency_loss.md §5): encoder shaped ONLY by single-step
        # + SIGReg (keeps it planning-good); an anti-drift multi-step term trains the
        # PREDICTOR ONLY, with the encoder stop-gradded (sg) so it can't shed info to
        # cheat multi-step drift. total pred_loss = single_step + beta * multistep_sg.
        hs = ctx_len
        beta = float(cfg.wm.get('beta', 1.0))
        # single-step term (shapes phi + f)
        pred_ss = self.model.predict(emb[:, :hs], act_emb[:, :hs])   # (B,hs,D)
        loss_ss = (pred_ss - emb[:, 1:hs + 1]).pow(2).mean()
        # multi-step-sg term (predictor-only): encoder detached in seed AND target
        emb_sg = emb.detach()
        hist = list(emb_sg[:, :hs].unbind(dim=1))
        preds = []
        for s in range(unroll_sg):
            e = hs - 1 + s
            ctx = torch.stack(hist[-hs:], dim=1)
            actw = act_emb[:, e - hs + 1:e + 1]        # action_encoder is part of f: keep grad
            nxt = self.model.predict(ctx, actw)[:, -1]
            preds.append(nxt)
            hist.append(nxt)
        pred_ms = torch.stack(preds, dim=1)
        loss_ms = (pred_ms - emb_sg[:, hs:hs + unroll_sg]).pow(2).mean()
        output['pred_loss'] = loss_ss + beta * loss_ms
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
        losses_dict = {
            f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
        }
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return output
    unroll_tf = int(cfg.wm.get('unroll_tf', 0) or 0)
    if unroll > 1:
        # multi-step OPEN-LOOP unroll: seed with ctx_len true frames, feed predictions
        # back for `unroll` steps, compare to true future frames. Encoder co-trained via
        # both seed and target embeddings. (window length = ctx_len + unroll)
        hs = ctx_len
        hist = list(emb[:, :hs].unbind(dim=1))
        preds = []
        for s in range(unroll):
            e = hs - 1 + s
            ctx = torch.stack(hist[-hs:], dim=1)            # (B,hs,D)
            actw = act_emb[:, e - hs + 1:e + 1]             # (B,hs,A)
            nxt = self.model.predict(ctx, actw)[:, -1]      # predict frame e+1
            preds.append(nxt)
            hist.append(nxt)
        pred_emb = torch.stack(preds, dim=1)               # (B,unroll,D)
        tgt_emb = emb[:, hs:hs + unroll]
    elif unroll_tf > 1:
        # TEACHER-FORCED multi-horizon (Fast-LeWM mechanism control): identical
        # window and supervision positions as open-loop unroll, but every context
        # is TRUE embeddings — no self-composition, so no gradient through
        # Jacobian products. Isolates multi-step supervision from composition.
        hs = ctx_len
        preds = []
        for s in range(unroll_tf):
            e = hs - 1 + s
            ctx = emb[:, e - hs + 1:e + 1]                  # (B,hs,D) true frames
            actw = act_emb[:, e - hs + 1:e + 1]             # (B,hs,A)
            preds.append(self.model.predict(ctx, actw)[:, -1])
        pred_emb = torch.stack(preds, dim=1)               # (B,unroll_tf,D)
        tgt_emb = emb[:, hs:hs + unroll_tf]
    elif int(cfg.wm.get('echo_m', 0) or 0) > 0:
        # EchoReg: single-step supervision (same positions as K=1 baseline) plus
        # a one-sided CRITICAL echo penalty. Inject noise delta at the last
        # context frame, roll nominal and perturbed branches echo_m steps with
        # true actions (no future targets needed), and penalize amplification
        # beyond 1. Noise = the unpredictable channel, so the action channel is
        # untouched (asymmetry theorem used constructively); relu is one-sided
        # so collapse is never rewarded; gradient reaches the encoder via ctx.
        hs = ctx_len
        m = int(cfg.wm.echo_m)
        eps = float(cfg.wm.get('echo_eps', 1.0))
        beta = float(cfg.wm.get('echo_beta', 1.0))
        # single-step term, identical to the K=1 baseline objective
        pred_emb = self.model.predict(emb[:, :hs], act_emb[:, :hs])
        tgt_emb = emb[:, 1:hs + 1]
        # greedy renormalized echo: isotropic noise barely feels sigma_max in
        # 192-d (measured: median gain 0.58 on the K=1 model), so after each
        # step the perturbation is renormalized to eps along the amplified
        # DIRECTION (detached) — a per-step power iteration along the rollout.
        # Penalizes per-step worst-direction gain above 1 (one-sided, critical
        # target); eps=1.0 ~ the physical one-step error scale (drift1~1.05).
        delta = torch.randn_like(emb[:, hs - 1])
        delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * eps
        hist_n = list(emb[:, :hs].unbind(dim=1))
        hist_p = hist_n[:-1] + [hist_n[-1] + delta]
        echo = 0.0
        for s in range(m):
            e = hs - 1 + s
            actw = act_emb[:, e - hs + 1:e + 1]
            nxt_n = self.model.predict(torch.stack(hist_n[-hs:], dim=1), actw)[:, -1]
            nxt_p = self.model.predict(torch.stack(hist_p[-hs:], dim=1), actw)[:, -1]
            diff = nxt_p - nxt_n
            gain = diff.norm(dim=-1, keepdim=True) / eps
            echo = echo + torch.relu(gain - 1.0).pow(2).mean()
            hist_n.append(nxt_n)
            direction = (diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-6)).detach()
            hist_p.append(nxt_n + direction * eps)
        output['echo_loss'] = echo / m
        output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean() + beta * output['echo_loss']
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
        losses_dict = {
            f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
        }
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return output
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == '__main__':
    run()
