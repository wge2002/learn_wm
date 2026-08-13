import hashlib
import os
from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from torchvision.transforms import v2

from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from stable_worldmodel.wm.utils import save_pretrained


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor content independently of torch.save container metadata."""

    digest = hashlib.sha256()
    for key, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(b'\0')
        digest.update(str(tensor.dtype).encode())
        digest.update(b'\0')
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(b'\0')
        digest.update(
            tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def export_initial_weights(model, path: Path) -> tuple[str, bool]:
    """Atomically export, or bitwise-validate, a shared initialization."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    if path.exists():
        existing = torch.load(path, map_location='cpu', weights_only=True)
        if existing.keys() != state.keys() or any(
            not torch.equal(existing[key], value.detach().cpu())
            for key, value in state.items()
        ):
            raise FileExistsError(
                f'initialization artifact exists but does not match the '
                f'deterministic model for this config/seed: {path}'
            )
        return state_dict_sha256(state), True

    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return state_dict_sha256(state), False


class ResizeField:
    """Resize one image field without relying on SPT's torchvision internals."""

    def __init__(self, size: int, source: str, target: str):
        self.resize = v2.Resize(size)
        self.source = source
        self.target = target

    def __call__(self, sample):
        sample[self.target] = self.resize(sample[self.source])
        return sample


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = ResizeField(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


def preprocess_pixels_on_device(batch, img_size: int):
    """Apply the ImageNet pixel transform after Lightning moves a batch.

    HDF5 clips are stored as uint8 tensors. Keeping them uint8 through the
    DataLoader substantially reduces worker CPU time and shared-memory
    traffic; resize and normalization are then batched on the accelerator.
    This is opt-in so existing runs retain their original preprocessing path.
    """

    pixels = batch['pixels']
    if pixels.ndim != 5:
        raise ValueError(
            'Expected pixels with shape (batch, time, channels, height, width), '
            f'got {tuple(pixels.shape)}'
        )

    batch_size, num_steps, channels, height, width = pixels.shape
    pixels = pixels.reshape(batch_size * num_steps, channels, height, width)
    if not pixels.is_floating_point():
        pixels = pixels.to(torch.float32).div_(255.0)

    mean = pixels.new_tensor(dt.dataset_stats.ImageNet['mean']).view(
        1, -1, 1, 1
    )
    std = pixels.new_tensor(dt.dataset_stats.ImageNet['std']).view(
        1, -1, 1, 1
    )
    pixels = pixels.sub(mean).div(std)
    if (height, width) != (img_size, img_size):
        pixels = F.interpolate(
            pixels,
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False,
            antialias=True,
        )

    batch['pixels'] = pixels.reshape(
        batch_size, num_steps, channels, img_size, img_size
    )
    return batch


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(
        self,
        run_name,
        cfg,
        epoch_interval: int = 1,
        epoch_offset: int = 0,
    ):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.epoch_offset = epoch_offset

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            local_epoch = trainer.current_epoch + 1
            exported_epoch = self.epoch_offset + local_epoch
            saved = False
            if exported_epoch % self.epoch_interval == 0:
                self._save(pl_module.model, exported_epoch)
                saved = True

            # save final epoch
            if local_epoch == trainer.max_epochs and not saved:
                self._save(pl_module.model, exported_epoch)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


class NaNGuardCallback(Callback):
    """Abort the run as soon as the loss becomes non-finite.

    A diverged run is unrecoverable: every later epoch trains on NaN weights and
    ``SaveCkptCallback`` happily writes them out, so the run looks successful and
    only fails much later at eval. ``curv_d192`` went NaN at step 247 of epoch 0
    and still burned all 30 epochs before anyone noticed. Failing loudly here
    turns a 24-hour waste into a 70-second one.
    """

    def __init__(self, patience: int = 3):
        super().__init__()
        self.patience = patience
        self.strikes = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        if loss is None or torch.isfinite(torch.as_tensor(loss)).all():
            self.strikes = 0
            return

        self.strikes += 1
        step = trainer.global_step
        print(
            f'[nan-guard] non-finite loss at epoch {trainer.current_epoch} '
            f'step {step} ({self.strikes}/{self.patience})',
            flush=True,
        )
        if self.strikes >= self.patience:
            raise RuntimeError(
                f'[nan-guard] loss non-finite for {self.patience} consecutive '
                f'batches (epoch {trainer.current_epoch}, step {step}); '
                f'aborting instead of writing NaN checkpoints'
            )


class NonFiniteGradGuardCallback(Callback):
    """Neutralize a non-finite gradient step instead of letting it kill the run.

    This restores, for the bf16 path, the one protection ``precision: fp16``
    would have given for free. Root cause of the h2hfix wave's NaN collapse
    (7 of 8 runs, at epochs 5-14 with no gradual divergence beforehand):

    1. one rare batch produces ``inf`` in a SINGLE gradient element;
    2. ``gradient_clip_val`` computes ``total_norm = inf``, hence
       ``clip_coef = 1.0 / inf = 0``, and multiplies every gradient by it.
       Healthy gradients become a harmless ``0`` -- but ``inf * 0 = NaN``;
    3. AdamW writes that NaN into the parameter AND into its ``exp_avg`` /
       ``exp_avg_sq``, so the poisoning is permanent;
    4. next step the forward emits a NaN loss, so ALL gradients are NaN,
       ``total_norm`` is NaN, ``clip_coef`` is NaN, and now *every* parameter
       is NaN. The model is already dead two steps before ``NaNGuardCallback``
       (patience 3) reports anything.

    ``fp16`` never reaches step 2: ``GradScaler`` inspects the gradients, skips
    the optimizer step, and training continues. ``bf16`` has no ``GradScaler``
    (it does not need loss scaling), so nothing intercepts the ``inf`` -- one
    unlucky batch in ~100k steps is fatal. This is independent of ``aux_reg``,
    ``aux_space`` and ``aux_beta_mode``, which is why static- and adaptive-beta
    runs of both ``curvature`` and ``bisim`` all died the same way.

    Setting every gradient to ``None`` makes AdamW skip every parameter: no
    momentum, weight-decay, or optimizer-state update occurs. This is the
    closest bf16 analogue of GradScaler declining ``optimizer.step``. The
    count is reported at the end of each epoch: a handful of skips is normal
    numerical noise, a persistently rising count means the recipe itself is
    unstable and the aux term still needs work.
    """

    def __init__(
        self,
        max_skip_frac: float = 0.01,
        min_steps_for_frac: int = 1000,
        max_total_skips: int | None = None,
        policy: str = 'skip',
    ):
        super().__init__()
        if policy not in {'skip', 'error'}:
            raise ValueError(
                f'non-finite gradient policy must be skip or error, got {policy!r}'
            )
        self.max_skip_frac = max_skip_frac
        self.min_steps_for_frac = min_steps_for_frac
        if max_total_skips is not None and max_total_skips < 0:
            raise ValueError('max_total_skips must be non-negative or None')
        self.max_total_skips = max_total_skips
        self.policy = policy
        self.skipped = 0
        self.epoch_skipped = 0
        self.epoch_steps = 0

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Runs after backward and BEFORE gradient_clip_val, which is the only
        # window where an inf can still be caught before it becomes a NaN.
        self.epoch_steps += 1
        parameters = [
            p for p in pl_module.parameters() if p.grad is not None
        ]
        if not parameters:
            return
        grads = [p.grad for p in parameters]
        # Regression hook: SWM_INJECT_INF_AT_STEP=<n> plants a single inf in one
        # gradient element at that step, which is exactly the h2hfix trigger.
        # Without the guard below the run is dead one step later; with it the
        # loss stays finite. Test-only, off unless the variable is set.
        inject = os.environ.get('SWM_INJECT_INF_AT_STEP')
        if inject and trainer.global_step == int(inject):
            grads[0].view(-1)[0] = float('inf')
            print(
                f'[inject] planted one inf gradient element at step '
                f'{trainer.global_step}',
                flush=True,
            )
        # One fused reduction, then ONE host sync. Testing each tensor with
        # `not torch.isfinite(g).all()` would sync once per tensor (~150 for
        # ViT-tiny) every step, which is a large throughput tax to pay for a
        # branch that almost never fires.
        finite = torch.stack(
            [torch.isfinite(g).all() for g in grads]
        ).all()
        if bool(finite):
            return

        # This branch is intentionally expensive: it only runs after a bad
        # gradient has already been detected.  Preserve enough evidence to
        # distinguish a rare bf16 overflow from gradual model divergence.
        parameter_names = {
            id(parameter): name
            for name, parameter in pl_module.named_parameters()
        }
        bad_gradients = []
        for parameter, grad in zip(parameters, grads, strict=True):
            finite_mask = torch.isfinite(grad)
            if bool(finite_mask.all()):
                continue
            grad32 = grad.detach().float()
            finite_values = grad32[torch.isfinite(grad32)]
            finite_max = (
                float(finite_values.abs().max())
                if finite_values.numel()
                else float('nan')
            )
            bad_gradients.append(
                f'{parameter_names.get(id(parameter), "<unnamed>")}'
                f' shape={tuple(grad.shape)} dtype={grad.dtype}'
                f' nan={int(torch.isnan(grad).sum())}'
                f' +inf={int(torch.isposinf(grad).sum())}'
                f' -inf={int(torch.isneginf(grad).sum())}'
                f' finite_max_abs={finite_max:.6g}'
            )
        losses = getattr(pl_module, '_swm_last_loss_components', {})
        loss_summary = ' '.join(
            f'{name}={float(value):.9g}' for name, value in losses.items()
        )
        batch_fields = getattr(pl_module, '_swm_last_batch_nonpixel', {})
        batch_digest = hashlib.sha256()
        for name in sorted(batch_fields):
            value = batch_fields[name].detach().cpu().contiguous()
            batch_digest.update(name.encode())
            batch_digest.update(str(tuple(value.shape)).encode())
            batch_digest.update(str(value.dtype).encode())
            batch_digest.update(value.view(torch.uint8).numpy().tobytes())
        print(
            f'[grad-guard] evidence epoch={trainer.current_epoch} '
            f'step={trainer.global_step} losses=({loss_summary}) '
            f'batch_sha256={batch_digest.hexdigest() if batch_fields else "n/a"}',
            flush=True,
        )
        for detail in bad_gradients:
            print(f'[grad-guard] offending {detail}', flush=True)

        evidence_dir = os.environ.get('SWM_NONFINITE_EVIDENCE_DIR')
        if evidence_dir:
            evidence_root = Path(evidence_dir).expanduser().resolve()
            evidence_root.mkdir(parents=True, exist_ok=True)
            evidence_path = evidence_root / (
                f'nonfinite_e{trainer.current_epoch}_s{trainer.global_step}.pt'
            )
            torch.save(
                {
                    'epoch': int(trainer.current_epoch),
                    'global_step': int(trainer.global_step),
                    'losses': {
                        name: value.detach().cpu()
                        for name, value in losses.items()
                    },
                    'batch_nonpixel': {
                        name: value.detach().cpu()
                        for name, value in batch_fields.items()
                    },
                    'offending': bad_gradients,
                    'model_state_dict': {
                        name: value.detach().cpu()
                        for name, value in pl_module.state_dict().items()
                    },
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                evidence_path,
            )
            print(
                f'[grad-guard] wrote evidence bundle {evidence_path}',
                flush=True,
            )

        self.skipped += 1
        self.epoch_skipped += 1
        # ``zero_`` is not a true skip for AdamW: it still applies decoupled
        # weight decay and advances momentum. ``grad=None`` makes the optimizer
        # omit the parameter entirely, leaving both weights and moments exact.
        for parameter in parameters:
            parameter.grad = None
        if self.policy == 'error':
            raise FloatingPointError(
                f'non-finite gradient at epoch {trainer.current_epoch} '
                f'global_step {trainer.global_step}; formal paired runs '
                'require every optimizer update to be valid'
            )
        if (
            self.max_total_skips is not None
            and self.skipped > self.max_total_skips
        ):
            raise RuntimeError(
                f'[grad-guard] total non-finite skips {self.skipped} exceed '
                f'the preregistered limit {self.max_total_skips}'
            )
        if self.skipped <= 20 or self.skipped % 100 == 0:
            print(
                f'[grad-guard] skipped non-finite gradient at epoch '
                f'{trainer.current_epoch} step {trainer.global_step} '
                f'({self.skipped} total)',
                flush=True,
            )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.epoch_skipped:
            frac = self.epoch_skipped / max(self.epoch_steps, 1)
            print(
                f'[grad-guard] epoch {trainer.current_epoch}: '
                f'{self.epoch_skipped}/{self.epoch_steps} steps skipped '
                f'({frac:.3%})',
                flush=True,
            )
            # The fraction is only meaningful over a real epoch. A short probe
            # or smoke run (14 steps) puts a single unlucky step at 7%, which
            # would abort a perfectly healthy configuration.
            if (
                frac > self.max_skip_frac
                and self.epoch_steps >= self.min_steps_for_frac
            ):
                raise RuntimeError(
                    f'[grad-guard] {frac:.2%} of steps in epoch '
                    f'{trainer.current_epoch} had non-finite gradients '
                    f'({self.epoch_skipped}/{self.epoch_steps}, '
                    f'> {self.max_skip_frac:.2%}); the recipe is unstable, '
                    f'not just unlucky -- aborting'
                )
        self.epoch_skipped = 0
        self.epoch_steps = 0


class DivergenceTraceCallback(Callback):
    """Ring-buffer trace of the steps leading up to divergence (opt-in).

    Enabled with ``SWM_TRACE_DIVERGENCE=1``. Answers the one question a NaN loss
    alone cannot: did the FORWARD blow up (some intermediate overflowed) or did
    the PARAMETERS already contain NaN from a previous step's backward? The
    per-step gradient norm before clipping distinguishes a genuine gradient
    explosion from a silent numerical hole.
    """

    def __init__(self, window: int = 12):
        super().__init__()
        self.window = window
        self.rows = []
        self.dumped = False
        self.last_grad = None

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        # Runs after backward, BEFORE gradient_clip_val is applied, so this is
        # the raw gradient norm.
        total, nonfinite = 0.0, 0
        for p in pl_module.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            if not torch.isfinite(g).all():
                nonfinite += 1
            total += g.pow(2).nansum().item()
        self.last_grad = (total ** 0.5, nonfinite)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        bad_params = sum(
            1 for p in pl_module.parameters() if not torch.isfinite(p).all()
        )
        gnorm, gbad = self.last_grad if self.last_grad else (float('nan'), -1)
        vals = {}
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if 'loss' in k and torch.is_tensor(v):
                    vals[k] = v.detach().float().item()
        self.rows.append(
            f'step={trainer.global_step:5d} '
            + ' '.join(f'{k}={v:.4g}' for k, v in sorted(vals.items()))
            + f' |grad|={gnorm:.4g} grad_nonfinite_tensors={gbad}'
            f' param_nonfinite_tensors={bad_params}'
        )
        if len(self.rows) > self.window:
            self.rows.pop(0)

        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        diverged = loss is not None and not torch.isfinite(
            torch.as_tensor(loss)
        ).all()
        if (diverged or bad_params) and not self.dumped:
            self.dumped = True
            print(
                f'\n[trace] divergence detected at step {trainer.global_step}; '
                f'last {len(self.rows)} steps:',
                flush=True,
            )
            for r in self.rows:
                print(f'[trace]   {r}', flush=True)


class CritWMStateCallback(Callback):
    """Persist CritWM's non-parameter controller state in Lightning ckpts."""

    checkpoint_key = 'critwm_state'

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if not hasattr(pl_module, '_critwm_gamma'):
            return
        checkpoint[self.checkpoint_key] = {
            'gamma': float(pl_module._critwm_gamma),
            'rate': float(pl_module._critwm_rate),
            'counter': int(pl_module._critwm_ctr),
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        state = checkpoint.get(self.checkpoint_key)
        if state is None:
            return
        pl_module._critwm_gamma = float(state['gamma'])
        pl_module._critwm_rate = float(state['rate'])
        pl_module._critwm_ctr = int(state['counter'])


class PairingTraceCallback(Callback):
    """Hash the first training batches so paired arms can prove data identity."""

    def __init__(self, num_batches: int):
        super().__init__()
        self.num_batches = num_batches

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ):
        if trainer.current_epoch != 0 or batch_idx >= self.num_batches:
            return
        digest = hashlib.sha256()
        keys = []
        for key in ('action', 'state', 'proprio', 'observation'):
            value = batch.get(key)
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().cpu().contiguous()
            keys.append(key)
            digest.update(key.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        if not keys:
            raise ValueError('pairing trace found no non-pixel tensor fields')
        print(
            f'[pairing] epoch=0 batch={batch_idx} '
            f'keys={",".join(keys)} sha256={digest.hexdigest()}',
            flush=True,
        )


def matched_one_step_prediction(model, emb, act_emb, history_size: int):
    """Use a long common clip while retaining the historical K=1 loss."""

    if emb.size(1) < history_size + 1:
        raise ValueError(
            f'matched K=1 needs at least {history_size + 1} frames, '
            f'got {emb.size(1)}'
        )
    prediction = model.predict(
        emb[:, :history_size], act_emb[:, :history_size]
    )
    target = emb[:, 1 : history_size + 1]
    return prediction, target


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    if cfg.get('gpu_image_preprocess', False):
        batch = preprocess_pixels_on_device(batch, cfg.img_size)

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    if stage == 'fit':
        self._swm_last_batch_nonpixel = {
            name: value.detach()
            for name, value in batch.items()
            if name != 'pixels' and torch.is_tensor(value)
        }

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    unroll = int(cfg.wm.get('unroll', 0) or 0)
    unroll_sg = int(cfg.wm.get('unroll_sg', 0) or 0)
    matched_one_step = bool(cfg.wm.get('matched_one_step', False))
    if matched_one_step and (
        unroll > 1
        or unroll_sg > 1
        or int(cfg.wm.get('unroll_tf', 0) or 0) > 1
    ):
        raise ValueError(
            'matched_one_step cannot be combined with a multi-step objective'
        )
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
    mix_gamma = float(cfg.wm.get('mix_gamma', 0) or 0)
    loss_ss_mix = None
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
        if mix_gamma > 0:
            # gamma-dose hybrid (Part VIII v2): full-weight K=1 objective plus
            # gamma * the open-loop K-step term. Keeps gain pressure and
            # accuracy COUPLED at the same rollout points (the echo v1 lesson);
            # gamma turns the integer horizon knob into a continuous dose.
            pred_ss = self.model.predict(emb[:, :hs], act_emb[:, :hs])
            loss_ss_mix = (pred_ss - emb[:, 1:hs + 1]).pow(2).mean()
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
    elif cfg.wm.get('aux_reg', None):
        # HEAD-TO-HEAD baselines: K=1 single-step objective + a competitor
        # auxiliary geometry regularizer, to test whether one-step + geometry
        # produces the far-goal composition gain that coupled multi-step does.
        #   'curvature' = Temporal Straightening (penalize TRUE trajectory
        #                 tangent curvature; on-trajectory geometry).
        #   'bisim'     = Invariant-JEPA-WM style reward-free bisimulation
        #                 (latent metric = discounted next-latent metric).
        # Prediction stays single-step (identical to K=1); only the aux term and
        # a longer true-trajectory window (num_steps) differ.
        hs = ctx_len
        reg = str(cfg.wm.aux_reg)
        beta = float(cfg.wm.get('aux_beta', 1.0))
        pred_emb = self.model.predict(emb[:, :hs], act_emb[:, :hs])
        tgt_emb = emb[:, 1:hs + 1]
        base = (pred_emb - tgt_emb).pow(2).mean()
        # `aux_space` picks the representation the AUXILIARY is measured in;
        # `base` always stays in the raw latent space. 'unit' projects onto the
        # sphere, which closes the curvature term's degenerate escape route: a
        # large common drift makes every raw velocity near-parallel, so raw
        # `1-cos` falls to 0.007 at drift=20 while |emb| grows to ~970 and the
        # raw-space `base` (an MSE, so quadratic in scale) explodes with it.
        # Measured on the sphere the same drift only moves it 1.497 -> 1.375.
        # Curvature is a pure direction quantity, so 'unit' is also the more
        # faithful reading of Temporal Straightening.
        # 'unit' is WRONG for bisim: pairwise differences already cancel a
        # common drift (raw stays 5.148 for any drift), and normalizing opens a
        # fresh hole instead (0.0002 at drift=20). Leave bisim in 'raw'.
        aux_space = str(cfg.wm.get('aux_space', 'raw'))

        def to_aux_space(z):
            if aux_space == 'raw':
                return z
            if aux_space == 'unit':
                return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            raise ValueError(f'unknown aux_space {aux_space!r}')

        if reg == 'curvature':
            # Temporal tangent direction is undefined when two consecutive
            # observations have identical embeddings. PushT contains exact
            # duplicate-frame transitions: normalizing those zero velocities
            # injected a 1/eps gradient and eventually destabilized otherwise
            # finite multi-epoch runs. Compute the angle in fp32 and exclude
            # tangent pairs for which either velocity is too small.
            #
            # NB the duplicate-frame hypothesis was tested directly and is NOT
            # what killed the multi-epoch runs: `x.norm(dim=-1).clamp_min(eps)`
            # backward at an exact zero vector returns a finite 0 gradient on
            # torch 2.4.1, and the full curvature expression on two identical
            # frames produced 0/160 NaN. The real cause was bf16 + gradient
            # clipping (see NonFiniteGradGuardCallback). This masking is kept
            # anyway: it is the faithful reading of a direction-only quantity,
            # and fp32 + clamped cos is strictly more robust regardless.
            e = to_aux_space(emb)
            v = (e[:, 1:] - e[:, :-1]).float()                  # (B,T-1,D)
            min_speed = float(cfg.wm.get('curvature_min_speed', 0.1))
            if min_speed <= 0:
                raise ValueError(
                    f'curvature_min_speed must be positive, got {min_speed}'
                )
            speed = v.norm(p=2, dim=-1)
            # The floor has to be RELATIVE to the scale of the space it is
            # applied in. An absolute 0.1 was calibrated for raw latents, where
            # |emb| reaches 142-970 and speeds are O(1)-O(200); on the unit
            # sphere every embedding has norm 1 and the speeds of genuinely
            # adjacent frames fall far below 0.1, so an absolute floor masks
            # EVERY pair and the auxiliary silently becomes exactly 0.0 with no
            # gradient (measured: aux 0.76-1.46 -> 0.0 for aux_space=unit).
            # Scaling by the batch's own median speed keeps the intent -- drop
            # the near-duplicate tail -- in either space.
            speed_ref = speed.detach().median().clamp_min(1e-12)
            valid_speed = speed >= min_speed * speed_ref
            vn = v / speed.clamp_min(min_speed * speed_ref).unsqueeze(-1)
            cos = (vn[:, 1:] * vn[:, :-1]).sum(-1).clamp(-1, 1)
            valid_pair = valid_speed[:, 1:] & valid_speed[:, :-1]
            valid_pair_f = valid_pair.to(cos.dtype)
            n_valid = valid_pair_f.sum()
            aux = ((1.0 - cos) * valid_pair_f).sum()
            aux = aux / n_valid.clamp_min(1.0)
            # A fully-masked batch makes aux exactly 0.0 with no gradient, i.e.
            # the regularizer is silently off. That is the failure mode the
            # relative floor above fixes, so say so out loud instead of
            # training for 30 epochs on an inert term. Checked only on the
            # first few calls; each check costs one host sync.
            _n = getattr(lejepa_forward, '_mask_checks', 0)
            if _n < 20:
                lejepa_forward._mask_checks = _n + 1
                if float(n_valid) == 0.0:
                    print(
                        f'[curvature] WARNING: all {valid_pair_f.numel()} '
                        f'direction pairs fell below curvature_min_speed='
                        f'{min_speed} (relative to median speed '
                        f'{float(speed_ref):.3e} in aux_space='
                        f'{cfg.wm.get("aux_space", "raw")}); aux_loss is '
                        f'identically 0 and contributes no gradient',
                        flush=True,
                    )
        elif reg == 'bisim':
            gamma = float(cfg.wm.get('bisim_gamma', 0.9))
            z0 = to_aux_space(emb[:, hs - 1])                  # (B,D) current
            znext = to_aux_space(pred_emb[:, -1])              # (B,D) predicted next
            idx = torch.randperm(z0.shape[0], device=z0.device)
            dz = (z0 - z0[idx]).norm(dim=-1)
            dn = (znext - znext[idx]).norm(dim=-1).detach()
            aux = (dz - gamma * dn).pow(2).mean()
        else:
            raise ValueError(f'unknown aux_reg {reg!r}')
        # A STATIC beta cannot hold the auxiliary in its place. `base` shrinks
        # as the model learns (0.087 -> 0.049 over 30 epochs) while neither
        # auxiliary converges, so the measured beta*aux/base ratio climbs on its
        # own: bisim went 0.00 -> 1.73 and curv was already 0.94 in epoch 1
        # before going NaN. Both configs claim the auxiliary stays "below" the
        # prediction term; neither did.
        #
        # In adaptive mode beta is a TARGET RATIO instead of a raw weight: the
        # rescaling factor is detached, so the gradient direction of `aux` is
        # untouched and only its magnitude is pinned to beta*base. Whatever the
        # auxiliary's natural scale is, it can no longer outvote prediction.
        #
        # The 1e-8 floor bounds beta_eff's VALUE but not the gradient it
        # delivers: d(beta_eff*aux)/d(aux) carries the whole 1/aux factor, so as
        # aux -> 0 the gradient reaching the encoder grows without bound
        # (measured: aux 1e-1 -> 5e-3, 1e-3 -> 5e-1, 1e-9 -> 5e+4). That is a
        # confirmed route to an `inf` gradient, which
        # ``NonFiniteGradGuardCallback`` then has to absorb. Cap the multiplier
        # at aux_beta_max so a near-zero auxiliary cannot amplify at all; the
        # cap only binds when the auxiliary has already essentially converged,
        # where its gradient direction no longer carries information.
        if str(cfg.wm.get('aux_beta_mode', 'static')) == 'adaptive':
            beta_max = float(cfg.wm.get('aux_beta_max', 10.0))
            beta_eff = (
                beta * base.detach() / aux.detach().abs().clamp_min(1e-8)
            ).clamp_max(beta_max)
        else:
            beta_eff = torch.as_tensor(beta, device=aux.device, dtype=aux.dtype)
        output['aux_loss'] = aux
        # Diagnostic only: the fraction of `pred_loss` the auxiliary accounts
        # for. Should sit at `beta` in adaptive mode and drift in static mode.
        output['auxratio_loss'] = (
            beta_eff * aux.detach() / base.detach().abs().clamp_min(1e-8)
        )
        output['pred_loss'] = base + beta_eff * aux
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
        components = {
            'base': base,
            'aux': aux,
            'sigreg': output['sigreg_loss'],
            'total': output['loss'],
        }
        bad_components = [
            name
            for name, value in components.items()
            if not bool(torch.isfinite(value.detach()).all())
        ]
        if bad_components:
            summary = ', '.join(
                f'{name}={float(value.detach())}'
                for name, value in components.items()
            )
            raise FloatingPointError(
                f'non-finite {reg} loss components {bad_components}: {summary}'
            )
        losses_dict = {
            f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
        }
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
        return output
    elif int(cfg.wm.get('thermo_k', 0) or 0) > 1:
        # CritWM thermostat: closed-loop critical training. Sensor = greedy
        # renormalized echo probe (measurement only, no_grad — the EchoReg v1
        # lesson); actuator = gamma, the weight of the COUPLED open-loop
        # K-step term (the coupling thesis: the only non-cheatable pressure
        # channel); setpoint = rate 1.0.
        hs = ctx_len
        K = int(cfg.wm.thermo_k)
        eta = float(cfg.wm.get('thermo_eta', 2.0))
        target = float(cfg.wm.get('thermo_target', 1.0))
        every = int(cfg.wm.get('thermo_every', 200))
        if not hasattr(self, '_critwm_gamma'):
            self._critwm_gamma = float(cfg.wm.get('thermo_gamma0', 0.3))
            self._critwm_rate = float(cfg.wm.get('thermo_rate0', target))
            self._critwm_ctr = int(cfg.wm.get('thermo_ctr0', 0))
        # coupled open-loop K-step term
        hist = list(emb[:, :hs].unbind(dim=1))
        preds = []
        for s in range(K):
            e = hs - 1 + s
            ctx = torch.stack(hist[-hs:], dim=1)
            actw = act_emb[:, e - hs + 1:e + 1]
            nxt = self.model.predict(ctx, actw)[:, -1]
            preds.append(nxt)
            hist.append(nxt)
        loss_ms = (torch.stack(preds, dim=1) - emb[:, hs:hs + K]).pow(2).mean()
        # full-weight single-step term (K=1 baseline objective)
        pred_ss = self.model.predict(emb[:, :hs], act_emb[:, :hs])
        loss_ss = (pred_ss - emb[:, 1:hs + 1]).pow(2).mean()
        # sensor: periodic echo probe, measurement only
        if stage == 'fit':
            self._critwm_ctr += 1
            if self._critwm_ctr % every == 0:
                with torch.no_grad():
                    eps = 1.0
                    m = 5
                    z0 = emb.detach()
                    delta = torch.randn_like(z0[:, hs - 1])
                    delta = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6) * eps
                    hn = list(z0[:, :hs].unbind(dim=1))
                    hp = hn[:-1] + [hn[-1] + delta]
                    gains = []
                    for s in range(m):
                        e = hs - 1 + s
                        actw = act_emb[:, e - hs + 1:e + 1].detach()
                        nn_ = self.model.predict(torch.stack(hn[-hs:], dim=1), actw)[:, -1]
                        np_ = self.model.predict(torch.stack(hp[-hs:], dim=1), actw)[:, -1]
                        diff = np_ - nn_
                        gains.append((diff.norm(dim=-1) / eps).median())
                        hn.append(nn_)
                        hp.append(nn_ + diff / diff.norm(dim=-1, keepdim=True).clamp_min(1e-6) * eps)
                    rate = torch.stack(gains[-2:]).mean()  # aligned-direction steps
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        torch.distributed.all_reduce(rate)
                        rate = rate / torch.distributed.get_world_size()
                    self._critwm_rate = 0.7 * self._critwm_rate + 0.3 * float(rate)
                    import math as _math
                    self._critwm_gamma *= _math.exp(eta * (self._critwm_rate - target))
                    self._critwm_gamma = min(max(self._critwm_gamma, 0.02), 5.0)
        output['thermo_rate'] = torch.tensor(self._critwm_rate)
        output['thermo_gamma_loss'] = torch.tensor(self._critwm_gamma)
        output['pred_loss'] = loss_ss + self._critwm_gamma * loss_ms
        output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
        output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']
        losses_dict = {
            f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k or 'thermo' in k
        }
        self.log_dict(losses_dict, on_step=True, sync_dist=False)
        return output
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
    elif matched_one_step:
        pred_emb, tgt_emb = matched_one_step_prediction(
            self.model,
            emb,
            act_emb,
            ctx_len,
        )
    else:
        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]  # label
        pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # LeWM loss
    output['pred_loss'] = (pred_emb - tgt_emb).pow(2).mean()
    if loss_ss_mix is not None:
        output['pred_loss'] = loss_ss_mix + mix_gamma * output['pred_loss']
    output['sigreg_loss'] = self.sigreg(emb.transpose(0, 1))
    output['loss'] = output['pred_loss'] + lambd * output['sigreg_loss']

    if stage == 'fit':
        self._swm_last_loss_components = {
            name: output[name].detach()
            for name in ('pred_loss', 'sigreg_loss', 'loss')
        }

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    seed = int(cfg.seed)
    pl.seed_everything(seed, workers=True)
    print(
        f'[protocol] global_seed={seed} workers_seeded=true',
        flush=True,
    )

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
    transforms = []
    if not cfg.get('gpu_image_preprocess', False):
        transforms.append(
            get_img_preprocessor(
                source='pixels', target='pixels', img_size=cfg.img_size
            )
        )

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

    split_digest = hashlib.sha256()
    for subset in (train_set, val_set):
        indices = torch.as_tensor(subset.indices, dtype=torch.int64)
        split_digest.update(indices.view(torch.uint8).numpy().tobytes())
    generator_digest = hashlib.sha256(
        rnd_gen.get_state().numpy().tobytes()
    ).hexdigest()

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)
    print(
        f'[protocol] dataset_num_steps={dataset.num_steps} '
        f'dataset_samples={len(dataset)} train_samples={len(train_set)} '
        f'val_samples={len(val_set)} train_batches={len(train)} '
        f'split_sha256={split_digest.hexdigest()} '
        f'loader_state_sha256={generator_digest}',
        flush=True,
    )

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    init_weights_path = cfg.get('init_weights_path')
    if init_weights_path:
        init_path = Path(init_weights_path).expanduser().resolve()
        if not init_path.is_file():
            raise FileNotFoundError(
                f'init_weights_path does not exist: {init_path}'
            )
        state_dict = torch.load(
            init_path,
            map_location='cpu',
            weights_only=True,
        )
        world_model.load_state_dict(state_dict, strict=True)
        print(
            f'[protocol] loaded_initialization={init_path} '
            f'state_sha256={state_dict_sha256(world_model.state_dict())} '
            f'file_sha256={file_sha256(init_path)}',
            flush=True,
        )

    export_init_weights_path = cfg.get('export_init_weights_path')
    if export_init_weights_path:
        export_path = Path(export_init_weights_path)
        fingerprint, reused = export_initial_weights(world_model, export_path)
        print(
            f'[protocol] {"reused" if reused else "exported"}_initialization='
            f'{export_path.expanduser().resolve()} state_sha256={fingerprint} '
            f'file_sha256={file_sha256(export_path.expanduser().resolve())}',
            flush=True,
        )

    if bool(cfg.get('init_only', False)):
        if not export_init_weights_path:
            raise ValueError('init_only=true requires export_init_weights_path')
        print('[protocol] init_only complete', flush=True)
        return

    devices = cfg.trainer.get('devices', 1)
    if devices == 'auto':
        num_devices = torch.cuda.device_count()
    elif isinstance(devices, int):
        num_devices = devices
    else:
        num_devices = len(devices)
    world_size = max(1, num_devices * int(cfg.trainer.get('num_nodes', 1)))
    total_steps = cfg.trainer.max_epochs * max(1, len(train) // world_size)
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

    epoch_offset = int(cfg.get('epoch_offset', 0) or 0)
    if epoch_offset < 0:
        raise ValueError(f'epoch_offset must be non-negative, got {epoch_offset}')
    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg,
        epoch_interval=1,
        epoch_offset=epoch_offset,
    )
    critwm_state_callback = CritWMStateCallback()

    extra_callbacks = []
    trace_batches = int(cfg.get('pairing_trace_batches', 0) or 0)
    if trace_batches < 0:
        raise ValueError('pairing_trace_batches must be non-negative')
    if trace_batches:
        extra_callbacks.append(PairingTraceCallback(trace_batches))
    if os.environ.get('SWM_TRACE_DIVERGENCE'):
        extra_callbacks.append(DivergenceTraceCallback())

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_kwargs.setdefault('default_root_dir', str(run_dir))
    trainer = pl.Trainer(
        **trainer_kwargs,
        callbacks=[
            object_dump_callback,
            critwm_state_callback,
            *extra_callbacks,
            NonFiniteGradGuardCallback(
                policy=str(cfg.get('nonfinite_grad_policy', 'skip')),
                max_skip_frac=float(
                    cfg.get('nonfinite_max_skip_frac', 0.01)
                ),
                max_total_skips=(
                    int(cfg.nonfinite_max_total_skips)
                    if cfg.get('nonfinite_max_total_skips') is not None
                    else None
                ),
            ),
            NaNGuardCallback(),
        ],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    resume_ckpt_path = cfg.get('resume_ckpt_path')
    if resume_ckpt_path:
        ckpt_path = Path(resume_ckpt_path).expanduser().resolve()
        weights_only = bool(cfg.get('resume_weights_only', False))
    else:
        default_ckpt_path = (
            run_dir / f'{cfg.output_model_name}_weights.ckpt'
        )
        ckpt_path = default_ckpt_path if default_ckpt_path.exists() else None
        weights_only = True

    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        seed=seed,
        ckpt_path=ckpt_path,
        weights_only=weights_only,
    )

    manager()
    return


if __name__ == '__main__':
    run()
