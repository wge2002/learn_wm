import contextlib

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class LeWM(nn.Module):
    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        encoder_fp32: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # Opt-in only: the default reproduces the historical bf16-everywhere
        # numerics bit for bit, so no existing run changes behavior.
        self.encoder_fp32 = bool(encoder_fp32)

    def _encoder_precision(self, device):
        """Optionally run the ViT encoder as an FP32 precision island.

        Under ``precision: bf16`` the ViT *backward* is where the v2 K1 runs
        lost numerical validity, and the damage is confined to the encoder.
        Replaying the two exact failing steps (identical batch, RNG and
        BatchNorm buffers) gives:

        =====  ==================  =========================  ==========
        seed   bf16 raw grad norm  encoder FP32 island        full FP32
        =====  ==================  =========================  ==========
        42     13982356            220.28                     242.60
        13     1907000.9           14.916                     --
        =====  ==================  =========================  ==========

        The forward loss is finite and nearly identical in every variant, so
        the defect is bf16 accumulation in the encoder backward rather than a
        diverged model or a bad batch. Keeping the island around the encoder
        alone leaves the projector, predictor and action encoder in bf16,
        which the same replay shows to be well conditioned.
        """
        if not self.encoder_fp32:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, enabled=False)

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(
            pixels, 'b t ... -> (b t) ...'
        )  # flatten for encoding
        with self._encoder_precision(pixels.device):
            output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info['emb'] = rearrange(emb, '(b t) d -> b t d', b=b)

        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, 'b t d -> (b t) d'))
        preds = rearrange(preds, '(b t) d -> b t d', b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        if history_size is None:
            history_size = getattr(self.predictor, 'num_frames', 3)

        assert 'pixels' in info, 'pixels not in info_dict'
        H = info['pixels'].size(2)
        # matched-history planning: candidates only contain FUTURE actions; the
        # executed past actions aligned with the H-1 history frames are supplied
        # separately so they are not confused with plan candidates.
        if 'past_action' in info:
            past = info['past_action']
            assert past.size(2) == H - 1, (
                f'past_action must cover H-1={H - 1} frames, got {past.size(2)}'
            )
            action_sequence = torch.cat(
                [
                    past.to(
                        dtype=action_sequence.dtype,
                        device=action_sequence.device,
                    ),
                    action_sequence,
                ],
                dim=2,
            )
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info['action'] = act_0
        n_steps = T - H

        # encode initial state, or reuse cached embedding from a prior rollout.
        # detach: to avoid backprop in encoder
        if 'emb' not in info:
            _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
            _init = self.encode(_init)
            info['emb'] = (
                _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)
            )

        # flatten batch and sample dimensions for rollout
        emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
        act_flat = rearrange(act_0, 'b s ... -> (b s) ...')
        act_future_flat = rearrange(act_future, 'b s ... -> (b s) ...')
        all_act_emb = self.action_encoder(
            torch.cat([act_flat, act_future_flat], dim=1)
        )  # (BS, T, A_emb)

        # rollout predictor autoregressively for n_steps + 1 (final) steps
        # emb_list holds individual (BS, D) frames, each with its own grad_fn
        HS = history_size
        emb_list = list(emb_init.unbind(dim=1))  # H tensors of shape (BS, D)
        for t in range(n_steps + 1):
            lo = max(0, H + t - HS)
            emb_trunc = torch.stack(emb_list[lo:], dim=1)  # (BS, HS, D)
            act_trunc = all_act_emb[:, lo : H + t]  # (BS, HS, A_emb)
            emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])

        emb = torch.stack(emb_list, dim=1)  # (BS, H + n_steps + 1, D)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
        info['predicted_emb'] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict['predicted_emb']  # (B,S, T-1, dim)
        goal_emb = info_dict['goal_emb']  # (B, T, dim)

        goal_emb = goal_emb[:, None, -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction='none',
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """Compute the cost of action candidates given an info dict with goal and initial state."""

        assert 'goal' in info_dict, 'goal not in info_dict'

        # encode goal state, or reuse cached embedding from a prior call
        if 'goal_emb' not in info_dict:
            goal = {
                k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)
            }
            goal['pixels'] = goal['goal']

            for k in info_dict:
                if k.startswith('goal_'):
                    goal[k[len('goal_') :]] = goal.pop(k)

            goal.pop('action')
            goal = self.encode(goal)

            info_dict['goal_emb'] = goal['emb']

        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)

        return cost


__all__ = ['LeWM']
