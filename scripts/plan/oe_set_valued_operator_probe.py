"""Nested state-held-out probe for a set-valued optimizer operator.

The learned-vs-oracle CEM correction residual is strongly multi-modal.  A
single direct head averages opposite directions, while a small codebook has a
large oracle-routed ceiling.  This probe tests the missing architectural
object: a population-conditioned *set* of proposal updates.

A small attention module receives the complete unordered CEM population,
frozen-WM candidate scores, and a planner/latent or privileged-state context.
It predicts several residual update hypotheses plus a routing logit per
hypothesis.  The hypotheses are initialized from a correction codebook fit on
training states only.  Training uses a best-of-M physical update loss; no
held-out state participates in codebook fitting, normalization, checkpoint
selection, or model fitting.

Three outputs have different interpretations:

``top1``
    Deployable single-route update selected by the learned logits.

``top2 coverage``
    Oracle best among the two highest-logit hypotheses.  This is not yet a
    deployable planner; it asks whether carrying two branches preserves the
    right basin.

``all-mode coverage``
    Oracle best among every retained hypothesis.  This is the learned
    branch-set ceiling and should be compared with the static codebook ceiling.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from oe_candidate_cost_head_probe import load_arrays as load_candidate_arrays
from oe_update_corrector_probe import (
    EPS,
    paired_bootstrap,
    selection_score,
    state_aggregate,
    update_metrics,
)
from oe_update_mode_codebook_probe import spherical_kmeans


@dataclass(frozen=True)
class TrainConfig:
    hidden: int
    attention_heads: int
    correction_modes: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_populations: int
    router_weight: float
    router_kind: str
    router_temperature: float
    delta_anchor_weight: float


class SetValuedOperator(nn.Module):
    def __init__(
        self,
        *,
        candidate_width: int,
        context_width: int,
        update_width: int,
        dense_frames: int = 0,
        dense_width: int = 0,
        dense_grid: int = 0,
        config: TrainConfig,
        initial_modes: np.ndarray,
    ):
        super().__init__()
        hidden = config.hidden
        modes = config.correction_modes
        self.candidate = nn.Sequential(
            nn.Linear(candidate_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(context_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.mode_queries = nn.Parameter(
            torch.randn(modes, hidden) / hidden**0.5
        )
        self.attention = nn.MultiheadAttention(
            hidden,
            config.attention_heads,
            batch_first=True,
        )
        self.mode_norm = nn.LayerNorm(hidden)
        self.dense_frames = dense_frames
        if dense_frames:
            if dense_width < 1 or dense_grid < 1:
                raise ValueError('dense token dimensions must be positive')
            # Each spatial cell keeps the ordered history, goal, signed
            # goal-minus-current difference, and its magnitude together.  It
            # therefore exposes relative geometry without pooling away the
            # 16x16 patch layout.
            visual_width = (dense_frames + 2) * dense_width
            self.visual = nn.Sequential(
                nn.LayerNorm(visual_width),
                nn.Linear(visual_width, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            coordinates = torch.linspace(-1.0, 1.0, dense_grid)
            yy, xx = torch.meshgrid(
                coordinates,
                coordinates,
                indexing='ij',
            )
            self.register_buffer(
                'patch_coordinates',
                torch.stack([xx, yy], dim=-1).reshape(-1, 2),
                persistent=False,
            )
            self.coordinate = nn.Linear(2, hidden, bias=False)
            self.visual_attention = nn.MultiheadAttention(
                hidden,
                config.attention_heads,
                batch_first=True,
            )
            self.visual_norm = nn.LayerNorm(hidden)
        self.delta = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, update_width),
        )
        self.router = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.mode_bias = nn.Parameter(
            torch.as_tensor(initial_modes, dtype=torch.float32)
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

    def forward(
        self,
        candidates: torch.Tensor,
        context: torch.Tensor,
        dense_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.candidate(candidates)
        context_encoded = self.context(context)
        queries = (
            self.mode_queries[None]
            + context_encoded[:, None]
        )
        attended, _ = self.attention(
            queries,
            tokens,
            tokens,
            need_weights=False,
        )
        modes = self.mode_norm(attended + queries)
        if self.dense_frames:
            if dense_tokens is None:
                raise ValueError('dense tokens are required by this model')
            if dense_tokens.ndim != 4:
                raise ValueError(
                    'dense tokens must be (B,F,P,D), got '
                    f'{tuple(dense_tokens.shape)}'
                )
            current = dense_tokens[:, -2]
            goal = dense_tokens[:, -1]
            visual_input = torch.cat(
                [
                    dense_tokens.transpose(1, 2).flatten(2, 3),
                    (goal - current),
                    torch.abs(goal - current),
                ],
                dim=-1,
            )
            visual_tokens = (
                self.visual(visual_input)
                + self.coordinate(self.patch_coordinates)[None]
            )
            visual_attended, _ = self.visual_attention(
                modes,
                visual_tokens,
                visual_tokens,
                need_weights=False,
            )
            modes = self.visual_norm(modes + visual_attended)
        expanded_context = context_encoded[:, None].expand_as(modes)
        combined = torch.cat([modes, expanded_context], dim=-1)
        contextual_delta = self.delta(combined)
        residual = self.mode_bias[None] + contextual_delta
        routing_logits = self.router(combined)[..., 0]
        return residual, routing_logits, contextual_delta


def population_indices(arrays: dict, states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(arrays['state_ids'], states))


def load_dense_cache(path: Path, *, source_rows: np.ndarray) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {'rows', 'state_tokens', 'goal_tokens'}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{path} is missing dense fields {missing}')
        rows = np.asarray(archive['rows'], dtype=np.int64)
        state = np.asarray(archive['state_tokens'])
        goal = np.asarray(archive['goal_tokens'])
        audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
    if not np.array_equal(rows, source_rows):
        raise ValueError(
            f'dense-cache/source row mismatch for {path}: '
            f'{np.sum(rows != source_rows)} differing rows'
        )
    if (
        state.ndim != 4
        or goal.ndim != 4
        or len(state) != len(rows)
        or len(goal) != len(rows)
        or state.shape[2:] != goal.shape[2:]
    ):
        raise ValueError(
            'dense state/goal tokens must be aligned (S,T,P,D), got '
            f'{state.shape} and {goal.shape}'
        )
    if goal.shape[1] != 1:
        raise ValueError(f'expected one goal frame, got {goal.shape}')
    tokens = np.concatenate([state, goal], axis=1)
    patch_count = int(tokens.shape[2])
    grid = int(round(patch_count**0.5))
    if grid * grid != patch_count:
        raise ValueError(f'dense patch count is not square: {patch_count}')
    return {
        'tokens': tokens,
        'grid': grid,
        'audit': audit,
    }


def load_outcome_cache(
    path: Path,
    *,
    source_rows: np.ndarray,
    states: int,
    rounds: int,
    candidates: int,
    cost_only: bool = False,
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive['rows'], dtype=np.int64)
        parent_outcome = (
            Path(str(np.asarray(archive['parent_outcome']).item()))
            if 'parent_outcome' in archive.files
            else None
        )
        feature_mask = (
            np.asarray(archive['feature_mask'], dtype=bool)
            if 'feature_mask' in archive.files
            else None
        )
        audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
        if parent_outcome is None:
            required = {'terminal_embeddings', 'goal_embeddings'}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(
                    f'{path} is missing outcome fields {missing}'
                )
            terminal = np.asarray(
                archive['terminal_embeddings'],
                dtype=np.float32,
            )
            goal = np.asarray(
                archive['goal_embeddings'],
                dtype=np.float32,
            )
            recomputed_cost = (
                np.asarray(
                    archive['recomputed_cost'],
                    dtype=np.float32,
                )
                if 'recomputed_cost' in archive.files
                else None
            )
    if parent_outcome is not None:
        if not parent_outcome.is_file():
            raise FileNotFoundError(
                f'masked outcome parent does not exist: {parent_outcome}'
            )
        with np.load(parent_outcome, allow_pickle=False) as parent:
            required = {
                'rows',
                'terminal_embeddings',
                'goal_embeddings',
            }
            missing = sorted(required - set(parent.files))
            if missing:
                raise ValueError(
                    f'{parent_outcome} is missing outcome fields {missing}'
                )
            parent_rows = np.asarray(parent['rows'], dtype=np.int64)
            terminal = np.asarray(
                parent['terminal_embeddings'],
                dtype=np.float32,
            )
            goal = np.asarray(
                parent['goal_embeddings'],
                dtype=np.float32,
            )
            recomputed_cost = (
                np.asarray(
                    parent['recomputed_cost'],
                    dtype=np.float32,
                )
                if 'recomputed_cost' in parent.files
                else None
            )
        if not np.array_equal(rows, parent_rows):
            raise ValueError(
                f'masked outcome rows do not match parent {parent_outcome}'
            )
    if not np.array_equal(rows, source_rows):
        raise ValueError(
            f'outcome-cache/source row mismatch for {path}: '
            f'{np.sum(rows != source_rows)} differing rows'
        )
    expected = (states, rounds, candidates)
    if terminal.ndim != 4 or terminal.shape[:3] != expected:
        raise ValueError(
            f'expected terminal embeddings {expected}xD, got '
            f'{terminal.shape}'
        )
    if goal.ndim != 2 or goal.shape != (states, terminal.shape[-1]):
        raise ValueError(
            f'goal embeddings do not align with terminal embeddings: '
            f'{goal.shape} vs {terminal.shape}'
        )
    # LeWM's scalar score is ||terminal-goal||^2.  Preserve the signed vector
    # that the scalar scorer discards; no simulator value enters this cache.
    relative = terminal - goal[:, None, None]
    if feature_mask is not None and feature_mask.shape != expected:
        raise ValueError(
            f'outcome feature mask must be {expected}, '
            f'got {feature_mask.shape}'
        )
    flat_mask = (
        feature_mask.reshape(states * rounds, candidates)
        if feature_mask is not None
        else None
    )
    feature_parts = []
    if not cost_only:
        relative_features = relative.reshape(
            states * rounds,
            candidates,
            terminal.shape[-1],
        )
        if flat_mask is not None:
            relative_features = (
                relative_features * flat_mask[..., None]
            )
        feature_parts.append(relative_features)
    if recomputed_cost is not None:
        if recomputed_cost.shape != (states, rounds, candidates):
            raise ValueError(
                'recomputed outcome cost does not align: '
                f'{recomputed_cost.shape}'
            )
        cost = recomputed_cost.reshape(states * rounds, candidates)
        if flat_mask is None:
            median = np.median(cost, axis=1, keepdims=True)
            q25 = np.quantile(cost, 0.25, axis=1, keepdims=True)
            q75 = np.quantile(cost, 0.75, axis=1, keepdims=True)
            robust = (cost - median) / np.maximum(
                (q75 - q25) / 1.349,
                1e-5,
            )
            order = np.argsort(cost, axis=1, kind='stable')
            rank = np.empty_like(order, dtype=np.float32)
            rank[
                np.arange(len(cost))[:, None],
                order,
            ] = np.arange(candidates, dtype=np.float32)[None]
            rank /= max(candidates - 1, 1)
        else:
            query_counts = flat_mask.sum(axis=1)
            if not np.all(query_counts == query_counts[0]):
                raise ValueError(
                    'every masked outcome population must use the '
                    'same query count'
                )
            query_count = int(query_counts[0])
            selected_cost = cost[flat_mask].reshape(
                len(cost),
                query_count,
            )
            median = np.median(
                selected_cost,
                axis=1,
                keepdims=True,
            )
            q25 = np.quantile(
                selected_cost,
                0.25,
                axis=1,
                keepdims=True,
            )
            q75 = np.quantile(
                selected_cost,
                0.75,
                axis=1,
                keepdims=True,
            )
            selected_robust = (
                (selected_cost - median)
                / np.maximum((q75 - q25) / 1.349, 1e-5)
            )
            selected_order = np.argsort(
                selected_cost,
                axis=1,
                kind='stable',
            )
            selected_rank = np.empty_like(
                selected_order,
                dtype=np.float32,
            )
            selected_rank[
                np.arange(len(cost))[:, None],
                selected_order,
            ] = np.arange(query_count, dtype=np.float32)[None]
            selected_rank /= max(query_count - 1, 1)
            robust = np.zeros_like(cost, dtype=np.float32)
            rank = np.zeros_like(cost, dtype=np.float32)
            robust[flat_mask] = selected_robust.reshape(-1)
            rank[flat_mask] = selected_rank.reshape(-1)
        scalar_features = [
            robust[..., None].astype(np.float32),
            rank[..., None],
        ]
        if flat_mask is not None:
            scalar_features = [
                feature * flat_mask[..., None]
                for feature in scalar_features
            ]
        feature_parts.extend(scalar_features)
    if flat_mask is not None:
        feature_parts.append(flat_mask[..., None].astype(np.float32))
    if not feature_parts:
        raise ValueError(f'{path} has no candidate outcome features')
    return {
        'features': np.concatenate(feature_parts, axis=-1),
        'audit': audit,
    }


def load_innovation_cache(
    path: Path,
    *,
    source_rows: np.ndarray,
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {'rows', 'features'}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{path} is missing innovation fields {missing}')
        rows = np.asarray(archive['rows'], dtype=np.int64)
        features = np.asarray(archive['features'], dtype=np.float32)
        audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
    if not np.array_equal(rows, source_rows):
        raise ValueError(
            f'innovation-cache/source row mismatch for {path}: '
            f'{np.sum(rows != source_rows)} differing rows'
        )
    if features.ndim != 2 or len(features) != len(rows):
        raise ValueError(
            f'innovation features must be (states,D), got {features.shape}'
        )
    return {'features': features, 'audit': audit}


def fit_normalization(
    arrays: dict,
    trace,
    indices: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates = arrays['candidate_features'][indices]
    context = arrays['context_features'][indices]
    residual = (
        trace.oracle_update_normalized[indices]
        - trace.model_update_normalized[indices]
    )

    def moments(
        values: np.ndarray,
        *,
        axes,
    ) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=axes)
        scale = values.std(axis=axes)
        return (
            mean.astype(np.float32),
            np.where(scale > 1e-5, scale, 1.0).astype(np.float32),
        )

    candidate_mean, candidate_scale = moments(
        candidates,
        axes=(0, 1),
    )
    context_mean, context_scale = moments(context, axes=0)
    residual_mean, residual_scale = moments(residual, axes=0)
    return {
        'candidate_mean': candidate_mean,
        'candidate_scale': candidate_scale,
        'context_mean': context_mean,
        'context_scale': context_scale,
        'residual_mean': residual_mean,
        'residual_scale': residual_scale,
    }


def initial_mode_array(
    trace,
    indices: np.ndarray,
    *,
    correction_modes: int,
    normalization: dict,
    seed: int,
) -> np.ndarray:
    residual = (
        trace.oracle_update_normalized[indices]
        - trace.model_update_normalized[indices]
    )
    if correction_modes == 1:
        modes = np.zeros((1, residual.shape[1]), dtype=np.float64)
    else:
        prototypes, _ = spherical_kmeans(
            residual,
            clusters=correction_modes - 1,
            seed=seed,
        )
        modes = np.concatenate(
            [np.zeros((1, residual.shape[1])), prototypes],
            axis=0,
        )
    return (
        (modes - normalization['residual_mean'])
        / normalization['residual_scale']
    ).astype(np.float32)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(
    arrays: dict,
    trace,
    config: TrainConfig,
    *,
    initial_modes: np.ndarray,
) -> SetValuedOperator:
    return SetValuedOperator(
        candidate_width=arrays['candidate_features'].shape[-1],
        context_width=arrays['context_features'].shape[-1],
        update_width=trace.model_update_normalized.shape[-1],
        dense_frames=(
            arrays['dense_tokens'].shape[1]
            if 'dense_tokens' in arrays
            else 0
        ),
        dense_width=(
            arrays['dense_tokens'].shape[-1]
            if 'dense_tokens' in arrays
            else 0
        ),
        dense_grid=arrays.get('dense_grid', 0),
        config=config,
        initial_modes=initial_modes,
    )


def batch_tensors(
    arrays: dict,
    trace,
    indices: np.ndarray,
    *,
    normalization: dict,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    candidates = (
        arrays['candidate_features'][indices]
        - normalization['candidate_mean']
    ) / normalization['candidate_scale']
    context = (
        arrays['context_features'][indices]
        - normalization['context_mean']
    ) / normalization['context_scale']
    residual = (
        trace.oracle_update_normalized[indices]
        - trace.model_update_normalized[indices]
    )
    target = (
        residual - normalization['residual_mean']
    ) / normalization['residual_scale']
    dense = (
        torch.as_tensor(
            arrays['dense_tokens'][arrays['state_ids'][indices]],
            device=device,
            dtype=torch.float32,
        )
        if 'dense_tokens' in arrays
        else None
    )
    return (
        torch.as_tensor(candidates, device=device),
        torch.as_tensor(context, device=device),
        torch.as_tensor(target, device=device),
        torch.as_tensor(
            trace.model_update_normalized[indices],
            device=device,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            trace.oracle_update_normalized[indices],
            device=device,
            dtype=torch.float32,
        ),
        torch.as_tensor(
            trace.proposal_std[indices],
            device=device,
            dtype=torch.float32,
        ),
        dense,
    )


def decode_residual(
    normalized: torch.Tensor,
    normalization: dict,
) -> torch.Tensor:
    mean = torch.as_tensor(
        normalization['residual_mean'],
        device=normalized.device,
    )
    scale = torch.as_tensor(
        normalization['residual_scale'],
        device=normalized.device,
    )
    return normalized * scale + mean


def physical_relative_error(
    residual: torch.Tensor,
    model_update: torch.Tensor,
    oracle_update: torch.Tensor,
    proposal_std: torch.Tensor,
) -> torch.Tensor:
    corrected = model_update[:, None] + residual
    difference = (
        corrected - oracle_update[:, None]
    ) * proposal_std[:, None]
    denominator = torch.linalg.vector_norm(
        oracle_update * proposal_std,
        dim=-1,
    ).clamp_min(EPS)
    return (
        difference.square().sum(dim=-1)
        / denominator[:, None].square()
    )


def train_checkpoints(
    arrays: dict,
    trace,
    states: np.ndarray,
    *,
    config: TrainConfig,
    checkpoints: list[int],
    seed: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict, np.ndarray]:
    seed_everything(seed)
    indices = population_indices(arrays, states)
    normalization = fit_normalization(arrays, trace, indices)
    initial_modes = initial_mode_array(
        trace,
        indices,
        correction_modes=config.correction_modes,
        normalization=normalization,
        seed=seed,
    )
    model = make_model(
        arrays,
        trace,
        config,
        initial_modes=initial_modes,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    saved = {
        0: {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
    }
    checkpoint_set = set(checkpoints)
    rng = np.random.default_rng(seed)
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        shuffled = rng.permutation(indices)
        for offset in range(
            0,
            len(shuffled),
            config.batch_populations,
        ):
            batch = shuffled[offset : offset + config.batch_populations]
            tensors = batch_tensors(
                arrays,
                trace,
                batch,
                normalization=normalization,
                device=device,
            )
            (
                candidates,
                context,
                _,
                model_update,
                oracle_update,
                proposal_std,
                dense,
            ) = tensors
            predicted_normalized, routing_logits, delta = model(
                candidates,
                context,
                dense,
            )
            predicted = decode_residual(
                predicted_normalized,
                normalization,
            )
            errors = physical_relative_error(
                predicted,
                model_update,
                oracle_update,
                proposal_std,
            )
            winner = torch.argmin(errors.detach(), dim=1)
            best_of_m = errors.gather(
                1,
                winner[:, None],
            ).mean()
            if config.correction_modes == 1:
                router = best_of_m.new_zeros(())
            elif config.router_kind == 'winner_ce':
                router = F.cross_entropy(routing_logits, winner)
            elif config.router_kind == 'softmin':
                target_probability = F.softmax(
                    -errors.detach() / config.router_temperature,
                    dim=1,
                )
                router = -(
                    target_probability
                    * F.log_softmax(routing_logits, dim=1)
                ).sum(dim=1).mean()
            elif config.router_kind == 'expected_regret':
                regret = errors.detach() - errors.detach().min(
                    dim=1,
                    keepdim=True,
                ).values
                regret = regret / regret.mean(
                    dim=1,
                    keepdim=True,
                ).clamp_min(EPS)
                router = (
                    F.softmax(routing_logits, dim=1) * regret
                ).sum(dim=1).mean()
            else:
                raise ValueError(
                    f'unknown router kind {config.router_kind!r}'
                )
            loss = (
                best_of_m
                + config.router_weight * router
                + config.delta_anchor_weight * delta.square().mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        if epoch in checkpoint_set:
            saved[epoch] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    del model
    torch.cuda.empty_cache()
    return saved, normalization, initial_modes


@torch.inference_mode()
def predict(
    model: SetValuedOperator,
    arrays: dict,
    trace,
    indices: np.ndarray,
    *,
    normalization: dict,
    config: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    residual_rows = []
    logit_rows = []
    model.eval()
    for offset in range(
        0,
        len(indices),
        config.batch_populations,
    ):
        batch = indices[offset : offset + config.batch_populations]
        tensors = batch_tensors(
            arrays,
            trace,
            batch,
            normalization=normalization,
            device=device,
        )
        residual, logits, _ = model(tensors[0], tensors[1], tensors[6])
        residual = decode_residual(residual, normalization)
        residual_rows.append(residual.float().cpu().numpy())
        logit_rows.append(logits.float().cpu().numpy())
    return np.concatenate(residual_rows), np.concatenate(logit_rows)


def gather_mode(
    values: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    return values[np.arange(len(values)), labels]


def best_labels(
    trace,
    indices: np.ndarray,
    residuals: np.ndarray,
    allowed: np.ndarray | None = None,
) -> np.ndarray:
    corrected = (
        trace.model_update_normalized[indices, None] + residuals
    )
    physical_difference = (
        corrected - trace.oracle_update_normalized[indices, None]
    ) * trace.proposal_std[indices, None]
    denominator = np.linalg.norm(
        trace.oracle_update_normalized[indices]
        * trace.proposal_std[indices],
        axis=1,
    )
    relative = np.linalg.norm(
        physical_difference,
        axis=2,
    ) / np.maximum(denominator[:, None], EPS)
    if allowed is None:
        return np.argmin(relative, axis=1)
    selected = np.take_along_axis(relative, allowed, axis=1)
    positions = np.argmin(selected, axis=1)
    return allowed[np.arange(len(allowed)), positions]


def state_quality(
    predicted: np.ndarray,
    trace,
    indices: np.ndarray,
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    cosine, relative = update_metrics(
        predicted,
        trace.oracle_update_normalized[indices],
        trace.proposal_std[indices],
    )
    example_states = trace.state_ids[indices]
    return {
        'update_cosine': state_aggregate(
            cosine,
            example_states,
            states,
        ),
        'relative_update_error': state_aggregate(
            relative,
            example_states,
            states,
        ),
    }


def inner_states(
    outer_train_states: np.ndarray,
    *,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(sorted(int(state) for state in outer_train_states))
    selector = np.arange(len(ordered)) % 5 == fold % 5
    return ordered[~selector], ordered[selector]


def select_checkpoint(
    arrays: dict,
    trace,
    outer_train_states: np.ndarray,
    *,
    config: TrainConfig,
    seed: int,
    fold: int,
    device: torch.device,
) -> tuple[int, float, list[dict]]:
    training_states, validation_states = inner_states(
        outer_train_states,
        fold=fold,
    )
    checkpoints = sorted(
        {
            0,
            1,
            min(2, config.max_epochs),
            min(5, config.max_epochs),
            min(10, config.max_epochs),
            min(20, config.max_epochs),
            config.max_epochs,
        }
    )
    saved, normalization, initial_modes = train_checkpoints(
        arrays,
        trace,
        training_states,
        config=config,
        checkpoints=checkpoints,
        seed=seed,
        device=device,
    )
    indices = population_indices(arrays, validation_states)
    rows = []
    for epoch in checkpoints:
        model = make_model(
            arrays,
            trace,
            config,
            initial_modes=initial_modes,
        ).to(device)
        model.load_state_dict(saved[epoch])
        residuals, logits = predict(
            model,
            arrays,
            trace,
            indices,
            normalization=normalization,
            config=config,
            device=device,
        )
        route = np.argmax(logits, axis=1)
        selected = gather_mode(residuals, route)
        for blend in (0.25, 0.5, 1.0):
            corrected = (
                trace.model_update_normalized[indices]
                + blend * selected
            )
            rows.append(
                {
                    'epoch': epoch,
                    'blend': blend,
                    'score': selection_score(
                        corrected,
                        trace,
                        indices,
                        validation_states,
                    ),
                }
            )
        del model
    best = max(
        rows,
        key=lambda row: (
            row['score'],
            -row['epoch'],
            -row['blend'],
        ),
    )
    del saved
    torch.cuda.empty_cache()
    return int(best['epoch']), float(best['blend']), rows


def fit_outer(
    arrays: dict,
    trace,
    training_states: np.ndarray,
    *,
    config: TrainConfig,
    epoch: int,
    seed: int,
    device: torch.device,
) -> tuple[SetValuedOperator, dict]:
    checkpoints, normalization, initial_modes = train_checkpoints(
        arrays,
        trace,
        training_states,
        config=config,
        checkpoints=[epoch],
        seed=seed,
        device=device,
    )
    model = make_model(
        arrays,
        trace,
        config,
        initial_modes=initial_modes,
    ).to(device)
    model.load_state_dict(checkpoints[epoch])
    return model.eval(), normalization


def metric_row(
    baseline: np.ndarray,
    corrected: np.ndarray,
    *,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict:
    delta = corrected - baseline
    return {
        'baseline': float(np.mean(baseline)),
        'corrected': float(np.mean(corrected)),
        'delta': float(np.mean(delta)),
        'delta_ci': list(
            paired_bootstrap(delta, samples=bootstrap, rng=rng)
        ),
    }


def analyze(
    arrays: dict,
    trace,
    *,
    config: TrainConfig,
    bootstrap: int,
    seed: int,
    device: torch.device,
) -> dict:
    states = np.arange(arrays['num_states'])
    population_count = len(trace.state_ids)
    modes = config.correction_modes
    residuals_all = np.empty(
        (
            population_count,
            modes,
            trace.model_update_normalized.shape[1],
        ),
        dtype=np.float64,
    )
    logits_all = np.empty((population_count, modes), dtype=np.float64)
    blends = np.empty(population_count, dtype=np.float64)
    folds = []
    for fold in range(3):
        validation_states = states[states % 3 == fold]
        training_states = states[states % 3 != fold]
        epoch, blend, selection_rows = select_checkpoint(
            arrays,
            trace,
            training_states,
            config=config,
            seed=seed + 10_000 * fold,
            fold=fold,
            device=device,
        )
        model, normalization = fit_outer(
            arrays,
            trace,
            training_states,
            config=config,
            epoch=epoch,
            seed=seed + 100_000 + 10_000 * fold,
            device=device,
        )
        indices = population_indices(arrays, validation_states)
        residuals, logits = predict(
            model,
            arrays,
            trace,
            indices,
            normalization=normalization,
            config=config,
            device=device,
        )
        residuals_all[indices] = residuals
        logits_all[indices] = logits
        blends[indices] = blend
        folds.append(
            {
                'fold': fold,
                'training_states': training_states.tolist(),
                'validation_states': validation_states.tolist(),
                'selected_epoch': epoch,
                'selected_blend': blend,
                'selection_rows': selection_rows,
            }
        )
        del model
        torch.cuda.empty_cache()

    indices = np.arange(population_count)
    baseline_update = trace.model_update_normalized
    top1_labels = np.argmax(logits_all, axis=1)
    top1_residual = gather_mode(residuals_all, top1_labels)
    top1_update = baseline_update + blends[:, None] * top1_residual

    top_count = min(2, modes)
    top2_allowed = np.argsort(
        -logits_all,
        axis=1,
        kind='stable',
    )[:, :top_count]
    top2_labels = best_labels(
        trace,
        indices,
        blends[:, None, None] * residuals_all,
        allowed=top2_allowed,
    )
    top2_update = baseline_update + blends[:, None] * gather_mode(
        residuals_all,
        top2_labels,
    )
    all_labels = best_labels(
        trace,
        indices,
        blends[:, None, None] * residuals_all,
    )
    all_update = baseline_update + blends[:, None] * gather_mode(
        residuals_all,
        all_labels,
    )

    baseline = state_quality(baseline_update, trace, indices, states)
    top1 = state_quality(top1_update, trace, indices, states)
    top2 = state_quality(top2_update, trace, indices, states)
    all_modes = state_quality(all_update, trace, indices, states)
    rng = np.random.default_rng(seed)
    metrics = {}
    for name, values in (
        ('top1', top1),
        ('top2_coverage', top2),
        ('all_mode_coverage', all_modes),
    ):
        metrics[name] = {
            key: metric_row(
                baseline[key],
                values[key],
                bootstrap=bootstrap,
                rng=rng,
            )
            for key in baseline
        }
    best_all = best_labels(
        trace,
        indices,
        blends[:, None, None] * residuals_all,
    )
    return {
        'metrics': metrics,
        'folds': folds,
        'routing': {
            'top1_matches_best_mode': float(
                np.mean(top1_labels == best_all)
            ),
            'top2_contains_best_mode': float(
                np.mean(
                    np.any(
                        top2_allowed == best_all[:, None],
                        axis=1,
                    )
                )
            ),
            'top1_mode_counts': np.bincount(
                top1_labels,
                minlength=modes,
            ).tolist(),
            'oracle_best_mode_counts': np.bincount(
                best_all,
                minlength=modes,
            ).tolist(),
        },
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--family', required=True)
    parser.add_argument('--latent-cache', type=Path)
    parser.add_argument('--dense-cache', type=Path)
    parser.add_argument('--outcome-cache', type=Path)
    parser.add_argument(
        '--extra-outcome-cache',
        action='append',
        type=Path,
        default=[],
    )
    parser.add_argument('--outcome-cost-only', action='store_true')
    parser.add_argument('--innovation-cache', type=Path)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--modes', type=int, default=5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--attention-heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--batch-populations', type=int, default=16)
    parser.add_argument('--router-weight', type=float, default=0.1)
    parser.add_argument(
        '--router-kind',
        choices=('winner_ce', 'softmin', 'expected_regret'),
        default='winner_ce',
    )
    parser.add_argument('--router-temperature', type=float, default=0.1)
    parser.add_argument('--delta-anchor-weight', type=float, default=1e-3)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260880)
    args = parser.parse_args()
    if args.modes < 1 or args.hidden < 1 or args.max_epochs < 1:
        raise ValueError('modes, hidden, and max-epochs must be positive')
    if args.hidden % args.attention_heads:
        raise ValueError('hidden must be divisible by attention-heads')
    if args.router_temperature <= 0:
        raise ValueError('router-temperature must be positive')
    if args.family in (
        'planner_latent',
        'planner_history_latent',
    ) and args.latent_cache is None:
        raise ValueError(f'{args.family} requires --latent-cache')
    if 'dense' in args.family and args.dense_cache is None:
        raise ValueError(f'{args.family} requires --dense-cache')
    if 'outcome' in args.family and args.outcome_cache is None:
        raise ValueError(f'{args.family} requires --outcome-cache')
    if 'innovation' in args.family and args.innovation_cache is None:
        raise ValueError(f'{args.family} requires --innovation-cache')

    config = TrainConfig(
        hidden=args.hidden,
        attention_heads=args.attention_heads,
        correction_modes=args.modes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_populations=args.batch_populations,
        router_weight=args.router_weight,
        router_kind=args.router_kind,
        router_temperature=args.router_temperature,
        delta_anchor_weight=args.delta_anchor_weight,
    )
    probe_families = {
        'planner_dense',
        'planner_outcome',
        'planner_dense_outcome',
        'planner_innovation',
        'planner_innovation_outcome',
        'planner_dense_innovation_outcome',
    }
    base_family = 'planner' if args.family in probe_families else args.family
    arrays = load_candidate_arrays(
        args.source,
        topk=args.topk,
        family=base_family,
        latent_cache=args.latent_cache,
    )
    dense_audit = None
    if args.dense_cache is not None:
        dense = load_dense_cache(
            args.dense_cache,
            source_rows=arrays['rows'],
        )
        arrays['dense_tokens'] = dense['tokens']
        arrays['dense_grid'] = dense['grid']
        dense_audit = dense['audit']
    outcome_audit = None
    extra_outcome_audits = []
    if args.outcome_cache is not None:
        outcome = load_outcome_cache(
            args.outcome_cache,
            source_rows=arrays['rows'],
            states=arrays['num_states'],
            rounds=arrays['num_rounds'],
            candidates=arrays['num_candidates'],
            cost_only=args.outcome_cost_only,
        )
        arrays['candidate_features'] = np.concatenate(
            [arrays['candidate_features'], outcome['features']],
            axis=-1,
        )
        outcome_audit = outcome['audit']
    for extra_path in args.extra_outcome_cache:
        extra = load_outcome_cache(
            extra_path,
            source_rows=arrays['rows'],
            states=arrays['num_states'],
            rounds=arrays['num_rounds'],
            candidates=arrays['num_candidates'],
            cost_only=args.outcome_cost_only,
        )
        arrays['candidate_features'] = np.concatenate(
            [arrays['candidate_features'], extra['features']],
            axis=-1,
        )
        extra_outcome_audits.append(
            {
                'path': str(extra_path.resolve()),
                'audit': extra['audit'],
            }
        )
    innovation_audit = None
    if args.innovation_cache is not None:
        innovation = load_innovation_cache(
            args.innovation_cache,
            source_rows=arrays['rows'],
        )
        arrays['context_features'] = np.concatenate(
            [
                arrays['context_features'],
                innovation['features'][arrays['state_ids']],
            ],
            axis=-1,
        )
        innovation_audit = innovation['audit']
    trace = arrays['trace']
    analysis = analyze(
        arrays,
        trace,
        config=config,
        bootstrap=args.bootstrap,
        seed=args.seed,
        device=torch.device('cuda'),
    )
    payload = {
        'version': 1,
        'source': str(args.source.resolve()),
        'rows': arrays['rows'].tolist(),
        'cell': trace.label,
        'family': args.family,
        'config': asdict(config),
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'dense_cache': (
            str(args.dense_cache.resolve())
            if args.dense_cache is not None
            else None
        ),
        'dense_audit': dense_audit,
        'outcome_cache': (
            str(args.outcome_cache.resolve())
            if args.outcome_cache is not None
            else None
        ),
        'outcome_audit': outcome_audit,
        'extra_outcome_audits': extra_outcome_audits,
        'outcome_cost_only': args.outcome_cost_only,
        'innovation_cache': (
            str(args.innovation_cache.resolve())
            if args.innovation_cache is not None
            else None
        ),
        'innovation_audit': innovation_audit,
        **analysis,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    lines = [
        '# Set-valued optimizer operator probe',
        '',
        'Top1 is a learned single route. Top2/all-mode rows use oracle '
        'selection only to measure retained branch coverage.',
        '',
        f'- Cell: `{trace.label}`',
        f'- Features: `{args.family}`',
        f'- Retained modes including no-op: `{args.modes}`',
        '',
        '| output | metric | baseline | corrected | delta | paired 95% CI |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for output in ('top1', 'top2_coverage', 'all_mode_coverage'):
        for metric in ('update_cosine', 'relative_update_error'):
            row = analysis['metrics'][output][metric]
            lines.append(
                f'| {output} | {metric} '
                f'| {row["baseline"]:.3f} '
                f'| {row["corrected"]:.3f} '
                f'| {row["delta"]:+.3f} '
                f'| {format_ci(row["delta_ci"])} |'
            )
    lines.extend(
        [
            '',
            f'- Top1 exact best-mode rate: '
            f'`{analysis["routing"]["top1_matches_best_mode"]:.3f}`',
            f'- Top2 contains best-mode rate: '
            f'`{analysis["routing"]["top2_contains_best_mode"]:.3f}`',
        ]
    )
    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print('\n'.join(lines), flush=True)
    print(f'results -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
