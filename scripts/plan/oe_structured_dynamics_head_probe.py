"""State-held-out probe for a structured counterfactual dynamics sidecar.

The scalar candidate-cost head can only learn a direct ranking correction.
This probe asks a different question: does the frozen LeWM context retain
enough information for a small action-conditioned model to predict explicit
terminal task geometry for counterfactual CEM candidates?

Two supervision interfaces are compared:

``terminal``
    Predict terminal PushT pose and the goal pose, then compute the evaluator
    cost analytically.  This is a small structured dynamics world model.

``relative``
    Predict terminal-minus-goal pose directly.  This is a denser but more
    task-specific geometry readout and serves as an easier upper rung.

The input context is either the frozen LeWM state/history/goal cache or the
privileged physical initial/goal state.  Privileged context is diagnostic
only.  Every metric comes from an outer state-held-out prediction.  Training
epoch and fusion with the frozen LeWM cost are selected on an inner
state-held-out split.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from oe_candidate_cost_head_probe import (
    population_metrics,
    robust_cost,
    selection_score,
    state_metrics,
)
from oe_update_corrector_probe import paired_bootstrap


@dataclass(frozen=True)
class TrainConfig:
    hidden: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_candidates: int
    goal_weight: float
    anchor_weight: float


class StructuredDynamicsHead(nn.Module):
    def __init__(
        self,
        *,
        candidate_width: int,
        context_width: int,
        output_width: int,
        hidden: int,
        predict_goal: bool,
    ):
        super().__init__()
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
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.terminal = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_width),
        )
        self.goal = (
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, output_width),
            )
            if predict_goal
            else None
        )

    def forward(
        self,
        candidate: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        candidate_encoded = self.candidate(candidate)
        context_encoded = self.context(context)
        terminal = self.terminal(
            torch.cat([candidate_encoded, context_encoded], dim=-1)
        )
        goal = self.goal(context_encoded) if self.goal is not None else None
        return terminal, goal


def pose_features(state: np.ndarray) -> np.ndarray:
    angle = state[..., 4]
    return np.concatenate(
        [
            state[..., :4],
            np.sin(angle)[..., None],
            np.cos(angle)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


def relative_features(
    terminal: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    angle = terminal[..., 4] - goal[..., 4]
    return np.concatenate(
        [
            terminal[..., :4] - goal[..., :4],
            np.sin(angle)[..., None],
            np.cos(angle)[..., None],
        ],
        axis=-1,
    ).astype(np.float32)


def physical_context(
    initial: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    difference = goal - initial
    return np.concatenate(
        [initial, goal, difference, np.abs(difference)],
        axis=-1,
    ).astype(np.float32)


def rank_fraction(cost: np.ndarray) -> np.ndarray:
    order = np.argsort(cost, axis=1, kind='stable')
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(len(cost))[:, None]
    ranks[rows, order] = np.arange(
        cost.shape[1],
        dtype=np.float32,
    )[None]
    return ranks / max(cost.shape[1] - 1, 1)


def load_latent_cache(path: Path, rows: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        features = np.asarray(archive['features'], dtype=np.float32)
        cached_rows = np.asarray(archive['rows'], dtype=np.int64)
    if not np.array_equal(cached_rows, rows):
        raise ValueError('latent cache/source rows do not match')
    return features


def load_dense_moment_cache(
    path: Path,
    rows: np.ndarray,
    *,
    projected_width: int = 32,
) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive['state_tokens'], dtype=np.float32)
        goal = np.asarray(archive['goal_tokens'], dtype=np.float32)
        cached_rows = np.asarray(archive['rows'], dtype=np.int64)
        audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
    if not np.array_equal(cached_rows, rows):
        raise ValueError('dense cache/source rows do not match')
    if (
        state.ndim != 4
        or goal.ndim != 4
        or state.shape[2:] != goal.shape[2:]
        or goal.shape[1] != 1
    ):
        raise ValueError(
            f'invalid dense token shapes {state.shape} and {goal.shape}'
        )
    tokens = np.concatenate([state, goal], axis=1)
    patches = tokens.shape[2]
    grid = int(round(patches**0.5))
    if grid * grid != patches:
        raise ValueError(f'patch count is not square: {patches}')
    # A label-independent fixed projection controls context dimension while
    # preserving channel geometry.  Spatial moments then retain where token
    # content occurs instead of collapsing the grid into a CLS vector.
    rng = np.random.default_rng(20261010)
    random_matrix = rng.normal(
        size=(tokens.shape[-1], projected_width)
    )
    projection, _ = np.linalg.qr(random_matrix)
    projected = np.einsum(
        'sfpd,dk->sfpk',
        tokens,
        projection.astype(np.float32),
        optimize=True,
    )
    coordinate = np.linspace(-1.0, 1.0, grid, dtype=np.float32)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing='ij')
    basis = np.stack(
        [
            np.ones_like(xx),
            xx,
            yy,
            xx**2,
            yy**2,
            xx * yy,
        ],
        axis=-1,
    ).reshape(patches, 6)
    moments = np.einsum(
        'sfpk,pq->sfqk',
        projected,
        basis,
        optimize=True,
    ) / patches
    return moments.reshape(len(rows), -1).astype(np.float32), audit


def load_arrays(
    source: Path,
    *,
    family: str,
    target: str,
    latent_cache: Path | None,
    dense_cache: Path | None,
) -> dict:
    with np.load(source, allow_pickle=False) as archive:
        required = {
            'candidates',
            'terminal_state',
            'initial_state',
            'goal_state',
            'true',
            'pred',
            'prev_mean',
            'rows',
            'horizon',
            'goal_offset',
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{source} is missing fields {missing}')
        candidates = np.asarray(archive['candidates'])[:, 0].astype(
            np.float32
        )
        terminal = np.asarray(archive['terminal_state'])[:, 0].astype(
            np.float32
        )
        initial = np.asarray(archive['initial_state']).astype(np.float32)
        goal = np.asarray(archive['goal_state']).astype(np.float32)
        true_cost = np.asarray(archive['true'])[:, 0].astype(np.float32)
        base_cost = np.asarray(archive['pred'])[:, 0, :, 0].astype(
            np.float32
        )
        previous_mean = np.asarray(archive['prev_mean'])[:, 0].astype(
            np.float32
        )
        rows = np.asarray(archive['rows'], dtype=np.int64)
        horizon = int(np.asarray(archive['horizon']).item())
        goal_offset = int(np.asarray(archive['goal_offset']).item())

    states, rounds, population, action_horizon, action_dim = candidates.shape
    if terminal.shape[:3] != (states, rounds, population):
        raise ValueError('terminal/candidate population shape mismatch')
    if family == 'latent':
        if latent_cache is None:
            raise ValueError('latent family requires --latent-cache')
        context = load_latent_cache(latent_cache, rows)
    elif family == 'state_oracle':
        context = physical_context(initial, goal)
    elif family == 'dense_moment':
        if dense_cache is None:
            raise ValueError('dense_moment family requires --dense-cache')
        context, dense_audit = load_dense_moment_cache(
            dense_cache,
            rows,
        )
    else:
        raise ValueError(f'unknown family {family!r}')

    candidate_action = candidates.reshape(
        states * rounds,
        population,
        action_horizon * action_dim,
    )
    base_normalized = robust_cost(
        base_cost.reshape(states * rounds, population)
    ).astype(np.float32)
    base_rank = rank_fraction(
        base_cost.reshape(states * rounds, population)
    )
    candidate_features = np.concatenate(
        [
            candidate_action,
            base_normalized[..., None],
            base_rank[..., None],
        ],
        axis=-1,
    ).reshape(states * rounds * population, -1)
    candidate_state_ids = np.repeat(
        np.arange(states, dtype=np.int64),
        rounds * population,
    )
    candidate_population_ids = np.repeat(
        np.arange(states * rounds, dtype=np.int64),
        population,
    )
    goal_per_candidate = np.repeat(
        goal[:, None, None, :],
        rounds,
        axis=1,
    )
    goal_per_candidate = np.repeat(
        goal_per_candidate,
        population,
        axis=2,
    )
    if target == 'terminal':
        target_values = pose_features(terminal).reshape(-1, 6)
        goal_values = pose_features(goal).astype(np.float32)
    elif target == 'relative':
        target_values = relative_features(
            terminal,
            goal_per_candidate,
        ).reshape(-1, 6)
        goal_values = None
    else:
        raise ValueError(f'unknown target {target!r}')

    flattened_populations = states * rounds
    return {
        'source': source,
        'label': f'h{horizon}_off{goal_offset}',
        'rows': rows,
        'family': family,
        'target': target,
        'context': context,
        'candidate_features': candidate_features,
        'candidate_state_ids': candidate_state_ids,
        'candidate_population_ids': candidate_population_ids,
        'target_values': target_values,
        'goal_values': goal_values,
        'candidate': candidates.reshape(
            flattened_populations,
            population,
            action_horizon,
            action_dim,
        ),
        'oracle_cost': true_cost.reshape(
            flattened_populations,
            population,
        ),
        'base_cost': base_cost.reshape(
            flattened_populations,
            population,
        ),
        'base_logits': -base_normalized,
        'proposal_mean': previous_mean.reshape(
            flattened_populations,
            action_horizon,
            action_dim,
        ),
        'state_ids': np.repeat(np.arange(states), rounds),
        'num_states': states,
        'num_rounds': rounds,
        'num_candidates': population,
        'dense_audit': (
            dense_audit if family == 'dense_moment' else None
        ),
    }


def indices_for_states(arrays: dict, states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(
        np.isin(arrays['candidate_state_ids'], states)
    )


def population_indices_for_states(
    arrays: dict,
    states: np.ndarray,
) -> np.ndarray:
    return np.flatnonzero(np.isin(arrays['state_ids'], states))


def fit_normalization(arrays: dict, states: np.ndarray) -> dict:
    candidate_indices = indices_for_states(arrays, states)
    context = arrays['context'][states]
    candidate = arrays['candidate_features'][candidate_indices]
    target = arrays['target_values'][candidate_indices]

    def moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        return (
            mean.astype(np.float32),
            np.where(scale > 1e-5, scale, 1.0).astype(np.float32),
        )

    context_mean, context_scale = moments(context)
    candidate_mean, candidate_scale = moments(candidate)
    target_mean, target_scale = moments(target)
    result = {
        'context_mean': context_mean,
        'context_scale': context_scale,
        'candidate_mean': candidate_mean,
        'candidate_scale': candidate_scale,
        'target_mean': target_mean,
        'target_scale': target_scale,
    }
    if arrays['goal_values'] is not None:
        goal_mean, goal_scale = moments(arrays['goal_values'][states])
        result['goal_mean'] = goal_mean
        result['goal_scale'] = goal_scale
    return result


def make_model(
    arrays: dict,
    config: TrainConfig,
) -> StructuredDynamicsHead:
    return StructuredDynamicsHead(
        candidate_width=arrays['candidate_features'].shape[-1],
        context_width=arrays['context'].shape[-1],
        output_width=arrays['target_values'].shape[-1],
        hidden=config.hidden,
        predict_goal=arrays['target'] == 'terminal',
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batch_tensors(
    arrays: dict,
    indices: np.ndarray,
    *,
    normalization: dict,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    state_ids = arrays['candidate_state_ids'][indices]
    candidate = (
        arrays['candidate_features'][indices]
        - normalization['candidate_mean']
    ) / normalization['candidate_scale']
    context = (
        arrays['context'][state_ids]
        - normalization['context_mean']
    ) / normalization['context_scale']
    target = (
        arrays['target_values'][indices]
        - normalization['target_mean']
    ) / normalization['target_scale']
    tensors = [
        torch.as_tensor(candidate, device=device),
        torch.as_tensor(context, device=device),
        torch.as_tensor(target, device=device),
    ]
    if arrays['goal_values'] is not None:
        goal = (
            arrays['goal_values'][state_ids]
            - normalization['goal_mean']
        ) / normalization['goal_scale']
        tensors.append(torch.as_tensor(goal, device=device))
    return tuple(tensors)


def train_checkpoints(
    arrays: dict,
    states: np.ndarray,
    *,
    config: TrainConfig,
    checkpoints: list[int],
    seed: int,
    device: torch.device,
) -> tuple[dict[int, dict[str, torch.Tensor]], dict]:
    seed_everything(seed)
    normalization = fit_normalization(arrays, states)
    model = make_model(arrays, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    indices = indices_for_states(arrays, states)
    rng = np.random.default_rng(seed)
    saved = {}
    checkpoint_set = set(checkpoints)
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        shuffled = rng.permutation(indices)
        for offset in range(
            0,
            len(shuffled),
            config.batch_candidates,
        ):
            batch = shuffled[offset : offset + config.batch_candidates]
            tensors = batch_tensors(
                arrays,
                batch,
                normalization=normalization,
                device=device,
            )
            candidate, context, target_values = tensors[:3]
            predicted, predicted_goal = model(candidate, context)
            terminal_loss = F.smooth_l1_loss(predicted, target_values)
            if predicted_goal is not None:
                goal_values = tensors[3]
                goal_loss = F.smooth_l1_loss(
                    predicted_goal,
                    goal_values,
                )
            else:
                goal_loss = terminal_loss.new_zeros(())
            anchor = sum(
                parameter.square().mean()
                for parameter in model.parameters()
            )
            loss = (
                terminal_loss
                + config.goal_weight * goal_loss
                + config.anchor_weight * anchor
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
    return saved, normalization


def train_model(
    arrays: dict,
    states: np.ndarray,
    *,
    config: TrainConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[StructuredDynamicsHead, dict]:
    checkpoints, normalization = train_checkpoints(
        arrays,
        states,
        config=config,
        checkpoints=[epochs],
        seed=seed,
        device=device,
    )
    model = make_model(arrays, config).to(device)
    model.load_state_dict(checkpoints[epochs])
    return model.eval(), normalization


def decode(
    values: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return values * scale + mean


def geometry_cost(
    terminal_or_relative: np.ndarray,
    *,
    target: str,
    goal: np.ndarray | None,
) -> np.ndarray:
    if target == 'relative':
        position = np.linalg.norm(
            terminal_or_relative[:, :4],
            axis=1,
        )
        angle = np.abs(
            np.arctan2(
                terminal_or_relative[:, 4],
                terminal_or_relative[:, 5],
            )
        )
    else:
        if goal is None:
            raise ValueError('terminal target requires predicted goal')
        position = np.linalg.norm(
            terminal_or_relative[:, :4] - goal[:, :4],
            axis=1,
        )
        terminal_angle = np.arctan2(
            terminal_or_relative[:, 4],
            terminal_or_relative[:, 5],
        )
        goal_angle = np.arctan2(goal[:, 4], goal[:, 5])
        difference = np.abs(terminal_angle - goal_angle)
        angle = np.minimum(difference, 2 * math.pi - difference)
    angle_scale = 20.0 / (math.pi / 9.0)
    return np.hypot(position, angle_scale * angle).astype(np.float32)


@torch.inference_mode()
def predict_cost(
    model: StructuredDynamicsHead,
    arrays: dict,
    candidate_indices: np.ndarray,
    *,
    normalization: dict,
    config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    output = np.empty(len(candidate_indices), dtype=np.float32)
    for offset in range(
        0,
        len(candidate_indices),
        config.batch_candidates,
    ):
        batch = candidate_indices[
            offset : offset + config.batch_candidates
        ]
        tensors = batch_tensors(
            arrays,
            batch,
            normalization=normalization,
            device=device,
        )
        predicted, predicted_goal = model(tensors[0], tensors[1])
        terminal = decode(
            predicted.float().cpu().numpy(),
            mean=normalization['target_mean'],
            scale=normalization['target_scale'],
        )
        if predicted_goal is not None:
            goal = decode(
                predicted_goal.float().cpu().numpy(),
                mean=normalization['goal_mean'],
                scale=normalization['goal_scale'],
            )
        else:
            goal = None
        output[offset : offset + len(batch)] = geometry_cost(
            terminal,
            target=arrays['target'],
            goal=goal,
        )
    return output


def corrected_metrics(
    arrays: dict,
    population_indices: np.ndarray,
    structured_cost: np.ndarray,
    *,
    blend: float,
    topk: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    structured = structured_cost.reshape(
        len(population_indices),
        arrays['num_candidates'],
    )
    structured_logits = -robust_cost(structured).astype(np.float32)
    base_logits = arrays['base_logits'][population_indices]
    logits = base_logits + blend * (structured_logits - base_logits)
    return (
        population_metrics(
            arrays,
            population_indices,
            logits,
            topk=topk,
        ),
        logits,
    )


def inner_states(
    outer_train_states: np.ndarray,
    *,
    fold: int,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.asarray(sorted(int(state) for state in outer_train_states))
    selector = np.arange(len(ordered)) % 5 == fold % 5
    validation = ordered[selector]
    training = ordered[~selector]
    if len(training) == 0 or len(validation) == 0:
        raise ValueError('empty inner split')
    return training, validation


def select_epoch_blend(
    arrays: dict,
    outer_train_states: np.ndarray,
    *,
    config: TrainConfig,
    topk: int,
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
            1,
            min(2, config.max_epochs),
            min(5, config.max_epochs),
            min(10, config.max_epochs),
            config.max_epochs,
        }
    )
    saved, normalization = train_checkpoints(
        arrays,
        training_states,
        config=config,
        checkpoints=checkpoints,
        seed=seed,
        device=device,
    )
    candidate_indices = indices_for_states(arrays, validation_states)
    population_indices = population_indices_for_states(
        arrays,
        validation_states,
    )
    rows = []
    for epoch in checkpoints:
        model = make_model(arrays, config).to(device)
        model.load_state_dict(saved[epoch])
        model.eval()
        predicted_cost = predict_cost(
            model,
            arrays,
            candidate_indices,
            normalization=normalization,
            config=config,
            device=device,
        )
        for blend in (0.25, 0.5, 0.75, 1.0):
            metrics, _ = corrected_metrics(
                arrays,
                population_indices,
                predicted_cost,
                blend=blend,
                topk=topk,
            )
            state = state_metrics(
                arrays,
                population_indices,
                metrics,
                validation_states,
            )
            rows.append(
                {
                    'epoch': epoch,
                    'blend': blend,
                    'score': selection_score(state),
                    **{
                        key: float(np.mean(value))
                        for key, value in state.items()
                    },
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


def analyze(
    arrays: dict,
    *,
    config: TrainConfig,
    topk: int,
    bootstrap: int,
    seed: int,
    device: torch.device,
) -> dict:
    states = np.arange(arrays['num_states'])
    corrected_logits = np.empty_like(arrays['base_logits'])
    folds = []
    for fold in range(3):
        validation_states = states[states % 3 == fold]
        training_states = states[states % 3 != fold]
        epoch, blend, selection_rows = select_epoch_blend(
            arrays,
            training_states,
            config=config,
            topk=topk,
            seed=seed + 10_000 * fold,
            fold=fold,
            device=device,
        )
        model, normalization = train_model(
            arrays,
            training_states,
            config=config,
            epochs=epoch,
            seed=seed + 100_000 + 10_000 * fold,
            device=device,
        )
        candidate_indices = indices_for_states(arrays, validation_states)
        population_indices = population_indices_for_states(
            arrays,
            validation_states,
        )
        predicted_cost = predict_cost(
            model,
            arrays,
            candidate_indices,
            normalization=normalization,
            config=config,
            device=device,
        )
        _, logits = corrected_metrics(
            arrays,
            population_indices,
            predicted_cost,
            blend=blend,
            topk=topk,
        )
        corrected_logits[population_indices] = logits
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

    population_indices = np.arange(len(arrays['base_logits']))
    baseline_population = population_metrics(
        arrays,
        population_indices,
        arrays['base_logits'],
        topk=topk,
    )
    corrected_population = population_metrics(
        arrays,
        population_indices,
        corrected_logits,
        topk=topk,
    )
    baseline_state = state_metrics(
        arrays,
        population_indices,
        baseline_population,
        states,
    )
    corrected_state = state_metrics(
        arrays,
        population_indices,
        corrected_population,
        states,
    )
    rng = np.random.default_rng(seed)
    metrics = {}
    state_rows = []
    for key in baseline_state:
        delta = corrected_state[key] - baseline_state[key]
        metrics[key] = {
            'baseline': float(np.mean(baseline_state[key])),
            'corrected': float(np.mean(corrected_state[key])),
            'delta': float(np.mean(delta)),
            'delta_ci': list(
                paired_bootstrap(
                    delta,
                    samples=bootstrap,
                    rng=rng,
                )
            ),
        }
    for state in states:
        row = {'state_index': int(state)}
        for key in baseline_state:
            row[f'baseline_{key}'] = float(baseline_state[key][state])
            row[f'corrected_{key}'] = float(corrected_state[key][state])
            row[f'delta_{key}'] = float(
                corrected_state[key][state] - baseline_state[key][state]
            )
        state_rows.append(row)
    return {
        'metrics': metrics,
        'folds': folds,
        'state_rows': state_rows,
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument(
        '--family',
        required=True,
        choices=('latent', 'state_oracle', 'dense_moment'),
    )
    parser.add_argument(
        '--target',
        required=True,
        choices=('terminal', 'relative'),
    )
    parser.add_argument('--latent-cache', type=Path)
    parser.add_argument('--dense-cache', type=Path)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--batch-candidates', type=int, default=4096)
    parser.add_argument('--goal-weight', type=float, default=1.0)
    parser.add_argument('--anchor-weight', type=float, default=1e-7)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260860)
    args = parser.parse_args()
    if args.hidden < 1 or args.max_epochs < 1:
        raise ValueError('hidden and max-epochs must be positive')
    if args.batch_candidates < 1:
        raise ValueError('batch-candidates must be positive')
    if args.family == 'latent' and args.latent_cache is None:
        raise ValueError('latent family requires --latent-cache')
    if args.family == 'dense_moment' and args.dense_cache is None:
        raise ValueError('dense_moment family requires --dense-cache')

    config = TrainConfig(
        hidden=args.hidden,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_candidates=args.batch_candidates,
        goal_weight=args.goal_weight,
        anchor_weight=args.anchor_weight,
    )
    arrays = load_arrays(
        args.source,
        family=args.family,
        target=args.target,
        latent_cache=args.latent_cache,
        dense_cache=args.dense_cache,
    )
    analysis = analyze(
        arrays,
        config=config,
        topk=args.topk,
        bootstrap=args.bootstrap,
        seed=args.seed,
        device=torch.device('cuda'),
    )
    payload = {
        'version': 1,
        'source': str(args.source.resolve()),
        'rows': arrays['rows'].tolist(),
        'cell': arrays['label'],
        'family': args.family,
        'target': args.target,
        'config': asdict(config),
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'dense_cache': (
            str(args.dense_cache.resolve())
            if args.dense_cache is not None
            else None
        ),
        'dense_audit': arrays['dense_audit'],
        **analysis,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    lines = [
        '# Structured counterfactual dynamics-head probe',
        '',
        'Training epoch and frozen/structured cost fusion are selected on '
        'inner held-out states. Every reported prediction is from an outer '
        'state-held-out model.',
        '',
        f'- Cell: `{arrays["label"]}`',
        f'- Context: `{args.family}`',
        f'- Supervision: `{args.target}`',
        '',
        '| metric | frozen LeWM | structured | delta | paired 95% CI |',
        '|---|---:|---:|---:|---:|',
    ]
    for key in (
        'update_cosine',
        'relative_update_error',
        'elite_overlap',
        'selected_elite_true_cost',
    ):
        row = analysis['metrics'][key]
        lines.append(
            f'| {key} | {row["baseline"]:.3f} '
            f'| {row["corrected"]:.3f} '
            f'| {row["delta"]:+.3f} '
            f'| {format_ci(row["delta_ci"])} |'
        )
    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print('\n'.join(lines), flush=True)
    print(f'results -> {args.out_dir}', flush=True)


if __name__ == '__main__':
    main()
