"""Nested state-held-out probe for an independent candidate cost head.

The continuous OE update corrector can learn update magnitude while averaging
incompatible directions.  This probe changes the supervision interface: a
small frozen-WM head scores every candidate, and CEM still obtains its update
by taking the head's top-k candidates.

The head sees no simulator label at inference.  Candidate features contain
proposal-normalized actions and the frozen WM score/rank.  Context features
are selected from deployable planner statistics, frozen state/goal latents,
or privileged PushT state (upper bound only).

Architecture, epoch, and residual blend are selected on inner held-out states.
Every reported outer prediction is therefore made for a state that
participated in neither fitting nor model selection.
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

from oe_update_corrector_probe import (
    EPS,
    feature_matrix,
    load_trace,
    paired_bootstrap,
)


@dataclass(frozen=True)
class TrainConfig:
    hidden: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    batch_populations: int
    listwise_weight: float
    anchor_weight: float


class CandidateCostHead(nn.Module):
    def __init__(
        self,
        *,
        candidate_width: int,
        context_width: int,
        hidden: int,
    ):
        super().__init__()
        if hidden > 0:
            self.candidate = nn.Sequential(
                nn.Linear(candidate_width, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.context = nn.Sequential(
                nn.Linear(context_width, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
            )
            self.output = nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
        else:
            self.candidate = nn.Linear(candidate_width, 1, bias=False)
            self.context = nn.Linear(context_width, 1, bias=False)
            self.output = None
            self.bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        candidate: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        candidate_encoded = self.candidate(candidate)
        context_encoded = self.context(context)
        if self.output is None:
            return (
                candidate_encoded[..., 0]
                + context_encoded[:, None, 0]
                + self.bias
            )
        expanded_context = context_encoded[:, None].expand(
            -1,
            candidate_encoded.shape[1],
            -1,
        )
        return self.output(
            torch.cat([candidate_encoded, expanded_context], dim=-1)
        )[..., 0]


def robust_cost(cost: np.ndarray) -> np.ndarray:
    median = np.median(cost, axis=1, keepdims=True)
    q25 = np.quantile(cost, 0.25, axis=1, keepdims=True)
    q75 = np.quantile(cost, 0.75, axis=1, keepdims=True)
    scale = np.maximum((q75 - q25) / 1.349, 1e-5)
    return (cost - median) / scale


def rank_fraction(cost: np.ndarray) -> np.ndarray:
    order = np.argsort(cost, axis=1, kind='stable')
    ranks = np.empty_like(order, dtype=np.float32)
    rows = np.arange(len(cost))[:, None]
    ranks[rows, order] = np.arange(cost.shape[1], dtype=np.float32)[None]
    return ranks / max(cost.shape[1] - 1, 1)


def load_arrays(
    source: Path,
    *,
    topk: int,
    family: str,
    latent_cache: Path | None,
) -> dict:
    trace = load_trace(
        source,
        topk=topk,
        latent_cache=latent_cache,
    )
    with np.load(source, allow_pickle=False) as archive:
        candidates = np.asarray(archive['candidates'])[:, 0].astype(
            np.float32
        )
        predicted = np.asarray(archive['pred'])[:, 0, :, 0].astype(
            np.float32
        )
        true_cost = np.asarray(archive['true'])[:, 0].astype(np.float32)
        previous_mean = np.asarray(archive['prev_mean'])[:, 0].astype(
            np.float32
        )
        previous_std = np.maximum(
            np.asarray(archive['prev_var'])[:, 0].astype(np.float32),
            1e-5,
        )
        rows = np.asarray(archive['rows'], dtype=np.int64)

    states, rounds, population, horizon, action_dim = candidates.shape
    flattened = states * rounds
    candidate = candidates.reshape(
        flattened,
        population,
        horizon,
        action_dim,
    )
    base_cost = predicted.reshape(flattened, population)
    oracle_cost = true_cost.reshape(flattened, population)
    proposal_mean = previous_mean.reshape(
        flattened,
        horizon,
        action_dim,
    )
    proposal_std = previous_std.reshape(
        flattened,
        horizon,
        action_dim,
    )
    state_ids = np.repeat(np.arange(states), rounds)
    round_ids = np.tile(np.arange(rounds), states)

    normalized_action = (
        (candidate - proposal_mean[:, None]) / proposal_std[:, None]
    )
    learned_order = np.argsort(base_cost, axis=1, kind='stable')[:, :topk]
    learned_elite = np.take_along_axis(
        candidate,
        learned_order[:, :, None, None],
        axis=1,
    )
    learned_mean = learned_elite.mean(axis=1)
    relative_to_learned = (
        (candidate - learned_mean[:, None]) / proposal_std[:, None]
    )
    base_normalized = robust_cost(base_cost).astype(np.float32)
    base_rank = rank_fraction(base_cost)
    learned_distance = np.mean(
        np.square(relative_to_learned),
        axis=(2, 3),
    )
    candidate_features = np.concatenate(
        [
            normalized_action.reshape(flattened, population, -1),
            relative_to_learned.reshape(flattened, population, -1),
            base_normalized[..., None],
            base_rank[..., None],
            learned_distance[..., None],
        ],
        axis=2,
    ).astype(np.float32)
    context_features = feature_matrix(trace, family).astype(np.float32)
    if len(context_features) != flattened:
        raise ValueError('trace context/population length mismatch')

    oracle_order = np.argsort(oracle_cost, axis=1, kind='stable')[:, :topk]
    oracle_mask = np.zeros(
        (flattened, population),
        dtype=np.float32,
    )
    oracle_mask[
        np.arange(flattened)[:, None],
        oracle_order,
    ] = 1.0
    return {
        'trace': trace,
        'rows': rows,
        'candidate_features': candidate_features,
        'context_features': context_features,
        'base_logits': -base_normalized.astype(np.float32),
        'oracle_mask': oracle_mask,
        'candidate': candidate,
        'oracle_cost': oracle_cost,
        'proposal_mean': proposal_mean,
        'proposal_std': proposal_std,
        'state_ids': state_ids,
        'round_ids': round_ids,
        'num_states': states,
        'num_rounds': rounds,
        'num_candidates': population,
    }


def append_outcome_features(
    arrays: dict,
    path: Path,
) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        required = {'rows', 'terminal_embeddings', 'goal_embeddings'}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f'{path} is missing outcome fields {missing}')
        rows = np.asarray(archive['rows'], dtype=np.int64)
        terminal = np.asarray(
            archive['terminal_embeddings'],
            dtype=np.float32,
        )
        goal = np.asarray(
            archive['goal_embeddings'],
            dtype=np.float32,
        )
        audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
    if not np.array_equal(rows, arrays['rows']):
        raise ValueError(
            f'outcome-cache/source row mismatch for {path}: '
            f'{np.sum(rows != arrays["rows"])} differing rows'
        )
    expected = (
        arrays['num_states'],
        arrays['num_rounds'],
        arrays['num_candidates'],
    )
    if terminal.ndim != 4 or terminal.shape[:3] != expected:
        raise ValueError(
            f'expected terminal embeddings {expected}xD, got '
            f'{terminal.shape}'
        )
    if goal.shape != (expected[0], terminal.shape[-1]):
        raise ValueError(
            f'goal embeddings do not align: {goal.shape} vs '
            f'{terminal.shape}'
        )
    relative = (terminal - goal[:, None, None]).reshape(
        expected[0] * expected[1],
        expected[2],
        terminal.shape[-1],
    )
    arrays['candidate_features'] = np.concatenate(
        [arrays['candidate_features'], relative],
        axis=-1,
    )
    arrays['outcome_audit'] = audit
    return arrays


def normalizers(
    arrays: dict,
    population_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    context = arrays['context_features'][population_indices]
    candidate = arrays['candidate_features'][population_indices]
    context_mean = context.mean(axis=0)
    context_scale = context.std(axis=0)
    candidate_mean = candidate.mean(axis=(0, 1))
    candidate_scale = candidate.std(axis=(0, 1))
    return {
        'context_mean': context_mean.astype(np.float32),
        'context_scale': np.where(
            context_scale > 1e-5,
            context_scale,
            1.0,
        ).astype(np.float32),
        'candidate_mean': candidate_mean.astype(np.float32),
        'candidate_scale': np.where(
            candidate_scale > 1e-5,
            candidate_scale,
            1.0,
        ).astype(np.float32),
    }


def population_indices(arrays: dict, states: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.isin(arrays['state_ids'], states))


def tensors_for(
    arrays: dict,
    indices: np.ndarray,
    *,
    normalization: dict,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    candidate = (
        arrays['candidate_features'][indices]
        - normalization['candidate_mean']
    ) / normalization['candidate_scale']
    context = (
        arrays['context_features'][indices]
        - normalization['context_mean']
    ) / normalization['context_scale']
    return (
        torch.as_tensor(candidate, device=device),
        torch.as_tensor(context, device=device),
        torch.as_tensor(
            arrays['base_logits'][indices],
            device=device,
        ),
        torch.as_tensor(
            arrays['oracle_mask'][indices],
            device=device,
        ),
    )


def train_model(
    arrays: dict,
    states: np.ndarray,
    *,
    config: TrainConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> tuple[CandidateCostHead, dict]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    indices = population_indices(arrays, states)
    normalization = normalizers(arrays, indices)
    model = CandidateCostHead(
        candidate_width=arrays['candidate_features'].shape[-1],
        context_width=arrays['context_features'].shape[-1],
        hidden=config.hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(seed)
    elite_fraction = float(
        arrays['oracle_mask'][indices[0]].mean()
    )
    positive_weight = (1.0 - elite_fraction) / elite_fraction
    for _ in range(epochs):
        shuffled = rng.permutation(indices)
        model.train()
        for offset in range(
            0,
            len(shuffled),
            config.batch_populations,
        ):
            batch = shuffled[offset : offset + config.batch_populations]
            candidate, context, base_logits, target = tensors_for(
                arrays,
                batch,
                normalization=normalization,
                device=device,
            )
            residual = model(candidate, context)
            logits = base_logits + residual
            binary = F.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=torch.as_tensor(
                    positive_weight,
                    device=device,
                ),
            )
            log_probability = F.log_softmax(logits, dim=1)
            listwise = -(
                log_probability * target
            ).sum(dim=1) / target.sum(dim=1)
            anchor = residual.square().mean()
            loss = (
                binary
                + config.listwise_weight * listwise.mean()
                + config.anchor_weight * anchor
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model.eval(), normalization


@torch.inference_mode()
def predict_residual(
    model: CandidateCostHead,
    arrays: dict,
    indices: np.ndarray,
    *,
    normalization: dict,
    device: torch.device,
    batch_populations: int,
) -> np.ndarray:
    output = []
    for offset in range(0, len(indices), batch_populations):
        batch = indices[offset : offset + batch_populations]
        candidate, context, _, _ = tensors_for(
            arrays,
            batch,
            normalization=normalization,
            device=device,
        )
        output.append(model(candidate, context).float().cpu().numpy())
    return np.concatenate(output, axis=0)


def population_metrics(
    arrays: dict,
    indices: np.ndarray,
    logits: np.ndarray,
    *,
    topk: int,
) -> dict[str, np.ndarray]:
    candidate = arrays['candidate'][indices].astype(np.float64)
    true_cost = arrays['oracle_cost'][indices].astype(np.float64)
    previous = arrays['proposal_mean'][indices].astype(np.float64)
    predicted_order = np.argsort(
        -logits,
        axis=1,
        kind='stable',
    )[:, :topk]
    oracle_order = np.argsort(
        true_cost,
        axis=1,
        kind='stable',
    )[:, :topk]
    predicted_elite = np.take_along_axis(
        candidate,
        predicted_order[:, :, None, None],
        axis=1,
    )
    oracle_elite = np.take_along_axis(
        candidate,
        oracle_order[:, :, None, None],
        axis=1,
    )
    predicted_update = (predicted_elite.mean(axis=1) - previous).reshape(
        len(indices),
        -1,
    )
    oracle_update = (oracle_elite.mean(axis=1) - previous).reshape(
        len(indices),
        -1,
    )
    predicted_norm = np.linalg.norm(predicted_update, axis=1)
    oracle_norm = np.linalg.norm(oracle_update, axis=1)
    cosine = np.sum(predicted_update * oracle_update, axis=1) / np.maximum(
        predicted_norm * oracle_norm,
        EPS,
    )
    relative = np.linalg.norm(
        predicted_update - oracle_update,
        axis=1,
    ) / np.maximum(oracle_norm, EPS)
    overlap = np.asarray(
        [
            len(set(predicted.tolist()) & set(oracle.tolist())) / topk
            for predicted, oracle in zip(
                predicted_order,
                oracle_order,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    selected_true = np.take_along_axis(
        true_cost,
        predicted_order,
        axis=1,
    ).mean(axis=1)
    return {
        'update_cosine': cosine,
        'relative_update_error': relative,
        'elite_overlap': overlap,
        'selected_elite_true_cost': selected_true,
    }


def state_metrics(
    arrays: dict,
    population_indices_: np.ndarray,
    population_values: dict[str, np.ndarray],
    states: np.ndarray,
) -> dict[str, np.ndarray]:
    example_states = arrays['state_ids'][population_indices_]
    return {
        key: np.asarray(
            [
                values[example_states == state].mean()
                for state in states
            ],
            dtype=np.float64,
        )
        for key, values in population_values.items()
    }


def selection_score(
    metrics: dict[str, np.ndarray],
) -> float:
    return float(
        np.mean(
            metrics['update_cosine']
            - 0.5 * metrics['relative_update_error']
            + 0.5 * metrics['elite_overlap']
        )
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
    if len(validation) == 0 or len(training) == 0:
        raise ValueError('empty inner state split')
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
    train_states_, validation_states = inner_states(
        outer_train_states,
        fold=fold,
    )
    checkpoints = sorted(
        {
            1,
            min(2, config.max_epochs),
            min(4, config.max_epochs),
            config.max_epochs,
        }
    )
    rows = []
    for epoch in checkpoints:
        model, normalization = train_model(
            arrays,
            train_states_,
            config=config,
            epochs=epoch,
            seed=seed,
            device=device,
        )
        indices = population_indices(arrays, validation_states)
        residual = predict_residual(
            model,
            arrays,
            indices,
            normalization=normalization,
            device=device,
            batch_populations=config.batch_populations,
        )
        for blend in (0.25, 0.5, 1.0):
            logits = arrays['base_logits'][indices] + blend * residual
            metrics = population_metrics(
                arrays,
                indices,
                logits,
                topk=topk,
            )
            state = state_metrics(
                arrays,
                indices,
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
        torch.cuda.empty_cache()
    best = max(
        rows,
        key=lambda row: (
            row['score'],
            -row['epoch'],
            -row['blend'],
        ),
    )
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
    num_states = arrays['num_states']
    all_states = np.arange(num_states)
    corrected_logits = np.empty_like(arrays['base_logits'])
    folds = []
    for fold in range(3):
        validation_states = all_states[all_states % 3 == fold]
        training_states = all_states[all_states % 3 != fold]
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
        indices = population_indices(arrays, validation_states)
        residual = predict_residual(
            model,
            arrays,
            indices,
            normalization=normalization,
            device=device,
            batch_populations=config.batch_populations,
        )
        corrected_logits[indices] = (
            arrays['base_logits'][indices] + blend * residual
        )
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

    all_indices = np.arange(len(arrays['base_logits']))
    baseline_population = population_metrics(
        arrays,
        all_indices,
        arrays['base_logits'],
        topk=topk,
    )
    corrected_population = population_metrics(
        arrays,
        all_indices,
        corrected_logits,
        topk=topk,
    )
    baseline_state = state_metrics(
        arrays,
        all_indices,
        baseline_population,
        all_states,
    )
    corrected_state = state_metrics(
        arrays,
        all_indices,
        corrected_population,
        all_states,
    )
    rng = np.random.default_rng(seed)
    metric_rows = {}
    state_rows = []
    for key in baseline_state:
        delta = corrected_state[key] - baseline_state[key]
        metric_rows[key] = {
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
    for state in all_states:
        row = {'state_index': int(state)}
        for key in baseline_state:
            row[f'baseline_{key}'] = float(baseline_state[key][state])
            row[f'corrected_{key}'] = float(corrected_state[key][state])
            row[f'delta_{key}'] = float(
                corrected_state[key][state] - baseline_state[key][state]
            )
        state_rows.append(row)
    return {
        'metrics': metric_rows,
        'folds': folds,
        'state_rows': state_rows,
    }


def format_ci(values: list[float]) -> str:
    return f'[{values[0]:+.3f}, {values[1]:+.3f}]'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--family', required=True)
    parser.add_argument('--latent-cache', type=Path)
    parser.add_argument('--outcome-cache', type=Path)
    parser.add_argument('--topk', type=int, default=30)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--max-epochs', type=int, default=8)
    parser.add_argument('--batch-populations', type=int, default=16)
    parser.add_argument('--listwise-weight', type=float, default=0.2)
    parser.add_argument('--anchor-weight', type=float, default=1e-3)
    parser.add_argument('--bootstrap', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260840)
    args = parser.parse_args()
    if args.hidden < 0 or args.max_epochs < 1:
        raise ValueError('hidden must be non-negative and epochs positive')
    if args.batch_populations < 1:
        raise ValueError('batch-populations must be positive')
    if args.family in (
        'planner_latent',
        'planner_history_latent',
    ) and args.latent_cache is None:
        raise ValueError(f'{args.family} requires --latent-cache')
    if args.family == 'planner_outcome' and args.outcome_cache is None:
        raise ValueError('planner_outcome requires --outcome-cache')

    config = TrainConfig(
        hidden=args.hidden,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        batch_populations=args.batch_populations,
        listwise_weight=args.listwise_weight,
        anchor_weight=args.anchor_weight,
    )
    device = torch.device('cuda')
    base_family = (
        'planner'
        if args.family == 'planner_outcome'
        else args.family
    )
    arrays = load_arrays(
        args.source,
        topk=args.topk,
        family=base_family,
        latent_cache=args.latent_cache,
    )
    if args.outcome_cache is not None:
        arrays = append_outcome_features(arrays, args.outcome_cache)
    analysis = analyze(
        arrays,
        config=config,
        topk=args.topk,
        bootstrap=args.bootstrap,
        seed=args.seed,
        device=device,
    )
    payload = {
        'version': 1,
        'source': str(args.source.resolve()),
        'rows': arrays['rows'].tolist(),
        'cell': arrays['trace'].label,
        'family': args.family,
        'config': asdict(config),
        'topk': args.topk,
        'bootstrap': args.bootstrap,
        'seed': args.seed,
        'outcome_cache': (
            str(args.outcome_cache.resolve())
            if args.outcome_cache is not None
            else None
        ),
        'outcome_audit': arrays.get('outcome_audit'),
        **analysis,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'results.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    metrics = analysis['metrics']
    lines = [
        '# Candidate-level OE cost-head probe',
        '',
        'Architecture, epoch, and residual blend are selected on inner '
        'state-held-out data. Metrics pool one outer-fold prediction per '
        'state.',
        '',
        f'- Cell: `{arrays["trace"].label}`',
        f'- Features: `{args.family}`',
        f'- Hidden width: `{args.hidden}`',
        '',
        '| metric | baseline | corrected | delta | paired 95% CI |',
        '|---|---:|---:|---:|---:|',
    ]
    for key in (
        'update_cosine',
        'relative_update_error',
        'elite_overlap',
        'selected_elite_true_cost',
    ):
        row = metrics[key]
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
