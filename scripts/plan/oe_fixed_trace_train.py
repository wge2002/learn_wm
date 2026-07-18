"""Fixed-trace feasibility trainer for Optimizer-Equivalent WM.

This is deliberately narrower than the eventual planner-query aggregation
method.  It asks whether a single LeWM checkpoint can learn the simulator
elite update on saved CEM populations and generalize to held-out *states*.
The encoder and goal representation stay frozen; only the action encoder,
predictor, and prediction projector are adapted.

The loss combines:

* a balanced oracle-elite boundary classification loss;
* soft elite mean and log-standard-deviation matching;
* a weak normalized-score anchor to the base checkpoint.

Passing this gate is necessary but not sufficient.  A positive fixed-trace
result must still survive recursive proposal re-sampling and full closed-loop
MPC, followed by planner-query data aggregation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

import stable_worldmodel as swm
from stable_worldmodel.data import get_cache_dir

from candidate_oracle import (
    make_process,
    prepare_model_info,
    prepare_world_info,
)
from eval_wm import get_dataset, img_transform
from oe_update_resample import (
    comma_ints,
    elite_moments,
    sha256,
    validate_source,
)

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


def optional_ints(value, *, name: str) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    return comma_ints(value, name=name)


def validate_indices(
    indices: list[int],
    *,
    size: int,
    name: str,
) -> list[int]:
    if len(indices) != len(set(indices)):
        raise ValueError(f'{name} contains duplicate indices')
    invalid = [index for index in indices if not 0 <= index < size]
    if invalid:
        raise ValueError(
            f'{name} has indices outside [0, {size - 1}]: {invalid}'
        )
    return indices


def state_split(
    oe: DictConfig,
    *,
    num_states: int,
) -> tuple[list[int], list[int]]:
    train = optional_ints(oe.get('train_states'), name='oe.train_states')
    val = optional_ints(oe.get('val_states'), name='oe.val_states')
    if (train is None) != (val is None):
        raise ValueError(
            'oe.train_states and oe.val_states must be provided together'
        )
    if train is None:
        val_every = int(oe.get('val_every', 3))
        val_offset = int(oe.get('val_offset', 2))
        if val_every < 2:
            raise ValueError('oe.val_every must be at least two')
        if not 0 <= val_offset < val_every:
            raise ValueError('oe.val_offset must be inside oe.val_every')
        val = [
            index
            for index in range(num_states)
            if index % val_every == val_offset
        ]
        train = [index for index in range(num_states) if index not in val]

    train = validate_indices(
        train,
        size=num_states,
        name='oe.train_states',
    )
    val = validate_indices(val, size=num_states, name='oe.val_states')
    overlap = sorted(set(train) & set(val))
    if overlap:
        raise ValueError(f'train/val state overlap: {overlap}')
    if not train or not val:
        raise ValueError('train and val must each contain at least one state')
    return train, val


def round_indices(
    oe: DictConfig,
    *,
    source_steps: list[int],
) -> list[int]:
    requested = optional_ints(oe.get('steps'), name='oe.steps')
    if requested is None:
        return list(range(len(source_steps)))
    missing = sorted(set(requested) - set(source_steps))
    if missing:
        raise ValueError(f'oe.steps {missing} are not in {source_steps}')
    return [source_steps.index(step) for step in requested]


def expand_model_info(
    info: dict,
    *,
    num_candidates: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    expanded = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            target_dtype = dtype if value.is_floating_point() else None
            expanded[key] = (
                value.to(device=device, dtype=target_dtype)
                .unsqueeze(1)
                .expand(1, num_candidates, *value.shape[1:])
            )
        elif isinstance(value, np.ndarray):
            expanded[key] = np.repeat(
                value[:, None, ...],
                num_candidates,
                axis=1,
            )
        else:
            expanded[key] = value
    return expanded


@torch.no_grad()
def cache_state_embeddings(
    model,
    model_info: dict,
    *,
    action_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    expanded = expand_model_info(
        model_info,
        num_candidates=1,
        device=device,
        dtype=dtype,
    )
    dummy = torch.zeros(
        (1, 1, *action_shape),
        device=device,
        dtype=dtype,
    )
    model.get_cost(expanded, dummy)
    history = int(expanded['emb'].shape[2])
    if 'past_action' not in expanded:
        raise ValueError('matched-history trace must provide past_action')
    return {
        'emb': expanded['emb'][:, :1].detach().clone(),
        'goal_emb': expanded['goal_emb'].detach().clone(),
        'past_action': expanded['past_action'][:, :1].detach().clone(),
        'history': torch.asarray(history, device=device),
    }


def score_cached(
    model,
    cache: dict[str, torch.Tensor],
    candidates: torch.Tensor,
) -> torch.Tensor:
    if candidates.ndim != 3:
        raise ValueError(
            f'candidates must be (N,H,D), got {tuple(candidates.shape)}'
        )
    num_candidates = len(candidates)
    history = int(cache['history'])
    info = {
        # With cached embeddings, rollout only uses the history dimension.
        'pixels': torch.empty(
            (1, num_candidates, history),
            device=candidates.device,
            dtype=candidates.dtype,
        ),
        'goal': torch.empty(
            (1, num_candidates, 1),
            device=candidates.device,
            dtype=candidates.dtype,
        ),
        'past_action': cache['past_action'].expand(
            1,
            num_candidates,
            *cache['past_action'].shape[2:],
        ),
        'emb': cache['emb'].expand(
            1,
            num_candidates,
            *cache['emb'].shape[2:],
        ),
        'goal_emb': cache['goal_emb'],
    }
    return model.get_cost(info, candidates.unsqueeze(0)).squeeze(0).float()


def robust_normalize(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    detached = cost.detach()
    center = detached.median()
    q25, q75 = torch.quantile(
        detached,
        torch.asarray([0.25, 0.75], device=cost.device),
    )
    scale = ((q75 - q25) / 1.349).clamp_min(1e-4)
    return (cost - center) / scale, scale


def oracle_target(
    candidates: torch.Tensor,
    true_cost: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    elite_count = min(topk, len(candidates))
    indices = torch.argsort(true_cost, stable=True)[:elite_count]
    elite = candidates[indices]
    mask = torch.zeros(
        len(candidates),
        device=candidates.device,
        dtype=torch.bool,
    )
    mask[indices] = True
    return mask, elite.mean(dim=0), elite.std(dim=0)


def population_loss(
    predicted_cost: torch.Tensor,
    base_cost: torch.Tensor,
    candidates: torch.Tensor,
    true_cost: torch.Tensor,
    proposal_std: torch.Tensor,
    *,
    topk: int,
    temperature: float,
    boundary_weight: float,
    mean_weight: float,
    logstd_weight: float,
    anchor_weight: float,
    std_floor: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_mask, oracle_mean, oracle_std = oracle_target(
        candidates,
        true_cost,
        topk=topk,
    )
    normalized, _ = robust_normalize(predicted_cost)
    base_normalized, _ = robust_normalize(base_cost)
    boundary = torch.kthvalue(
        normalized.detach(),
        min(topk, len(normalized)),
    ).values
    logits = (boundary - normalized) / temperature

    positive = F.softplus(-logits[target_mask]).mean()
    negative = F.softplus(logits[~target_mask]).mean()
    boundary_loss = 0.5 * (positive + negative)

    membership = torch.sigmoid(logits)
    weights = membership / membership.sum().clamp_min(1e-8)
    predicted_mean = torch.sum(
        weights[:, None, None] * candidates,
        dim=0,
    )
    variance = torch.sum(
        weights[:, None, None] * (candidates - predicted_mean[None]).square(),
        dim=0,
    )
    effective_n = weights.square().sum().reciprocal()
    correction = effective_n / (effective_n - 1.0).clamp_min(1.0)
    predicted_std = (variance * correction).clamp_min(std_floor**2).sqrt()

    update_scale = proposal_std.clamp_min(0.05)
    mean_loss = ((predicted_mean - oracle_mean) / update_scale).square().mean()
    logstd_loss = (
        (
            torch.log(predicted_std.clamp_min(std_floor))
            - torch.log(oracle_std.clamp_min(std_floor))
        )
        .square()
        .mean()
    )
    anchor_loss = (normalized - base_normalized).square().mean()
    total = (
        boundary_weight * boundary_loss
        + mean_weight * mean_loss
        + logstd_weight * logstd_loss
        + anchor_weight * anchor_loss
    )
    metrics = {
        'loss': float(total.detach()),
        'boundary_loss': float(boundary_loss.detach()),
        'mean_loss': float(mean_loss.detach()),
        'logstd_loss': float(logstd_loss.detach()),
        'anchor_loss': float(anchor_loss.detach()),
        'soft_elite_mass': float(membership.sum().detach()),
    }
    return total, metrics


def hard_update_metrics(
    predicted: np.ndarray,
    true_cost: np.ndarray,
    candidates: np.ndarray,
    prev_mean: np.ndarray,
    *,
    topk: int,
) -> dict[str, float]:
    learned_mean, _, learned_indices = elite_moments(
        candidates,
        predicted,
        topk=topk,
        std_floor=1e-6,
    )
    oracle_mean, _, oracle_indices = elite_moments(
        candidates,
        true_cost,
        topk=topk,
        std_floor=1e-6,
    )
    learned_update = (learned_mean - prev_mean).reshape(-1).astype(np.float64)
    oracle_update = (oracle_mean - prev_mean).reshape(-1).astype(np.float64)
    learned_norm = float(np.linalg.norm(learned_update))
    oracle_norm = float(np.linalg.norm(oracle_update))
    denominator = max(learned_norm * oracle_norm, 1e-12)
    cosine = float(np.dot(learned_update, oracle_update) / denominator)
    relative_error = float(
        np.linalg.norm(learned_update - oracle_update)
        / max(oracle_norm, 1e-12)
    )
    overlap = len(
        set(learned_indices.tolist()) & set(oracle_indices.tolist())
    ) / min(topk, len(candidates))
    return {
        'update_cosine': cosine,
        'relative_update_error': relative_error,
        'elite_overlap': float(overlap),
        'selected_elite_true_cost': float(np.mean(true_cost[learned_indices])),
    }


@torch.no_grad()
def evaluate_split(
    model,
    caches: list[dict[str, torch.Tensor]],
    result: dict[str, np.ndarray],
    *,
    generator_i: int,
    base_scorer_i: int,
    state_indices: list[int],
    selected_rounds: list[int],
    topk: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    state_rows = []
    for state_i in state_indices:
        population_rows = []
        for round_i in selected_rounds:
            candidates = result['candidates'][
                state_i,
                generator_i,
                round_i,
            ].astype(np.float32)
            candidates_t = torch.as_tensor(
                candidates,
                device=device,
                dtype=dtype,
            )
            predicted = (
                score_cached(model, caches[state_i], candidates_t)
                .cpu()
                .numpy()
            )
            metrics = hard_update_metrics(
                predicted,
                result['true'][state_i, generator_i, round_i],
                candidates,
                result['prev_mean'][
                    state_i,
                    generator_i,
                    round_i,
                ].astype(np.float32),
                topk=topk,
            )
            base_predicted = result['pred'][
                state_i,
                generator_i,
                round_i,
                base_scorer_i,
            ]
            metrics['base_score_mae'] = float(
                np.mean(np.abs(predicted - base_predicted))
            )
            population_rows.append(metrics)
        state_metric: dict[str, float | int] = {'state_index': state_i}
        state_metric.update(mean_metrics(population_rows))
        state_rows.append(state_metric)
    aggregate = {
        key: float(np.mean([row[key] for row in state_rows]))
        for key in state_rows[0]
        if key != 'state_index'
    }
    return aggregate, state_rows


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    oe = cfg.get('oe', {})
    source = Path(str(oe.get('source', '')))
    if not source.exists():
        raise FileNotFoundError(f'oe.source does not exist: {source}')
    base_policy = str(oe.get('policy', 'pd_d192_k3_eval'))
    source_generator = str(oe.get('source_generator', base_policy))
    run_name = str(oe.get('run_name', 'oe_fixed_trace'))
    output_dir = Path(
        str(
            oe.get(
                'out_dir',
                f'outputs/week1/{run_name}',
            )
        )
    )
    epochs = int(oe.get('epochs', 20))
    save_every = int(oe.get('save_every', 5))
    save_weights = bool(oe.get('save_weights', True))
    topk = int(oe.get('topk', 30))
    learning_rate = float(oe.get('lr', 1e-5))
    weight_decay = float(oe.get('weight_decay', 1e-4))
    temperature = float(oe.get('temperature', 0.2))
    boundary_weight = float(oe.get('boundary_weight', 1.0))
    mean_weight = float(oe.get('mean_weight', 1.0))
    logstd_weight = float(oe.get('logstd_weight', 0.25))
    anchor_weight = float(oe.get('anchor_weight', 0.02))
    std_floor = float(oe.get('std_floor', 1e-4))
    grad_clip = float(oe.get('grad_clip', 1.0))
    requested_modules = [
        item.strip()
        for item in str(
            oe.get(
                'modules',
                'action_encoder,predictor,pred_proj',
            )
        ).split(',')
        if item.strip()
    ]
    if epochs < 1 or save_every < 1:
        raise ValueError('oe.epochs and oe.save_every must be positive')
    if topk < 2:
        raise ValueError('oe.topk must be at least two')
    if learning_rate <= 0 or temperature <= 0 or std_floor <= 0:
        raise ValueError('lr, temperature, and std_floor must be positive')

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    for key in ('prev_mean', 'prev_var'):
        if key not in result:
            raise ValueError(f'source is missing {key!r}')
    generators = result['generators'].astype(str).tolist()
    scorers = result['scorers'].astype(str).tolist()
    if source_generator not in generators:
        raise ValueError(
            f'{source_generator!r} not in generators {generators}'
        )
    if base_policy not in scorers:
        raise ValueError(
            'The fixed-trace anchor requires the base policy to be one of '
            f'the stored scorers, got {base_policy!r} vs {scorers}'
        )
    generator_i = generators.index(source_generator)
    base_scorer_i = scorers.index(base_policy)

    num_states = result['candidates'].shape[0]
    train_states, val_states = state_split(oe, num_states=num_states)
    source_steps = result['steps'].astype(int).tolist()
    selected_rounds = round_indices(oe, source_steps=source_steps)
    num_candidates = result['candidates'].shape[3]
    if topk >= num_candidates:
        raise ValueError(
            f'oe.topk={topk} must be smaller than N={num_candidates}'
        )

    seed = int(oe.get('seed', cfg.seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda')
    model = swm.wm.utils.load_pretrained(base_policy).to(device).eval()
    model.interpolate_pos_encoding = True
    model.requires_grad_(False)
    dtype = next(model.parameters()).dtype

    cfg.eval.num_eval = 1
    cfg.world.max_episode_steps = (
        int(cfg.plan_config.horizon) * int(cfg.plan_config.action_block) + 5
    )
    world = swm.World(**cfg.world, image_shape=(224, 224))
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    process = make_process(dataset, cfg.dataset.keys_to_cache)
    transform = {
        'pixels': img_transform(cfg),
        'pixels_hist': img_transform(cfg),
        'goal': img_transform(cfg),
    }
    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)
    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    action_shape = tuple(result['candidates'].shape[-2:])
    caches = []
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    try:
        for state_i in range(num_states):
            info, initial_state, goal_state = prepare_world_info(
                world,
                dataset,
                episode=int(result['episodes'][state_i]),
                start=int(result['starts'][state_i]),
                goal_offset=int(result['goal_offset']),
                callables=callables,
                history_len=int(cfg.plan_config.history_len),
                action_block=int(cfg.plan_config.action_block),
            )
            max_state_mismatch = max(
                max_state_mismatch,
                float(
                    np.max(
                        np.abs(
                            initial_state - result['initial_state'][state_i]
                        )
                    )
                ),
            )
            max_goal_mismatch = max(
                max_goal_mismatch,
                float(
                    np.max(np.abs(goal_state - result['goal_state'][state_i]))
                ),
            )
            caches.append(
                cache_state_embeddings(
                    model,
                    prepare_model_info(policy, info),
                    action_shape=action_shape,
                )
            )
    finally:
        world.close()
    if max_state_mismatch > 1e-5 or max_goal_mismatch > 1e-5:
        raise RuntimeError(
            f'trace reconstruction mismatch: state={max_state_mismatch:.3e}, '
            f'goal={max_goal_mismatch:.3e}'
        )

    available_modules = {
        'action_encoder': model.action_encoder,
        'predictor': model.predictor,
        'pred_proj': model.pred_proj,
    }
    unknown_modules = sorted(set(requested_modules) - set(available_modules))
    if unknown_modules:
        raise ValueError(
            f'oe.modules has unknown entries {unknown_modules}; '
            f'choose from {sorted(available_modules)}'
        )
    if not requested_modules:
        raise ValueError('oe.modules must select at least one module')
    if len(requested_modules) != len(set(requested_modules)):
        raise ValueError('oe.modules contains duplicate entries')
    trainable_modules = {
        name: available_modules[name] for name in requested_modules
    }
    for module in trainable_modules.values():
        module.requires_grad_(True)
    parameters = [
        parameter
        for module in trainable_modules.values()
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = get_cache_dir(sub_folder='checkpoints') / run_name
    if save_weights:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        _, base_config = swm.wm.utils._resolve(
            base_policy,
            get_cache_dir(sub_folder='checkpoints'),
        )
        write_json(checkpoint_root / 'config.json', base_config)

    audit = {
        'version': 1,
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'base_policy': base_policy,
        'source_generator': source_generator,
        'run_name': run_name,
        'train_states': train_states,
        'val_states': val_states,
        'source_steps': source_steps,
        'selected_steps': [source_steps[index] for index in selected_rounds],
        'num_candidates': num_candidates,
        'topk': topk,
        'epochs': epochs,
        'save_every': save_every,
        'save_weights': save_weights,
        'seed': seed,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'temperature': temperature,
        'boundary_weight': boundary_weight,
        'mean_weight': mean_weight,
        'logstd_weight': logstd_weight,
        'anchor_weight': anchor_weight,
        'std_floor': std_floor,
        'grad_clip': grad_clip,
        'trainable_modules': sorted(trainable_modules),
        'trainable_parameters': sum(
            parameter.numel() for parameter in parameters
        ),
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
    }
    write_json(output_dir / 'audit.json', audit)
    if save_weights:
        write_json(checkpoint_root / 'oe_audit.json', audit)

    records = [
        (state_i, round_i)
        for state_i in train_states
        for round_i in selected_rounds
    ]
    history = []
    started = time.time()

    def evaluate(epoch: int, train_loss: dict[str, float] | None) -> dict:
        train_metrics, train_state_metrics = evaluate_split(
            model,
            caches,
            result,
            generator_i=generator_i,
            base_scorer_i=base_scorer_i,
            state_indices=train_states,
            selected_rounds=selected_rounds,
            topk=topk,
            device=device,
            dtype=dtype,
        )
        val_metrics, val_state_metrics = evaluate_split(
            model,
            caches,
            result,
            generator_i=generator_i,
            base_scorer_i=base_scorer_i,
            state_indices=val_states,
            selected_rounds=selected_rounds,
            topk=topk,
            device=device,
            dtype=dtype,
        )
        row = {
            'epoch': epoch,
            'elapsed_seconds': time.time() - started,
            'train_loss': train_loss,
            'train': train_metrics,
            'val': val_metrics,
            'train_state_metrics': train_state_metrics,
            'val_state_metrics': val_state_metrics,
        }
        history.append(row)
        write_json(output_dir / 'metrics.json', {'history': history})
        print(
            f'epoch={epoch:03d} '
            f'train_cos={train_metrics["update_cosine"]:.3f} '
            f'train_rel={train_metrics["relative_update_error"]:.3f} '
            f'val_cos={val_metrics["update_cosine"]:.3f} '
            f'val_rel={val_metrics["relative_update_error"]:.3f} '
            f'val_overlap={val_metrics["elite_overlap"]:.3f} '
            f'elapsed={(time.time() - started) / 60:.1f}m'
        )
        return row

    baseline_row = evaluate(0, None)
    baseline_score_mae = max(
        baseline_row['train']['base_score_mae'],
        baseline_row['val']['base_score_mae'],
    )
    if baseline_score_mae > 1e-3:
        raise RuntimeError(
            'cached scorer does not reproduce the source trace: '
            f'MAE={baseline_score_mae:.3e}'
        )
    for epoch in range(1, epochs + 1):
        random.shuffle(records)
        epoch_rows = []
        for state_i, round_i in records:
            candidates = torch.as_tensor(
                result['candidates'][
                    state_i,
                    generator_i,
                    round_i,
                ].astype(np.float32),
                device=device,
                dtype=dtype,
            )
            true_cost = torch.as_tensor(
                result['true'][state_i, generator_i, round_i],
                device=device,
                dtype=torch.float32,
            )
            base_cost = torch.as_tensor(
                result['pred'][
                    state_i,
                    generator_i,
                    round_i,
                    base_scorer_i,
                ],
                device=device,
                dtype=torch.float32,
            )
            proposal_std = torch.as_tensor(
                result['prev_var'][
                    state_i,
                    generator_i,
                    round_i,
                ].astype(np.float32),
                device=device,
                dtype=dtype,
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = score_cached(model, caches[state_i], candidates)
            loss, loss_metrics = population_loss(
                predicted,
                base_cost,
                candidates,
                true_cost,
                proposal_std,
                topk=topk,
                temperature=temperature,
                boundary_weight=boundary_weight,
                mean_weight=mean_weight,
                logstd_weight=logstd_weight,
                anchor_weight=anchor_weight,
                std_floor=std_floor,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'non-finite loss at state={state_i}, round={round_i}'
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, grad_clip)
            optimizer.step()
            epoch_rows.append(loss_metrics)

        train_loss = mean_metrics(epoch_rows)
        if epoch % save_every == 0 or epoch == epochs:
            evaluate(epoch, train_loss)
            if save_weights:
                checkpoint = checkpoint_root / f'weights_epoch_{epoch}.pt'
                torch.save(model.state_dict(), checkpoint)
                print(f'checkpoint -> {checkpoint}')

    if save_weights:
        torch.save(model.state_dict(), checkpoint_root / 'weights_final.pt')
    print(f'metrics -> {output_dir}')
    print(f'elapsed={(time.time() - started) / 60:.1f} minutes')


if __name__ == '__main__':
    run()
