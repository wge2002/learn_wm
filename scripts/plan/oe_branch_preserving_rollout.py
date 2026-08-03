"""Disjoint recursive smoke test for branch-preserving optimizer equivalence.

The fixed-trace audit found that true-vs-LeWM CEM corrections occupy opposite
spatial modes.  A best-of-M population operator retains a strong held-out
branch-set ceiling, and K3+K10 signed imagined terminal vectors provide the
best deployable routing signal.  This script tests whether that signal
survives the operation that matters: recursive proposal re-sampling.

The operator is trained on one complete query bank and evaluated on a
row-disjoint source trace.  Four methods start from the same saved K3 CEM
proposal:

``k3_1x300``
    Ordinary K3 CEM, 300 K3 trajectory calls per round.

``k3_2x150``
    Same-model two-start control, 300 K3 calls per round.

``bp_matched``
    Two K3+K10 branches with 75 candidates each after branching: 300 total
    world-model trajectory calls per round.

``bp_full``
    Two K3+K10 branches with 150 candidates each: 600 total calls, retained as
    a compute-unmatched mechanism ceiling.

``bp_sparse_matched`` (when a sparse K10 cache is supplied)
    K3 evaluates 272 candidates before the first split and 136 per retained
    branch thereafter.  K10 is queried only on the best 10% under K3, for
    299--300 total world-model trajectory calls per round.

Simulator outcomes are hidden from every update and branch selector.  They
are executed only to report candidate coverage and final outcome.  The
primary BP selector is the learned cumulative beam score; K3, K10, consensus,
and oracle-union final selectors are reported separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
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

from candidate_oracle import (
    execute_candidate,
    make_process,
    prepare_model_info,
    prepare_world_info,
)
from cem_round_oracle import execute_population
from eval_wm import get_dataset, img_transform
from oe_candidate_cost_head_probe import (
    load_arrays as load_candidate_arrays,
    rank_fraction,
    robust_cost,
)
from oe_fixed_trace_train import cache_state_embeddings
from oe_set_valued_operator_probe import (
    TrainConfig,
    decode_residual,
    load_outcome_cache,
    make_model,
    train_checkpoints,
)
from oe_update_corrector_probe import EPS, cost_features
from oe_update_resample import (
    elite_moments,
    quantize_candidates,
    sha256,
    validate_source,
)

warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)


@dataclass
class ProposalBranch:
    mean: np.ndarray
    std: np.ndarray
    log_score: float
    lineage: str
    mode: int = 0


@dataclass(frozen=True)
class Method:
    name: str
    branch_count: int
    candidates_per_branch: int
    use_operator: bool
    use_k10: bool
    operator_kind: str = 'none'
    k10_query_fraction: float = 1.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def cached_terminal_and_cost(
    model,
    cache: dict[str, torch.Tensor],
    candidates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    candidate_tensor = torch.as_tensor(
        candidates,
        device=device,
        dtype=dtype,
    )
    num_candidates = len(candidate_tensor)
    history = int(cache['history'])
    info = {
        'pixels': torch.empty(
            (1, num_candidates, history),
            device=device,
            dtype=dtype,
        ),
        'goal': torch.empty(
            (1, num_candidates, 1),
            device=device,
            dtype=dtype,
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
    cost = model.get_cost(
        info,
        candidate_tensor.unsqueeze(0),
    )[0]
    terminal = info['predicted_emb'][0, :, -1]
    return (
        terminal.float().cpu().numpy(),
        cost.float().cpu().numpy(),
    )


def outcome_features(
    terminal: np.ndarray,
    goal: np.ndarray,
    cost: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    relative = terminal - goal[None]
    if mask is None:
        normalized = robust_cost(cost[None])[0].astype(np.float32)
        rank = rank_fraction(cost[None])[0].astype(np.float32)
        parts = [
            relative.astype(np.float32),
            normalized[:, None],
            rank[:, None],
        ]
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(cost),) or int(mask.sum()) < 2:
            raise ValueError('outcome mask must select at least 2 candidates')
        selected_cost = cost[mask]
        normalized = np.zeros(len(cost), dtype=np.float32)
        normalized[mask] = robust_cost(selected_cost[None])[0]
        rank = np.zeros(len(cost), dtype=np.float32)
        rank[mask] = rank_fraction(selected_cost[None])[0]
        parts = [
            (relative * mask[:, None]).astype(np.float32),
            normalized[:, None],
            rank[:, None],
            mask[:, None].astype(np.float32),
        ]
    return np.concatenate(parts, axis=1)


def population_features(
    candidates: np.ndarray,
    *,
    proposal_mean: np.ndarray,
    proposal_std: np.ndarray,
    k3_cost: np.ndarray,
    k3_terminal: np.ndarray,
    k3_goal: np.ndarray,
    k10_cost: np.ndarray,
    k10_terminal: np.ndarray,
    k10_goal: np.ndarray,
    k10_mask: np.ndarray | None,
    topk: int,
    round_bucket: int,
    round_buckets: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    learned_order = np.argsort(
        k3_cost,
        kind='stable',
    )[:topk]
    learned_elite = candidates[learned_order].astype(np.float64)
    learned_mean = learned_elite.mean(axis=0).astype(np.float32)
    learned_std = np.maximum(
        learned_elite.std(axis=0, ddof=1),
        1e-4,
    ).astype(np.float32)
    normalized_action = (
        (candidates - proposal_mean[None]) / proposal_std[None]
    )
    relative_to_learned = (
        (candidates - learned_mean[None]) / proposal_std[None]
    )
    base_normalized = robust_cost(k3_cost[None])[0].astype(np.float32)
    base_rank = rank_fraction(k3_cost[None])[0].astype(np.float32)
    learned_distance = np.mean(
        np.square(relative_to_learned),
        axis=(1, 2),
    )
    candidate = np.concatenate(
        [
            normalized_action.reshape(len(candidates), -1),
            relative_to_learned.reshape(len(candidates), -1),
            base_normalized[:, None],
            base_rank[:, None],
            learned_distance[:, None],
            outcome_features(k3_terminal, k3_goal, k3_cost),
            outcome_features(
                k10_terminal,
                k10_goal,
                k10_cost,
                mask=k10_mask,
            ),
        ],
        axis=1,
    ).astype(np.float32)

    model_update = (
        (learned_mean - proposal_mean) / proposal_std
    ).reshape(-1)
    model_logstd = np.log(
        np.maximum(learned_std, 1e-6) / proposal_std
    ).reshape(-1)
    one_hot = np.zeros(round_buckets, dtype=np.float64)
    one_hot[round_bucket] = 1.0
    context = np.concatenate(
        [
            model_update,
            model_logstd,
            proposal_mean.reshape(-1),
            np.log(proposal_std).reshape(-1),
            one_hot,
            cost_features(k3_cost),
        ]
    ).astype(np.float32)
    return candidate, context, learned_mean, learned_std


@torch.inference_mode()
def operator_children(
    operator,
    normalization: dict,
    *,
    candidate_features: np.ndarray,
    context_features: np.ndarray,
    learned_mean: np.ndarray,
    learned_std: np.ndarray,
    proposal_mean: np.ndarray,
    proposal_std: np.ndarray,
    parent_score: float,
    parent_lineage: str,
    blend: float,
) -> list[ProposalBranch]:
    device = next(operator.parameters()).device
    candidate = (
        candidate_features - normalization['candidate_mean']
    ) / normalization['candidate_scale']
    context = (
        context_features - normalization['context_mean']
    ) / normalization['context_scale']
    residual_normalized, logits, _ = operator(
        torch.as_tensor(candidate[None], device=device),
        torch.as_tensor(context[None], device=device),
    )
    residual = decode_residual(
        residual_normalized,
        normalization,
    )[0].float().cpu().numpy()
    log_probability = F.log_softmax(logits[0], dim=0).cpu().numpy()
    learned_update = (
        (learned_mean - proposal_mean) / proposal_std
    ).reshape(-1)
    children = []
    for mode in range(len(residual)):
        corrected_update = learned_update + blend * residual[mode]
        corrected_mean = (
            proposal_mean
            + corrected_update.reshape(proposal_mean.shape) * proposal_std
        ).astype(np.float32)
        children.append(
            ProposalBranch(
                mean=corrected_mean,
                std=learned_std.copy(),
                log_score=float(parent_score + log_probability[mode]),
                lineage=f'{parent_lineage}.{mode}',
                mode=mode,
            )
        )
    return children


def prune_diverse(
    candidates: list[ProposalBranch],
    *,
    keep: int,
    threshold: float,
) -> list[ProposalBranch]:
    ordered = sorted(candidates, key=lambda branch: -branch.log_score)
    selected: list[ProposalBranch] = []
    for branch in ordered:
        if not selected:
            selected.append(branch)
        else:
            distances = []
            for previous in selected:
                scale = np.maximum(
                    0.5 * (branch.std + previous.std),
                    1e-4,
                )
                distance = np.linalg.norm(
                    (branch.mean - previous.mean) / scale
                ) / math.sqrt(branch.mean.size)
                distances.append(distance)
            if min(distances) >= threshold:
                selected.append(branch)
        if len(selected) == keep:
            break
    for branch in ordered:
        if len(selected) == keep:
            break
        if all(branch is not previous for previous in selected):
            selected.append(branch)
    return selected


def train_operator(
    *,
    source: Path,
    k3_outcome: Path,
    k10_outcome: Path,
    topk: int,
    epochs: int,
    seed: int,
    device: torch.device,
    exclude_rows: np.ndarray | None = None,
) -> tuple[torch.nn.Module, dict, TrainConfig, dict]:
    arrays = load_candidate_arrays(
        source,
        topk=topk,
        family='planner',
        latent_cache=None,
    )
    audits = []
    for outcome_path in (k3_outcome, k10_outcome):
        outcome = load_outcome_cache(
            outcome_path,
            source_rows=arrays['rows'],
            states=arrays['num_states'],
            rounds=arrays['num_rounds'],
            candidates=arrays['num_candidates'],
        )
        arrays['candidate_features'] = np.concatenate(
            [arrays['candidate_features'], outcome['features']],
            axis=-1,
        )
        audits.append(outcome['audit'])
    config = TrainConfig(
        hidden=128,
        attention_heads=4,
        correction_modes=5,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_epochs=epochs,
        batch_populations=8,
        router_weight=0.1,
        router_kind='winner_ce',
        router_temperature=0.1,
        delta_anchor_weight=1e-3,
    )
    states = np.arange(arrays['num_states'])
    excluded_rows = np.asarray(
        [] if exclude_rows is None else exclude_rows,
        dtype=np.int64,
    )
    if len(excluded_rows):
        states = states[
            ~np.isin(
                np.asarray(arrays['rows'], dtype=np.int64),
                excluded_rows,
            )
        ]
    if len(states) < 2:
        raise ValueError(
            'operator training needs at least two states after exclusion'
        )
    checkpoints, normalization, initial_modes = train_checkpoints(
        arrays,
        arrays['trace'],
        states,
        config=config,
        checkpoints=[epochs],
        seed=seed,
        device=device,
    )
    model = make_model(
        arrays,
        arrays['trace'],
        config,
        initial_modes=initial_modes,
    ).to(device)
    model.load_state_dict(checkpoints[epochs])
    model.eval()
    audit = {
        'source': str(source.resolve()),
        'source_sha256': sha256(source),
        'rows': arrays['rows'][states].tolist(),
        'excluded_rows': excluded_rows.tolist(),
        'outcomes': audits,
    }
    return model, normalization, config, audit


def method_definitions(
    *,
    include_sparse: bool,
    include_compute_controls: bool,
) -> list[Method]:
    methods = [
        Method('k3_1x300', 1, 300, False, False),
        Method('k3_2x150', 2, 150, False, False),
        Method(
            'bp_matched',
            2,
            75,
            True,
            True,
            operator_kind='full',
        ),
        Method(
            'bp_full',
            2,
            150,
            True,
            True,
            operator_kind='full',
        ),
    ]
    if include_sparse:
        methods.append(
            Method(
                'bp_sparse_matched',
                2,
                136,
                True,
                True,
                operator_kind='sparse',
                k10_query_fraction=0.1,
            )
        )
    if include_compute_controls:
        methods.extend(
            [
                Method('k3_1x600', 1, 600, False, False),
                Method('k3_2x300', 2, 300, False, False),
            ]
        )
    return methods


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    branch_cfg = cfg.get('branch_oe', {})
    train_source = Path(str(branch_cfg.get('train_source', '')))
    eval_source = Path(str(branch_cfg.get('eval_source', '')))
    k3_outcome = Path(str(branch_cfg.get('k3_outcome', '')))
    k10_outcome = Path(str(branch_cfg.get('k10_outcome', '')))
    sparse_raw = str(branch_cfg.get('k10_sparse_outcome', '')).strip()
    k10_sparse_outcome = Path(sparse_raw) if sparse_raw else None
    output = Path(str(branch_cfg.get('out', '')))
    for path in (train_source, eval_source, k3_outcome, k10_outcome):
        if not path.exists():
            raise FileNotFoundError(path)
    if (
        k10_sparse_outcome is not None
        and not k10_sparse_outcome.exists()
    ):
        raise FileNotFoundError(k10_sparse_outcome)
    if output == Path('.'):
        raise ValueError('branch_oe.out is required')

    state_start = int(branch_cfg.get('state_start', 0))
    num_states = int(branch_cfg.get('num_states', 2))
    num_rounds = int(branch_cfg.get('num_rounds', 3))
    start_step = int(branch_cfg.get('start_step', 4))
    train_topk = int(branch_cfg.get('train_topk', 30))
    train_epochs = int(branch_cfg.get('train_epochs', 10))
    blend = float(branch_cfg.get('blend', 0.5))
    diversity_threshold = float(
        branch_cfg.get('diversity_threshold', 0.25)
    )
    k3_name = str(branch_cfg.get('k3_policy', 'pd_d192_k3_eval'))
    k10_name = str(branch_cfg.get('k10_policy', 'pd_d192_k10_eval'))
    include_compute_controls = bool(
        branch_cfg.get('compute_controls', False)
    )
    evaluate_simulator = bool(
        branch_cfg.get('evaluate_simulator', True)
    )
    evaluate_populations = bool(
        branch_cfg.get(
            'evaluate_populations',
            evaluate_simulator,
        )
    )
    evaluate_final = bool(
        branch_cfg.get('evaluate_final', evaluate_simulator)
    )
    allow_train_eval_overlap = bool(
        branch_cfg.get('allow_train_eval_overlap', False)
    )
    operator_exclude_eval_rows = bool(
        branch_cfg.get('operator_exclude_eval_rows', False)
    )
    method_filter = {
        item.strip()
        for item in str(branch_cfg.get('methods', '')).split(',')
        if item.strip()
    }
    if state_start < 0:
        raise ValueError('state_start must be non-negative')
    if num_states < 1 or num_rounds < 1:
        raise ValueError('num_states and num_rounds must be positive')

    excluded_operator_rows = np.empty(0, dtype=np.int64)
    if operator_exclude_eval_rows:
        with np.load(eval_source, allow_pickle=False) as archive:
            eval_rows = np.asarray(archive['rows'], dtype=np.int64)
        excluded_operator_rows = eval_rows[
            state_start:min(state_start + num_states, len(eval_rows))
        ]

    seed_everything(int(cfg.seed))
    device = torch.device('cuda')
    operator, normalization, operator_config, train_audit = train_operator(
        source=train_source,
        k3_outcome=k3_outcome,
        k10_outcome=k10_outcome,
        topk=train_topk,
        epochs=train_epochs,
        seed=int(cfg.seed) + 701,
        device=device,
        exclude_rows=excluded_operator_rows,
    )
    operators = {'full': operator}
    normalizations = {'full': normalization}
    train_audits = {'full': train_audit}
    if k10_sparse_outcome is not None:
        sparse_operator, sparse_normalization, _, sparse_audit = (
            train_operator(
                source=train_source,
                k3_outcome=k3_outcome,
                k10_outcome=k10_sparse_outcome,
                topk=train_topk,
                epochs=train_epochs,
                seed=int(cfg.seed) + 701,
                device=device,
                exclude_rows=excluded_operator_rows,
            )
        )
        operators['sparse'] = sparse_operator
        normalizations['sparse'] = sparse_normalization
        train_audits['sparse'] = sparse_audit

    with np.load(eval_source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    total_eval_states = len(result['rows'])
    state_stop = min(state_start + num_states, total_eval_states)
    if state_start >= state_stop:
        raise ValueError(
            f'state slice [{state_start}:{state_start + num_states}] '
            f'is outside eval source with {total_eval_states} states'
        )
    # Every state-major array in this source has the same leading dimension.
    # Slice once so all downstream indexing stays local to this shard.
    for key, value in list(result.items()):
        if value.ndim > 0 and value.shape[0] == total_eval_states:
            result[key] = value[state_start:state_stop]
    num_states = state_stop - state_start
    train_rows = np.asarray(train_audit['rows'], dtype=np.int64)
    overlap = np.intersect1d(result['rows'], train_rows)
    if len(overlap) and not allow_train_eval_overlap:
        raise ValueError(f'train/eval overlap on {len(overlap)} rows')
    generators = result['generators'].astype(str).tolist()
    if k3_name not in generators:
        raise ValueError(f'{k3_name} is not an eval-source generator')
    generator_i = generators.index(k3_name)
    steps = result['steps'].astype(int).tolist()
    if start_step not in steps:
        raise ValueError(f'start step {start_step} absent from {steps}')
    start_round_i = steps.index(start_step)
    action_shape = tuple(result['candidates'].shape[-2:])

    models = {
        k3_name: swm.wm.utils.load_pretrained(k3_name).to(device).eval(),
        k10_name: swm.wm.utils.load_pretrained(k10_name).to(device).eval(),
    }
    for model in models.values():
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True

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
    plan_config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver, model=models[k3_name])
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=plan_config,
        process=process,
        transform=transform,
    )
    world.set_policy(policy)
    callables = cfg.eval.get('callables')
    if callables is not None:
        callables = OmegaConf.to_container(callables, resolve=True)

    methods = method_definitions(
        include_sparse=k10_sparse_outcome is not None,
        include_compute_controls=include_compute_controls,
    )
    if method_filter:
        unknown = sorted(
            method_filter - {method.name for method in methods}
        )
        if unknown:
            raise ValueError(f'unknown branch_oe.methods: {unknown}')
        methods = [
            method for method in methods
            if method.name in method_filter
        ]
    method_names = [method.name for method in methods]
    shape = (num_states, len(methods), num_rounds)
    population_min_true = np.full(shape, np.nan, dtype=np.float64)
    population_success = np.zeros(shape, dtype=bool)
    population_mean_true = np.full(shape, np.nan, dtype=np.float64)
    branch_count_history = np.zeros(shape, dtype=np.int16)
    branch_distance_history = np.full(shape, np.nan, dtype=np.float64)
    final_branch_true = np.full(
        (num_states, len(methods), 2),
        np.nan,
        dtype=np.float64,
    )
    final_branch_success = np.zeros(
        (num_states, len(methods), 2),
        dtype=bool,
    )
    selector_names = [
        'primary',
        'k3',
        'k10',
        'consensus',
        'oracle_union',
    ]
    selected_true = np.full(
        (num_states, len(methods), len(selector_names)),
        np.nan,
        dtype=np.float64,
    )
    selected_success = np.zeros_like(selected_true, dtype=bool)
    selected_index = np.full_like(selected_true, -1, dtype=np.int16)
    selected_modes = np.full(
        (num_states, len(methods), num_rounds, 2),
        -1,
        dtype=np.int16,
    )
    proposal_mean_history = np.full(
        (
            num_states,
            len(methods),
            num_rounds + 1,
            2,
            *action_shape,
        ),
        np.nan,
        dtype=np.float32,
    )
    proposal_std_history = np.full_like(
        proposal_mean_history,
        np.nan,
    )
    branch_log_score_history = np.full(
        (num_states, len(methods), num_rounds + 1, 2),
        np.nan,
        dtype=np.float32,
    )
    final_model_cost = np.full(
        (num_states, len(methods), 2, 2),
        np.nan,
        dtype=np.float32,
    )
    final_model_relative = None
    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_roundtrip_error = 0.0
    started = time.time()

    try:
        for state_i in range(num_states):
            source_state_i = state_start + state_i
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
            state_mismatch = float(
                np.max(
                    np.abs(
                        initial_state - result['initial_state'][state_i]
                    )
                )
            )
            goal_mismatch = float(
                np.max(
                    np.abs(goal_state - result['goal_state'][state_i])
                )
            )
            max_state_mismatch = max(max_state_mismatch, state_mismatch)
            max_goal_mismatch = max(max_goal_mismatch, goal_mismatch)
            if state_mismatch > 1e-5 or goal_mismatch > 1e-5:
                raise RuntimeError('eval source reconstruction mismatch')

            model_info = prepare_model_info(policy, info)
            caches = {
                name: cache_state_embeddings(
                    model,
                    model_info,
                    action_shape=action_shape,
                )
                for name, model in models.items()
            }
            goals = {
                name: cache['goal_emb'][0, -1]
                .float()
                .cpu()
                .numpy()
                for name, cache in caches.items()
            }
            if final_model_relative is None:
                goal_dims = {len(goal) for goal in goals.values()}
                if len(goal_dims) != 1:
                    raise ValueError(
                        'K3 and K10 terminal embeddings have different dims'
                    )
                final_model_relative = np.full(
                    (
                        num_states,
                        len(methods),
                        2,
                        2,
                        goal_dims.pop(),
                    ),
                    np.nan,
                    dtype=np.float32,
                )
            initial_mean = result['mean'][
                state_i,
                generator_i,
                start_round_i,
            ].astype(np.float32)
            initial_std = np.maximum(
                result['var'][
                    state_i,
                    generator_i,
                    start_round_i,
                ].astype(np.float32),
                1e-4,
            )
            execution_cache: dict[bytes, dict] = {}

            for method_i, method in enumerate(methods):
                initial_branches = (
                    method.branch_count
                    if not method.use_operator
                    else 1
                )
                branches = [
                    ProposalBranch(
                        mean=initial_mean.copy(),
                        std=initial_std.copy(),
                        log_score=0.0,
                        lineage=f'{branch_i}',
                    )
                    for branch_i in range(initial_branches)
                ]
                for branch_i, branch in enumerate(branches[:2]):
                    proposal_mean_history[
                        state_i, method_i, 0, branch_i
                    ] = branch.mean
                    proposal_std_history[
                        state_i, method_i, 0, branch_i
                    ] = branch.std
                    branch_log_score_history[
                        state_i, method_i, 0, branch_i
                    ] = branch.log_score
                for round_i in range(num_rounds):
                    hidden_true_rows = []
                    hidden_success_rows = []
                    children = []
                    for branch_i, branch in enumerate(branches):
                        per_branch = method.candidates_per_branch
                        if (
                            method.use_operator
                            and len(branches) == 1
                            and method.branch_count > 1
                        ):
                            per_branch *= method.branch_count
                        noise_rng = np.random.default_rng(
                            np.random.SeedSequence(
                                [
                                    int(cfg.seed),
                                    source_state_i,
                                    round_i,
                                    branch_i,
                                ]
                            )
                        )
                        raw = (
                            noise_rng.standard_normal(
                                (per_branch, *action_shape),
                                dtype=np.float32,
                            )
                            * branch.std[None]
                            + branch.mean[None]
                        )
                        raw[0] = branch.mean
                        candidates = quantize_candidates(raw)
                        k3_terminal, k3_cost = cached_terminal_and_cost(
                            models[k3_name],
                            caches[k3_name],
                            candidates,
                        )
                        k10_mask = None
                        if method.use_k10:
                            if method.k10_query_fraction < 1.0:
                                query_count = max(
                                    2,
                                    int(
                                        round(
                                            method.k10_query_fraction
                                            * per_branch
                                        )
                                    ),
                                )
                                query_indices = np.argsort(
                                    k3_cost,
                                    kind='stable',
                                )[:query_count]
                                queried_terminal, queried_cost = (
                                    cached_terminal_and_cost(
                                        models[k10_name],
                                        caches[k10_name],
                                        candidates[query_indices],
                                    )
                                )
                                k10_mask = np.zeros(
                                    per_branch,
                                    dtype=bool,
                                )
                                k10_mask[query_indices] = True
                                k10_terminal = np.repeat(
                                    goals[k10_name][None],
                                    per_branch,
                                    axis=0,
                                )
                                k10_cost = np.zeros(
                                    per_branch,
                                    dtype=np.float32,
                                )
                                k10_terminal[query_indices] = (
                                    queried_terminal
                                )
                                k10_cost[query_indices] = queried_cost
                            else:
                                k10_terminal, k10_cost = (
                                    cached_terminal_and_cost(
                                        models[k10_name],
                                        caches[k10_name],
                                        candidates,
                                    )
                                )
                        else:
                            k10_terminal = k3_terminal
                            k10_cost = k3_cost
                        topk = max(2, int(round(0.1 * per_branch)))
                        features, context, learned_mean, learned_std = (
                            population_features(
                                candidates,
                                proposal_mean=branch.mean,
                                proposal_std=branch.std,
                                k3_cost=k3_cost,
                                k3_terminal=k3_terminal,
                                k3_goal=goals[k3_name],
                                k10_cost=k10_cost,
                                k10_terminal=k10_terminal,
                                k10_goal=(
                                    goals[k10_name]
                                    if method.use_k10
                                    else goals[k3_name]
                                ),
                                k10_mask=k10_mask,
                                topk=topk,
                                round_bucket=int(
                                    np.argmin(
                                        np.abs(
                                            np.asarray([4, 9, 19, 29])
                                            - (
                                                start_step
                                                + round_i
                                                + 1
                                            )
                                        )
                                    )
                                ),
                                round_buckets=4,
                            )
                        )
                        if method.use_operator:
                            active_operator = operators[
                                method.operator_kind
                            ]
                            active_normalization = normalizations[
                                method.operator_kind
                            ]
                            children.extend(
                                operator_children(
                                    active_operator,
                                    active_normalization,
                                    candidate_features=features,
                                    context_features=context,
                                    learned_mean=learned_mean,
                                    learned_std=learned_std,
                                    proposal_mean=branch.mean,
                                    proposal_std=branch.std,
                                    parent_score=branch.log_score,
                                    parent_lineage=branch.lineage,
                                    blend=blend,
                                )
                            )
                        else:
                            children.append(
                                ProposalBranch(
                                    mean=learned_mean,
                                    std=learned_std,
                                    log_score=branch.log_score,
                                    lineage=branch.lineage,
                                )
                            )
                        if evaluate_populations:
                            execution = execute_population(
                                world.envs.envs[0],
                                candidates=candidates,
                                initial_state=initial_state,
                                goal_state=goal_state,
                                action_scaler=process['action'],
                                action_block=int(result['action_block']),
                                seed=int(cfg.seed) + source_state_i,
                                cache=execution_cache,
                            )
                            hidden_true_rows.append(execution['true'])
                            hidden_success_rows.append(
                                execution['success']
                            )
                            max_roundtrip_error = max(
                                max_roundtrip_error,
                                float(
                                    np.max(execution['roundtrip_error'])
                                ),
                            )

                    if method.use_operator:
                        branches = prune_diverse(
                            children,
                            keep=method.branch_count,
                            threshold=diversity_threshold,
                        )
                    else:
                        branches = children
                    if evaluate_populations:
                        hidden_true = np.concatenate(hidden_true_rows)
                        hidden_success = np.concatenate(
                            hidden_success_rows
                        )
                        population_min_true[
                            state_i, method_i, round_i
                        ] = float(np.min(hidden_true))
                        population_success[
                            state_i, method_i, round_i
                        ] = bool(np.any(hidden_success))
                        population_mean_true[
                            state_i, method_i, round_i
                        ] = float(np.mean(hidden_true))
                    branch_count_history[
                        state_i, method_i, round_i
                    ] = len(branches)
                    if len(branches) > 1:
                        scale = np.maximum(
                            0.5 * (branches[0].std + branches[1].std),
                            1e-4,
                        )
                        branch_distance_history[
                            state_i, method_i, round_i
                        ] = float(
                            np.linalg.norm(
                                (branches[0].mean - branches[1].mean)
                                / scale
                            )
                            / math.sqrt(branches[0].mean.size)
                        )
                    for out_i, branch in enumerate(branches[:2]):
                        selected_modes[
                            state_i, method_i, round_i, out_i
                        ] = branch.mode
                        proposal_mean_history[
                            state_i,
                            method_i,
                            round_i + 1,
                            out_i,
                        ] = branch.mean
                        proposal_std_history[
                            state_i,
                            method_i,
                            round_i + 1,
                            out_i,
                        ] = branch.std
                        branch_log_score_history[
                            state_i,
                            method_i,
                            round_i + 1,
                            out_i,
                        ] = branch.log_score
                    hidden_summary = (
                        f'min={hidden_true.min():.2f} '
                        f'coverage={int(hidden_success.any())}'
                        if evaluate_populations
                        else 'simulator=skipped'
                    )
                    print(
                        f'[source={source_state_i} '
                        f'local={state_i + 1}/{num_states}] '
                        f'{method.name} round={round_i + 1}/{num_rounds} '
                        f'{hidden_summary} '
                        f'branches={len(branches)} '
                        f'elapsed={(time.time() - started) / 60:.1f}m',
                        flush=True,
                    )

                means = np.asarray([branch.mean for branch in branches])
                k3_terminal, k3_final = cached_terminal_and_cost(
                    models[k3_name],
                    caches[k3_name],
                    means,
                )
                k10_terminal, k10_final = cached_terminal_and_cost(
                    models[k10_name],
                    caches[k10_name],
                    means,
                )
                final_model_cost[
                    state_i, method_i, : len(branches), 0
                ] = k3_final
                final_model_cost[
                    state_i, method_i, : len(branches), 1
                ] = k10_final
                final_model_relative[
                    state_i, method_i, : len(branches), 0
                ] = k3_terminal - goals[k3_name][None]
                final_model_relative[
                    state_i, method_i, : len(branches), 1
                ] = k10_terminal - goals[k10_name][None]
                final_true = []
                final_success = []
                for branch_i, branch in enumerate(branches):
                    if evaluate_final:
                        execution = execute_candidate(
                            world.envs.envs[0],
                            initial_state=initial_state,
                            goal_state=goal_state,
                            candidate=branch.mean,
                            action_scaler=process['action'],
                            action_block=int(result['action_block']),
                            seed=int(cfg.seed) + source_state_i,
                        )
                        final_true.append(execution['cost'])
                        final_success.append(execution['success'])
                        final_branch_true[
                            state_i, method_i, branch_i
                        ] = execution['cost']
                        final_branch_success[
                            state_i, method_i, branch_i
                        ] = execution['success']
                    else:
                        final_true.append(float('nan'))
                        final_success.append(False)
                final_true = np.asarray(final_true)
                final_success = np.asarray(final_success)
                beam_index = int(
                    np.argmax([branch.log_score for branch in branches])
                )
                k3_index = int(np.argmin(k3_final))
                k10_index = int(np.argmin(k10_final))
                k3_rank = rank_fraction(k3_final[None])[0]
                k10_rank = rank_fraction(k10_final[None])[0]
                consensus_index = int(np.argmin(k3_rank + k10_rank))
                oracle_index = (
                    int(np.argmin(final_true))
                    if evaluate_final
                    else 0
                )
                primary_index = (
                    beam_index if method.use_operator else k3_index
                )
                selectors = [
                    primary_index,
                    k3_index,
                    k10_index,
                    consensus_index,
                    oracle_index,
                ]
                for selector_i, index in enumerate(selectors):
                    selected_index[
                        state_i, method_i, selector_i
                    ] = index
                    selected_true[
                        state_i, method_i, selector_i
                    ] = final_true[index]
                    selected_success[
                        state_i, method_i, selector_i
                    ] = final_success[index]
    finally:
        world.close()

    audit = {
        'version': 3,
        'train': train_audits,
        'eval_source': str(eval_source.resolve()),
        'eval_source_sha256': sha256(eval_source),
        'eval_state_start': state_start,
        'eval_state_stop': state_stop,
        'train_eval_row_overlap': int(len(overlap)),
        'allow_train_eval_overlap': allow_train_eval_overlap,
        'operator_exclude_eval_rows': operator_exclude_eval_rows,
        'operator_config': asdict(operator_config),
        'k10_sparse_outcome': (
            str(k10_sparse_outcome.resolve())
            if k10_sparse_outcome is not None
            else None
        ),
        'train_epochs': train_epochs,
        'blend': blend,
        'diversity_threshold': diversity_threshold,
        'common_random_numbers_across_methods': True,
        'evaluate_simulator': evaluate_simulator,
        'evaluate_populations': evaluate_populations,
        'evaluate_final': evaluate_final,
        'methods': [asdict(method) for method in methods],
        'selector_names': selector_names,
        'max_state_mismatch': max_state_mismatch,
        'max_goal_mismatch': max_goal_mismatch,
        'max_roundtrip_error': max_roundtrip_error,
        'elapsed_seconds': time.time() - started,
    }
    atomic_savez(
        output,
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
        method_names=np.asarray(method_names),
        selector_names=np.asarray(selector_names),
        rows=result['rows'][:num_states],
        population_min_true=population_min_true,
        population_success=population_success,
        population_mean_true=population_mean_true,
        branch_count_history=branch_count_history,
        branch_distance_history=branch_distance_history,
        final_branch_true=final_branch_true,
        final_branch_success=final_branch_success,
        selected_true=selected_true,
        selected_success=selected_success,
        selected_index=selected_index,
        selected_modes=selected_modes,
        proposal_mean_history=proposal_mean_history,
        proposal_std_history=proposal_std_history,
        branch_log_score_history=branch_log_score_history,
        final_model_cost=final_model_cost,
        final_model_relative=final_model_relative,
    )
    report = {
        'audit': audit,
        'round': {
            method_names[method_i]: {
                'mean_min_true': population_min_true[
                    :, method_i
                ].mean(axis=0).tolist(),
                'coverage': population_success[
                    :, method_i
                ].mean(axis=0).tolist(),
            }
            for method_i in range(len(methods))
        },
        'final': {
            method_names[method_i]: {
                selector_names[selector_i]: {
                    'mean_true': float(
                        np.mean(
                            selected_true[
                                :, method_i, selector_i
                            ]
                        )
                    ),
                    'success': float(
                        np.mean(
                            selected_success[
                                :, method_i, selector_i
                            ]
                        )
                    ),
                }
                for selector_i in range(len(selector_names))
            }
            for method_i in range(len(methods))
        },
    }
    report_path = output.with_suffix('.json')
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report['final'], indent=2, sort_keys=True), flush=True)
    print(f'branch-preserving result -> {output}', flush=True)


if __name__ == '__main__':
    run()
