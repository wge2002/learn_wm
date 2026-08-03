"""Recursive CEM test for sparse-query discrete update correction.

Fixed-trace probes show that the simulator correction to a learned CEM mean
update is multi-modal: a small residual codebook has a strong held-out-state
oracle-routing ceiling, while a state-only router averages incompatible
directions.  This experiment asks the causal question that a fixed trace
cannot answer:

    Can a few queried true candidate outcomes select the right update mode,
    and does that advantage compound under recursive proposal re-sampling?

A codebook is fit on a disjoint population trace.  At every recursive round,
all branches use common Gaussian noise.  Sparse branches reveal true costs
only for ``m`` selected candidates, estimate an update from the best queried
fraction, and choose the nearest learned residual mode (with an explicit
no-op).  The full population is executed only to measure hidden performance.

The ordinary learned update and a full simulator-oracle update are included as
paired lower/upper references.  Sparse branches keep the learned update's
standard deviation and correct only its mean.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import time
import warnings

os.environ['MUJOCO_GL'] = 'egl'

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

import stable_worldmodel as swm

from candidate_oracle import (
    execute_candidate,
    make_process,
    prepare_model_info,
    prepare_world_info,
)
from cem_round_oracle import execute_population
from eval_wm import get_dataset, img_transform
from oe_fixed_trace_train import cache_state_embeddings, score_cached
from oe_sparse_query_mode_probe import (
    partial_oracle_update,
    query_indices,
)
from oe_update_corrector_probe import EPS, load_trace
from oe_update_mode_codebook_probe import spherical_kmeans
from oe_update_resample import (
    comma_ints,
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


def comma_list(value, *, name: str) -> list[str]:
    items = [item.strip() for item in str(value).split(',') if item.strip()]
    if not items:
        raise ValueError(f'{name} must contain at least one item')
    return items


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def routed_update(
    baseline: np.ndarray,
    estimate: np.ndarray,
    proposal_std: np.ndarray,
    prototypes: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    """Route a partial oracle estimate to a residual mode or no-op."""
    codebook = np.concatenate(
        [np.zeros((1, prototypes.shape[1])), prototypes],
        axis=0,
    )
    candidates = baseline[None] + codebook
    scale = proposal_std.reshape(-1)
    actual = candidates * scale[None]
    target = estimate * scale
    relative = (
        np.linalg.norm(actual - target[None], axis=1)
        / max(np.linalg.norm(target), EPS)
    )
    selected = int(np.argmin(relative))
    return candidates[selected], selected, float(relative[selected])


@dataclass(frozen=True)
class Branch:
    name: str
    budget: int
    query_interval: int = 1
    correction_scale: float = 1.0
    reuse_mode: bool = False
    query_schedule: str = 'periodic_end'
    query_allowance: int = 0
    query_seed_offset: int = 0
    tournament_modes: int = 0


def parse_branches(
    *,
    budgets_value,
    specs_value,
    strategy: str,
) -> list[Branch]:
    """Parse legacy budgets or explicit sparse-query schedule branches.

    Periodic entries use
    ``m<budget>:q<interval>:s<scale>:<query|reuse>``.  The ``q`` schedule
    queries at the end of each interval, matching the historical behavior.
    Replacing ``q`` with ``p`` queries at the start of each interval.

    Budgeted event entries use
    ``m<budget>:a<query-rounds>:s<scale>:drift``.  They divide the rollout
    into ``a`` windows, spend exactly one query in each window, and move that
    query earlier when the learned CEM update has drifted sufficiently from
    the update observed at the previous query.

    For example, ``m10:q5:s0.75:reuse`` queries 10 candidates every fifth
    round and reuses the most recently selected mode between query rounds.
    """
    specs = [
        item.strip()
        for item in str(specs_value).split(',')
        if item.strip()
    ]
    if not specs:
        budgets = comma_ints(budgets_value, name='oe.budgets')
        branches = []
        for budget in budgets:
            if budget == 0:
                branches.append(Branch(name='learned', budget=0))
            elif budget == -1:
                branches.append(Branch(name='oracle', budget=-1))
            elif budget > 0:
                branches.append(
                    Branch(
                        name=f'sparse_{strategy}_m{budget}',
                        budget=budget,
                    )
                )
            else:
                raise ValueError(
                    'oe.budgets entries must be -1 (oracle), 0 (learned), '
                    'or positive query counts'
                )
    else:
        periodic_pattern = re.compile(
            r'^m(?P<budget>\d+):q(?P<interval>\d+):'
            r's(?P<scale>\d+(?:\.\d+)?):'
            r'(?P<reuse>query|reuse)$'
        )
        phase_pattern = re.compile(
            r'^m(?P<budget>\d+):p(?P<interval>\d+):'
            r's(?P<scale>\d+(?:\.\d+)?)'
            r'(?::r(?P<repeat>\d+))?:reuse$'
        )
        drift_pattern = re.compile(
            r'^m(?P<budget>\d+):a(?P<allowance>\d+):'
            r's(?P<scale>\d+(?:\.\d+)?):drift$'
        )
        tournament_pattern = re.compile(
            r'^t(?P<modes>\d+):p(?P<interval>\d+):'
            r's(?P<scale>\d+(?:\.\d+)?)$'
        )
        branches = []
        for spec in specs:
            if spec in ('learned', 'oracle'):
                branches.append(
                    Branch(
                        name=spec,
                        budget=0 if spec == 'learned' else -1,
                    )
                )
                continue
            periodic_match = periodic_pattern.fullmatch(spec)
            phase_match = phase_pattern.fullmatch(spec)
            drift_match = drift_pattern.fullmatch(spec)
            tournament_match = tournament_pattern.fullmatch(spec)
            match = (
                periodic_match
                or phase_match
                or drift_match
                or tournament_match
            )
            if match is None:
                raise ValueError(
                    f'invalid oe.branch_specs entry {spec!r}; expected '
                    'learned, oracle, m10:q5:s0.75:reuse, '
                    'm10:p5:s0.75:reuse, m10:a5:s0.75:drift, '
                    'or t8:p5:s0.75'
                )
            budget = (
                int(match.group('budget'))
                if tournament_match is None
                else 0
            )
            scale = float(match.group('scale'))
            scale_text = f'{scale:g}'.replace('.', 'p')
            if tournament_match is not None:
                modes = int(match.group('modes'))
                interval = int(match.group('interval'))
                branches.append(
                    Branch(
                        name=(
                            f'tournament_top{modes}_p{interval}_'
                            f's{scale_text}'
                        ),
                        budget=0,
                        query_interval=interval,
                        correction_scale=scale,
                        reuse_mode=True,
                        query_schedule='periodic_start',
                        tournament_modes=modes,
                    )
                )
                continue
            if drift_match is not None:
                allowance = int(match.group('allowance'))
                branches.append(
                    Branch(
                        name=(
                            f'sparse_{strategy}_m{budget}_a{allowance}_'
                            f's{scale_text}_drift'
                        ),
                        budget=budget,
                        correction_scale=scale,
                        reuse_mode=True,
                        query_schedule='drift_window',
                        query_allowance=allowance,
                    )
                )
                continue
            interval = int(match.group('interval'))
            if phase_match is not None:
                repeat_text = match.group('repeat')
                repeat = int(repeat_text) if repeat_text is not None else 0
                repeat_suffix = (
                    f'_r{repeat}' if repeat_text is not None else ''
                )
                branches.append(
                    Branch(
                        name=(
                            f'sparse_{strategy}_m{budget}_p{interval}_'
                            f's{scale_text}{repeat_suffix}_reuse'
                        ),
                        budget=budget,
                        query_interval=interval,
                        correction_scale=scale,
                        reuse_mode=True,
                        query_schedule='periodic_start',
                        query_seed_offset=repeat,
                    )
                )
                continue
            reuse = match.group('reuse') == 'reuse'
            suffix = 'reuse' if reuse else 'query'
            branches.append(
                Branch(
                    name=(
                        f'sparse_{strategy}_m{budget}_q{interval}_'
                        f's{scale_text}_{suffix}'
                    ),
                    budget=budget,
                    query_interval=interval,
                    correction_scale=scale,
                    reuse_mode=reuse,
                    query_schedule='periodic_end',
                )
            )
    names = [branch.name for branch in branches]
    if not names:
        raise ValueError('at least one rollout branch is required')
    if len(set(names)) != len(names):
        raise ValueError('branch specification produces duplicate names')
    return branches


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig) -> None:
    oe = cfg.get('oe', {})
    source = Path(str(oe.get('source', '')))
    codebook_source = Path(str(oe.get('codebook_source', '')))
    out = Path(str(oe.get('out', '')))
    if not source.exists():
        raise FileNotFoundError(f'oe.source does not exist: {source}')
    if not codebook_source.exists():
        raise FileNotFoundError(
            f'oe.codebook_source does not exist: {codebook_source}'
        )
    if out == Path('.'):
        raise ValueError('oe.out is required')

    source_generator_name = str(
        oe.get('source_generator', 'pd_d192_k3_eval')
    )
    scorer_names = comma_list(
        oe.get(
            'scorers',
            'pd_d192_k3_eval,iter2_multistep_eval,pd_d192_k10_eval',
        ),
        name='oe.scorers',
    )
    if scorer_names[0] != source_generator_name:
        raise ValueError(
            'the first scorer must equal oe.source_generator so the learned '
            'reference matches the source distribution'
        )
    start_step = int(oe.get('start_step', 4))
    num_rounds = int(oe.get('num_rounds', 25))
    num_samples = int(oe.get('num_samples', 100))
    num_states = int(oe.get('num_states', -1))
    topk = int(oe.get('topk', 30))
    codebook_topk = int(oe.get('codebook_topk', 90))
    clusters = int(oe.get('clusters', 16))
    codebook_seed = int(oe.get('codebook_seed', 20260730))
    query_strategy = str(oe.get('query_strategy', 'random'))
    branches = parse_branches(
        budgets_value=oe.get('budgets', '0,10,30,-1'),
        specs_value=oe.get('branch_specs', ''),
        strategy=query_strategy,
    )
    budgets = [branch.budget for branch in branches]
    names = [branch.name for branch in branches]
    query_intervals = [
        branch.query_interval for branch in branches
    ]
    correction_scales = [
        branch.correction_scale for branch in branches
    ]
    reuse_modes = [branch.reuse_mode for branch in branches]
    query_schedules = [branch.query_schedule for branch in branches]
    query_allowances = [branch.query_allowance for branch in branches]
    query_seed_offsets = [
        branch.query_seed_offset for branch in branches
    ]
    tournament_modes = [
        branch.tournament_modes for branch in branches
    ]
    drift_threshold = float(oe.get('drift_threshold', 0.9))
    std_floor = float(oe.get('std_floor', 1e-4))
    resume = bool(oe.get('resume', True))

    if num_rounds < 1:
        raise ValueError('oe.num_rounds must be positive')
    if num_samples < 2:
        raise ValueError('oe.num_samples must be at least two')
    if not 2 <= topk < num_samples:
        raise ValueError(
            f'oe.topk must be in [2, {num_samples - 1}]'
        )
    if clusters < 1:
        raise ValueError('oe.clusters must be positive')
    if std_floor <= 0:
        raise ValueError('oe.std_floor must be positive')
    for branch in branches:
        budget = branch.budget
        if budget > num_samples:
            raise ValueError(
                f'query budget {budget} exceeds N={num_samples}'
            )
        if budget > 0 and round(budget * topk / num_samples) < 2:
            raise ValueError(
                f'query budget {budget} selects fewer than two partial '
                'oracle elites'
            )
        if branch.query_interval < 1:
            raise ValueError('query intervals must be positive')
        if branch.query_schedule == 'drift_window':
            if not 1 <= branch.query_allowance <= num_rounds:
                raise ValueError(
                    'drift query allowance must be in '
                    f'[1, {num_rounds}]'
                )
        if branch.query_seed_offset < 0:
            raise ValueError('query seed offsets must be non-negative')
        if not 0 <= branch.tournament_modes <= clusters:
            raise ValueError(
                'tournament mode counts must be inside '
                f'[0, {clusters}]'
            )
        if not 0.0 <= branch.correction_scale <= 2.0:
            raise ValueError(
                'correction scales must be inside [0, 2]'
            )
    if not 0.0 <= drift_threshold <= 2.0:
        raise ValueError('oe.drift_threshold must be inside [0, 2]')

    with np.load(source, allow_pickle=False) as archive:
        result = {key: np.asarray(archive[key]) for key in archive.files}
    validate_source(result, cfg=cfg)
    for key in ('mean', 'var'):
        if key not in result:
            raise ValueError(f'source is missing {key!r}')

    generators = result['generators'].astype(str).tolist()
    if source_generator_name not in generators:
        raise ValueError(
            f'oe.source_generator={source_generator_name!r} '
            f'not in {generators}'
        )
    generator_i = generators.index(source_generator_name)
    source_steps = result['steps'].astype(int).tolist()
    if start_step not in source_steps:
        raise ValueError(
            f'oe.start_step={start_step} is not present in {source_steps}'
        )
    source_round_i = source_steps.index(start_step)

    total_states = result['candidates'].shape[0]
    if num_states < 0:
        num_states = total_states
    if not 1 <= num_states <= total_states:
        raise ValueError(f'oe.num_states must be in [1, {total_states}] or -1')
    state_indices = np.arange(num_states, dtype=np.int64)
    action_shape = tuple(result['candidates'].shape[-2:])

    codebook_trace = load_trace(
        codebook_source,
        topk=codebook_topk,
    )
    if codebook_trace.horizon != int(result['horizon']):
        raise ValueError('codebook/source horizon mismatch')
    if codebook_trace.goal_offset != int(result['goal_offset']):
        raise ValueError('codebook/source goal-offset mismatch')
    with np.load(codebook_source, allow_pickle=False) as archive:
        codebook_n = int(archive['candidates'].shape[3])
        codebook_rows = np.asarray(archive['rows']).astype(np.int64)
    elite_fraction = topk / num_samples
    codebook_elite_fraction = codebook_topk / codebook_n
    if not np.isclose(elite_fraction, codebook_elite_fraction, atol=1e-12):
        raise ValueError(
            'target/codebook elite fractions must match: '
            f'{topk}/{num_samples}={elite_fraction:.6f} vs '
            f'{codebook_topk}/{codebook_n}='
            f'{codebook_elite_fraction:.6f}'
        )
    overlap_rows = np.intersect1d(
        result['rows'][state_indices].astype(np.int64),
        codebook_rows,
    )
    if len(overlap_rows):
        raise ValueError(
            f'codebook/evaluation traces overlap on {len(overlap_rows)} rows'
        )

    residual = (
        codebook_trace.oracle_update_normalized
        - codebook_trace.model_update_normalized
    )
    prototypes, codebook_labels = spherical_kmeans(
        residual,
        clusters=clusters,
        seed=codebook_seed,
    )
    cluster_sizes = np.bincount(
        codebook_labels,
        minlength=clusters,
    ).astype(np.int64)
    frequent_mode_ids = (
        np.argsort(-cluster_sizes, kind='stable') + 1
    ).astype(np.int64)

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

    device = torch.device('cuda')
    models = {}
    for scorer_name in scorer_names:
        model = swm.wm.utils.load_pretrained(scorer_name).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        models[scorer_name] = model
    solver = hydra.utils.instantiate(
        cfg.solver,
        model=models[scorer_names[0]],
    )
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

    num_branches = len(branches)
    num_scorers = len(scorer_names)
    population_shape = (
        len(state_indices),
        num_branches,
        num_rounds,
        num_samples,
    )
    population_true = np.full(population_shape, np.nan, dtype=np.float64)
    population_success = np.zeros(population_shape, dtype=bool)
    population_pred = np.full(
        (
            len(state_indices),
            num_branches,
            num_rounds,
            num_scorers,
            num_samples,
        ),
        np.nan,
        dtype=np.float32,
    )
    query_mask = np.zeros(population_shape, dtype=bool)
    mean_history = np.full(
        (
            len(state_indices),
            num_branches,
            num_rounds + 1,
            *action_shape,
        ),
        np.nan,
        dtype=np.float32,
    )
    std_history = np.full_like(mean_history, np.nan)
    mean_true = np.full(
        population_shape[:3],
        np.nan,
        dtype=np.float64,
    )
    mean_success = np.zeros(population_shape[:3], dtype=bool)
    selected_mode = np.full(
        population_shape[:3],
        -99,
        dtype=np.int16,
    )
    mode_was_queried = np.zeros(
        population_shape[:3],
        dtype=bool,
    )
    routing_relative_error = np.full(
        population_shape[:3],
        np.nan,
        dtype=np.float32,
    )
    trigger_drift = np.full(
        population_shape[:3],
        np.nan,
        dtype=np.float32,
    )
    auxiliary_query_count = np.zeros(
        population_shape[:3],
        dtype=np.int16,
    )
    tournament_true = np.full(
        (*population_shape[:3], clusters + 1),
        np.nan,
        dtype=np.float64,
    )
    tournament_success = np.zeros(
        (*population_shape[:3], clusters + 1),
        dtype=bool,
    )
    final_mean_true = np.full(
        population_shape[:2],
        np.nan,
        dtype=np.float64,
    )
    final_mean_success = np.zeros(population_shape[:2], dtype=bool)
    completed_states = np.zeros(len(state_indices), dtype=bool)

    source_hash = sha256(source)
    codebook_hash = sha256(codebook_source)
    previous_elapsed = 0.0
    resumable_arrays = (
        'population_true',
        'population_success',
        'population_pred',
        'query_mask',
        'mean_history',
        'std_history',
        'mean_true',
        'mean_success',
        'selected_mode',
        'mode_was_queried',
        'routing_relative_error',
        'trigger_drift',
        'auxiliary_query_count',
        'tournament_true',
        'tournament_success',
        'final_mean_true',
        'final_mean_success',
        'completed_states',
    )
    if out.exists():
        if not resume:
            raise FileExistsError(
                f'output exists: {out}; set oe.resume=true or use a new path'
            )
        with np.load(out, allow_pickle=False) as archive:
            expected = {
                'source_sha256': source_hash,
                'codebook_source_sha256': codebook_hash,
            }
            for key, value in expected.items():
                actual = str(np.asarray(archive[key]).item())
                if actual != value:
                    raise ValueError(
                        f'resume mismatch for {key}: {actual} != {value}'
                    )
            if not np.array_equal(archive['budgets'], budgets):
                raise ValueError('resume budget mismatch')
            if not np.array_equal(archive['branch_names'], names):
                raise ValueError('resume branch-name mismatch')
            if not np.array_equal(
                archive['query_intervals'],
                query_intervals,
            ):
                raise ValueError('resume query-interval mismatch')
            if not np.allclose(
                archive['correction_scales'],
                correction_scales,
            ):
                raise ValueError('resume correction-scale mismatch')
            if not np.array_equal(
                archive['reuse_modes'],
                reuse_modes,
            ):
                raise ValueError('resume mode-reuse mismatch')
            if not np.array_equal(
                archive['query_schedules'],
                query_schedules,
            ):
                raise ValueError('resume query-schedule mismatch')
            if not np.array_equal(
                archive['query_allowances'],
                query_allowances,
            ):
                raise ValueError('resume query-allowance mismatch')
            if not np.array_equal(
                archive['query_seed_offsets'],
                query_seed_offsets,
            ):
                raise ValueError('resume query-seed-offset mismatch')
            if not np.array_equal(
                archive['tournament_modes'],
                tournament_modes,
            ):
                raise ValueError('resume tournament-mode mismatch')
            if not np.isclose(
                float(archive['drift_threshold']),
                drift_threshold,
            ):
                raise ValueError('resume drift-threshold mismatch')
            if not np.array_equal(
                archive['state_indices'],
                state_indices,
            ):
                raise ValueError('resume state-index mismatch')
            if int(archive['num_rounds']) != num_rounds:
                raise ValueError('resume round-count mismatch')
            if int(archive['num_samples']) != num_samples:
                raise ValueError('resume sample-count mismatch')
            if int(archive['clusters']) != clusters:
                raise ValueError('resume cluster-count mismatch')
            local_arrays = locals()
            for key in resumable_arrays:
                target = local_arrays[key]
                stored = np.asarray(archive[key])
                if target.shape != stored.shape:
                    raise ValueError(
                        f'resume shape mismatch for {key}: '
                        f'{stored.shape} != {target.shape}'
                    )
                target[...] = stored
            previous_elapsed = float(archive['elapsed_seconds'])
        if np.all(completed_states):
            print(f'complete result already exists -> {out}', flush=True)
            world.close()
            return
        print(
            f'resuming {int(completed_states.sum())}/'
            f'{len(completed_states)} completed states',
            flush=True,
        )

    max_state_mismatch = 0.0
    max_goal_mismatch = 0.0
    max_candidate_quantization_error = 0.0
    max_roundtrip_error = 0.0
    started = time.time()

    def output_arrays(*, complete: bool) -> dict[str, np.ndarray]:
        return {
            'version': np.asarray(5),
            'complete': np.asarray(complete),
            'source': np.asarray(str(source)),
            'source_sha256': np.asarray(source_hash),
            'codebook_source': np.asarray(str(codebook_source)),
            'codebook_source_sha256': np.asarray(codebook_hash),
            'generator': np.asarray(source_generator_name),
            'scorers': np.asarray(scorer_names),
            'query_strategy': np.asarray(query_strategy),
            'budgets': np.asarray(budgets, dtype=np.int64),
            'branch_names': np.asarray(names),
            'query_intervals': np.asarray(
                query_intervals,
                dtype=np.int64,
            ),
            'correction_scales': np.asarray(
                correction_scales,
                dtype=np.float64,
            ),
            'reuse_modes': np.asarray(reuse_modes, dtype=bool),
            'query_schedules': np.asarray(query_schedules),
            'query_allowances': np.asarray(
                query_allowances,
                dtype=np.int64,
            ),
            'query_seed_offsets': np.asarray(
                query_seed_offsets,
                dtype=np.int64,
            ),
            'tournament_modes': np.asarray(
                tournament_modes,
                dtype=np.int64,
            ),
            'drift_threshold': np.asarray(drift_threshold),
            'state_indices': state_indices,
            'rows': result['rows'][state_indices],
            'episodes': result['episodes'][state_indices],
            'starts': result['starts'][state_indices],
            'start_step': np.asarray(start_step),
            'num_rounds': np.asarray(num_rounds),
            'num_samples': np.asarray(num_samples),
            'topk': np.asarray(topk),
            'elite_fraction': np.asarray(elite_fraction),
            'codebook_topk': np.asarray(codebook_topk),
            'clusters': np.asarray(clusters),
            'codebook_seed': np.asarray(codebook_seed),
            'codebook_prototypes': prototypes.astype(np.float32),
            'codebook_cluster_sizes': cluster_sizes,
            'std_floor': np.asarray(std_floor),
            'population_true': population_true,
            'population_success': population_success,
            'population_pred': population_pred,
            'query_mask': query_mask,
            'mean_history': mean_history,
            'std_history': std_history,
            'mean_true': mean_true,
            'mean_success': mean_success,
            'selected_mode': selected_mode,
            'mode_was_queried': mode_was_queried,
            'routing_relative_error': routing_relative_error,
            'trigger_drift': trigger_drift,
            'auxiliary_query_count': auxiliary_query_count,
            'tournament_true': tournament_true,
            'tournament_success': tournament_success,
            'final_mean_true': final_mean_true,
            'final_mean_success': final_mean_success,
            'completed_states': completed_states,
            'max_state_mismatch': np.asarray(max_state_mismatch),
            'max_goal_mismatch': np.asarray(max_goal_mismatch),
            'candidate_storage_dtype': np.asarray('float16'),
            'max_candidate_quantization_error': np.asarray(
                max_candidate_quantization_error
            ),
            'max_roundtrip_error': np.asarray(max_roundtrip_error),
            'elapsed_seconds': np.asarray(
                previous_elapsed + time.time() - started
            ),
            'config': np.asarray(
                json.dumps(
                    OmegaConf.to_container(cfg, resolve=True),
                    sort_keys=True,
                )
            ),
        }

    try:
        for output_state_i, state_i in enumerate(state_indices):
            if completed_states[output_state_i]:
                continue
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
                raise RuntimeError(
                    f'trace reconstruction mismatch at state {state_i}: '
                    f'state={state_mismatch:.3e}, goal={goal_mismatch:.3e}'
                )

            model_info = prepare_model_info(policy, info)
            caches = {
                scorer_name: cache_state_embeddings(
                    models[scorer_name],
                    model_info,
                    action_shape=action_shape,
                )
                for scorer_name in scorer_names
            }
            execution_cache: dict[bytes, dict] = {}
            initial_mean = result['mean'][
                state_i,
                generator_i,
                source_round_i,
            ].astype(np.float32)
            # Despite its historical name, CEM ``var`` is the standard
            # deviation used directly in ``noise * var + mean``.
            initial_std = np.maximum(
                result['var'][
                    state_i,
                    generator_i,
                    source_round_i,
                ].astype(np.float32),
                np.float32(std_floor),
            )
            branch_mean = np.repeat(
                initial_mean[None],
                num_branches,
                axis=0,
            )
            branch_std = np.repeat(
                initial_std[None],
                num_branches,
                axis=0,
            )
            mean_history[output_state_i, :, 0] = branch_mean
            std_history[output_state_i, :, 0] = branch_std
            last_selected_mode = np.zeros(
                num_branches,
                dtype=np.int64,
            )
            last_query_learned_update = np.full(
                (num_branches, int(np.prod(action_shape))),
                np.nan,
                dtype=np.float64,
            )
            last_queried_window = np.full(
                num_branches,
                -1,
                dtype=np.int64,
            )

            for branch_round in range(num_rounds):
                noise_rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [
                            int(cfg.seed),
                            int(state_i),
                            int(start_step),
                            int(branch_round),
                        ]
                    )
                )
                noise = noise_rng.standard_normal(
                    (num_samples, *action_shape),
                    dtype=np.float32,
                )
                round_text = []
                for branch_i, branch in enumerate(branches):
                    budget = branch.budget
                    current_mean = branch_mean[branch_i]
                    current_std = branch_std[branch_i]
                    raw_candidates = (
                        noise * current_std[None] + current_mean[None]
                    )
                    raw_candidates[0] = current_mean
                    candidates = quantize_candidates(raw_candidates)
                    max_candidate_quantization_error = max(
                        max_candidate_quantization_error,
                        float(
                            np.max(
                                np.abs(candidates - raw_candidates)
                            )
                        ),
                    )

                    predicted = []
                    for scorer_name in scorer_names:
                        model = models[scorer_name]
                        dtype = next(model.parameters()).dtype
                        candidates_t = torch.as_tensor(
                            candidates,
                            device=device,
                            dtype=dtype,
                        )
                        values = (
                            score_cached(
                                model,
                                caches[scorer_name],
                                candidates_t,
                            )
                            .cpu()
                            .numpy()
                        )
                        predicted.append(values)
                    predicted = np.asarray(predicted, dtype=np.float32)

                    execution = execute_population(
                        world.envs.envs[0],
                        candidates=candidates,
                        initial_state=initial_state,
                        goal_state=goal_state,
                        action_scaler=process['action'],
                        action_block=int(result['action_block']),
                        seed=int(cfg.seed) + int(state_i),
                        cache=execution_cache,
                    )
                    learned_mean, learned_std, _ = elite_moments(
                        candidates,
                        predicted[0],
                        topk=topk,
                        std_floor=std_floor,
                    )
                    oracle_mean, oracle_std, _ = elite_moments(
                        candidates,
                        execution['true'],
                        topk=topk,
                        std_floor=std_floor,
                    )
                    learned_update = (
                        (learned_mean - current_mean) / current_std
                    ).reshape(-1)

                    index = (
                        output_state_i,
                        branch_i,
                        branch_round,
                    )
                    if branch.tournament_modes > 0:
                        should_query = (
                            branch_round % branch.query_interval == 0
                        )
                        if should_query:
                            mode_ids = np.concatenate(
                                [
                                    np.zeros(1, dtype=np.int64),
                                    frequent_mode_ids[
                                        : branch.tournament_modes
                                    ],
                                ]
                            )
                            mode_updates = np.repeat(
                                learned_update[None],
                                len(mode_ids),
                                axis=0,
                            )
                            nonzero = mode_ids > 0
                            mode_updates[nonzero] += prototypes[
                                mode_ids[nonzero] - 1
                            ]
                            mode_updates = (
                                learned_update[None]
                                + branch.correction_scale
                                * (
                                    mode_updates
                                    - learned_update[None]
                                )
                            )
                            mode_means = (
                                current_mean[None]
                                + mode_updates.reshape(
                                    -1,
                                    *action_shape,
                                )
                                * current_std[None]
                            ).astype(np.float32)
                            mode_execution = execute_population(
                                world.envs.envs[0],
                                candidates=mode_means,
                                initial_state=initial_state,
                                goal_state=goal_state,
                                action_scaler=process['action'],
                                action_block=int(
                                    result['action_block']
                                ),
                                seed=int(cfg.seed) + int(state_i),
                                cache=execution_cache,
                            )
                            tournament_true[index][mode_ids] = (
                                mode_execution['true']
                            )
                            tournament_success[index][mode_ids] = (
                                mode_execution['success']
                            )
                            successful = np.flatnonzero(
                                mode_execution['success']
                            )
                            if len(successful):
                                selected_position = int(
                                    successful[
                                        np.argmin(
                                            mode_execution['true'][
                                                successful
                                            ]
                                        )
                                    ]
                                )
                            else:
                                selected_position = int(
                                    np.argmin(mode_execution['true'])
                                )
                            mode = int(mode_ids[selected_position])
                            last_selected_mode[branch_i] = mode
                            mode_was_queried[index] = True
                            auxiliary_query_count[index] = len(mode_ids)
                            max_roundtrip_error = max(
                                max_roundtrip_error,
                                float(
                                    np.max(
                                        mode_execution[
                                            'roundtrip_error'
                                        ]
                                    )
                                ),
                            )
                        elif last_selected_mode[branch_i] > 0:
                            mode = int(last_selected_mode[branch_i])
                        else:
                            mode = 0
                        corrected = learned_update
                        if mode > 0:
                            corrected = (
                                learned_update + prototypes[mode - 1]
                            )
                        corrected = (
                            learned_update
                            + branch.correction_scale
                            * (corrected - learned_update)
                        )
                        next_mean = (
                            current_mean
                            + corrected.reshape(action_shape) * current_std
                        ).astype(np.float32)
                        next_std = learned_std
                        selected_mode[index] = mode
                    elif budget == 0:
                        next_mean = learned_mean
                        next_std = learned_std
                        selected_mode[index] = -1
                    elif budget == -1:
                        next_mean = oracle_mean
                        next_std = oracle_std
                        selected_mode[index] = -2
                        query_mask[index] = True
                    else:
                        if branch.query_schedule == 'periodic_end':
                            should_query = (
                                (branch_round + 1)
                                % branch.query_interval
                                == 0
                            )
                        elif branch.query_schedule == 'periodic_start':
                            should_query = (
                                branch_round % branch.query_interval == 0
                            )
                        elif branch.query_schedule == 'drift_window':
                            allowance = branch.query_allowance
                            window = min(
                                branch_round * allowance // num_rounds,
                                allowance - 1,
                            )
                            window_end = (
                                (
                                    (window + 1) * num_rounds
                                    + allowance
                                    - 1
                                )
                                // allowance
                                - 1
                            )
                            physical_update = (
                                learned_mean - current_mean
                            ).reshape(-1).astype(np.float64)
                            anchor = last_query_learned_update[branch_i]
                            if np.all(np.isfinite(anchor)):
                                denominator = max(
                                    float(
                                        np.linalg.norm(physical_update)
                                        * np.linalg.norm(anchor)
                                    ),
                                    EPS,
                                )
                                drift = 1.0 - float(
                                    np.dot(physical_update, anchor)
                                ) / denominator
                                trigger_drift[index] = drift
                            else:
                                drift = np.inf
                            window_available = (
                                last_queried_window[branch_i] != window
                            )
                            should_query = window_available and (
                                branch_round == 0
                                or drift >= drift_threshold
                                or branch_round >= window_end
                            )
                        else:
                            raise AssertionError(
                                'unknown query schedule '
                                f'{branch.query_schedule!r}'
                            )
                        if should_query:
                            query_rng = np.random.default_rng(
                                int(cfg.seed)
                                + 10_000_000 * int(state_i)
                                + 10_000 * int(branch_round)
                                + 10 * int(budget)
                                + 1_000_000_000
                                * int(branch.query_seed_offset)
                            )
                            query = query_indices(
                                predicted,
                                budget=budget,
                                strategy=query_strategy,
                                rng=query_rng,
                            )
                            query_mask[index][query] = True
                            estimate = partial_oracle_update(
                                candidates,
                                execution['true'],
                                query,
                                prev_mean=current_mean,
                                proposal_std=current_std,
                                elite_fraction=elite_fraction,
                            )
                            corrected, mode, route_error = routed_update(
                                learned_update,
                                estimate,
                                current_std,
                                prototypes,
                            )
                            last_selected_mode[branch_i] = mode
                            mode_was_queried[index] = True
                            routing_relative_error[index] = route_error
                            if branch.query_schedule == 'drift_window':
                                last_query_learned_update[branch_i] = (
                                    physical_update
                                )
                                last_queried_window[branch_i] = window
                        elif (
                            branch.reuse_mode
                            and last_selected_mode[branch_i] > 0
                        ):
                            mode = int(last_selected_mode[branch_i])
                            corrected = (
                                learned_update + prototypes[mode - 1]
                            )
                        else:
                            mode = 0
                            corrected = learned_update
                        corrected = (
                            learned_update
                            + branch.correction_scale
                            * (corrected - learned_update)
                        )
                        next_mean = (
                            current_mean
                            + corrected.reshape(action_shape) * current_std
                        ).astype(np.float32)
                        next_std = learned_std
                        selected_mode[index] = mode

                    population_true[index] = execution['true']
                    population_success[index] = execution['success']
                    population_pred[index] = predicted
                    mean_true[index] = execution['true'][0]
                    mean_success[index] = execution['success'][0]
                    mean_history[
                        output_state_i,
                        branch_i,
                        branch_round + 1,
                    ] = next_mean
                    std_history[
                        output_state_i,
                        branch_i,
                        branch_round + 1,
                    ] = next_std
                    branch_mean[branch_i] = next_mean
                    branch_std[branch_i] = next_std
                    max_roundtrip_error = max(
                        max_roundtrip_error,
                        float(np.max(execution['roundtrip_error'])),
                    )
                    round_text.append(
                        f'{names[branch_i]}='
                        f'{execution["true"].min():.1f}/'
                        f'{int(execution["success"].any())}'
                    )

                print(
                    f'[{output_state_i + 1}/{len(state_indices)}] '
                    f'round={branch_round + 1}/{num_rounds} '
                    + ' '.join(round_text)
                    + f' elapsed={(time.time() - started) / 60:.1f}m',
                    flush=True,
                )

            for branch_i in range(num_branches):
                final_execution = execute_candidate(
                    world.envs.envs[0],
                    initial_state=initial_state,
                    goal_state=goal_state,
                    candidate=branch_mean[branch_i],
                    action_scaler=process['action'],
                    action_block=int(result['action_block']),
                    seed=int(cfg.seed) + int(state_i),
                )
                final_mean_true[output_state_i, branch_i] = final_execution[
                    'cost'
                ]
                final_mean_success[
                    output_state_i,
                    branch_i,
                ] = final_execution['success']
                max_roundtrip_error = max(
                    max_roundtrip_error,
                    float(final_execution['roundtrip_error']),
                )
            completed_states[output_state_i] = True
            atomic_savez(
                out,
                output_arrays(complete=bool(np.all(completed_states))),
            )
            print(
                f'checkpoint -> {out} '
                f'({int(completed_states.sum())}/{len(completed_states)})',
                flush=True,
            )
    finally:
        world.close()

    print('\nSparse-mode recursive intervention', flush=True)
    print(
        'branch average_coverage last_coverage last_min_true '
        'final_mean_true final_mean_success true_queries',
        flush=True,
    )
    for branch_i, name in enumerate(names):
        coverage = np.any(
            population_success[:, branch_i],
            axis=-1,
        )
        min_true = np.min(population_true[:, branch_i], axis=-1)
        true_queries = (
            query_mask[:, branch_i].sum(axis=(1, 2))
            + auxiliary_query_count[:, branch_i].sum(axis=1)
        )
        print(
            f'{name} '
            f'{coverage.mean():.3f} '
            f'{coverage[:, -1].mean():.3f} '
            f'{min_true[:, -1].mean():.2f} '
            f'{final_mean_true[:, branch_i].mean():.2f} '
            f'{final_mean_success[:, branch_i].mean():.3f} '
            f'{true_queries.mean():.1f}',
            flush=True,
        )
    print(
        f'\nstate_mismatch={max_state_mismatch:.3e} '
        f'goal_mismatch={max_goal_mismatch:.3e} '
        f'quantization={max_candidate_quantization_error:.3e} '
        f'roundtrip={max_roundtrip_error:.3e}',
        flush=True,
    )
    print(f'results -> {out}', flush=True)
    print(
        f'elapsed={(previous_elapsed + time.time() - started) / 60:.1f} '
        'minutes',
        flush=True,
    )


if __name__ == '__main__':
    run()
