"""Script to evaluate a World Model using MPC on a dataset of episodes."""

import os

os.environ['MUJOCO_GL'] = 'egl'

import json
import time
import warnings
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm


warnings.filterwarnings(
    'ignore',
    message='.*Casting input x to numpy array.*',
    category=UserWarning,
    module='gymnasium.spaces.box',
)
warnings.filterwarnings(
    'ignore',
    message=".*Box (low|high)'s precision lowered.*",
    category=UserWarning,
    module='gymnasium.spaces.box',
)


def configure_torch_threads_from_env():
    raw = os.environ.get('SWM_TORCH_THREADS')
    if not raw:
        return
    try:
        threads = max(1, int(raw))
    except ValueError:
        print(f'[eval] ignoring invalid SWM_TORCH_THREADS={raw!r}')
        return
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:
        # Inter-op threads can only be set before any parallel work starts.
        pass
    print(f'[eval] torch CPU threads set to {threads}')


def img_transform(cfg, dtype=torch.float32):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(dtype, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data('step_idx')
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset = swm.data.load_dataset(
        dataset_name,
        cache_dir=cfg.get('cache_dir', None),
        keys_to_cache=list(cfg.dataset.keys_to_cache),
    )
    return dataset


def build_dataset_initial_action_prior(
    dataset,
    episodes: np.ndarray,
    starts: np.ndarray,
    *,
    horizon: int,
    action_block: int,
    action_scaler,
    alignment: str = 'next',
) -> np.ndarray:
    """Build a hidden-oracle one-shot CEM initialization from dataset actions.

    This is a support intervention, not a deployable policy input.  ``next``
    matches datasets whose reset row has a NaN action and uses the actions on
    the rows immediately following the seeded state.
    """
    if alignment not in {'next', 'same'}:
        raise ValueError("dataset_action_prior.alignment must be 'next' or 'same'")
    num_actions = int(horizon) * int(action_block)
    chunks = dataset.load_chunk(
        np.asarray(episodes),
        np.asarray(starts),
        np.asarray(starts) + num_actions + 1,
    )
    plans = []
    for episode, start, chunk in zip(
        episodes, starts, chunks, strict=True
    ):
        raw = np.asarray(chunk['action'], dtype=np.float32)
        offset = 1 if alignment == 'next' else 0
        selected = raw[offset : offset + num_actions]
        if len(selected) != num_actions:
            raise ValueError(
                f'Action prior for episode={episode}, start={start} needs '
                f'{num_actions} actions, got {len(selected)}'
            )
        normalized = action_scaler.transform(
            np.nan_to_num(selected)
        ).astype(np.float32, copy=False)
        plans.append(normalized.reshape(horizon, -1))
    return np.stack(plans)


def to_container_or_none(value):
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


@hydra.main(version_base=None, config_path='./config', config_name='pusht')
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    configure_torch_threads_from_env()

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), 'Planning horizon must be smaller than or equal to eval_budget'

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    img_dtype = torch.bfloat16 if cfg.get('bf16', False) else torch.float32
    transform = {
        'pixels': img_transform(cfg, img_dtype),
        'pixels_hist': img_transform(cfg, img_dtype),
        'goal': img_transform(cfg, img_dtype),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    ep_indices, _ = np.unique(
        stats_dataset.get_col_data(col_name), return_index=True
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ['pixels']:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != 'action':
            process[f'goal_{col}'] = process[col]

    if 'action' in process:
        process['action_hist'] = process['action']

    # -- run evaluation
    policy = cfg.get('policy', 'random')
    solver = None

    if policy != 'random':
        model = swm.wm.utils.load_pretrained(cfg.policy)
        if cfg.get('bf16', False):
            model = model.to(torch.bfloat16)
        model = model.to('cuda')
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        if cfg.get('compile', False):
            encoder_attr = (
                'backbone' if hasattr(model, 'backbone') else 'encoder'
            )
            setattr(
                model,
                encoder_attr,
                torch.compile(getattr(model, encoder_attr)),
            )
            model.predictor = torch.compile(model.predictor)
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        cross_validate = cfg.get('cross_validate')
        portfolio = cfg.get('portfolio')
        score_ensemble = cfg.get('score_ensemble')
        enabled_extensions = sum(
            extension is not None
            for extension in (
                cross_validate,
                portfolio,
                score_ensemble,
            )
        )
        if enabled_extensions > 1:
            raise ValueError(
                'cross_validate, portfolio, and score_ensemble are '
                'mutually exclusive'
            )
        if score_ensemble is not None:
            ensemble_names = [str(name) for name in score_ensemble.models]
            if ensemble_names[0] != str(cfg.policy):
                raise ValueError(
                    'score_ensemble.models must start with cfg.policy'
                )
            labels = [
                str(name)
                for name in score_ensemble.get(
                    'labels',
                    ensemble_names,
                )
            ]
            if len(labels) != len(ensemble_names):
                raise ValueError(
                    'score_ensemble.labels must match score_ensemble.models'
                )
            ensemble_models = [model]
            for ensemble_name in ensemble_names[1:]:
                ensemble_model = swm.wm.utils.load_pretrained(ensemble_name)
                if cfg.get('bf16', False):
                    ensemble_model = ensemble_model.to(torch.bfloat16)
                ensemble_model = ensemble_model.to('cuda').eval()
                ensemble_model.requires_grad_(False)
                ensemble_model.interpolate_pos_encoding = True
                ensemble_models.append(ensemble_model)
            ensemble_cost = swm.solver.RankEnsembleCost(
                ensemble_models,
                labels,
            )
            solver = hydra.utils.instantiate(
                cfg.solver,
                model=ensemble_cost,
            )
            print(
                '[eval] shared-population rank ensemble enabled: '
                f'models={ensemble_names}, labels={labels}'
            )
        elif portfolio is not None:
            proposer_names = [str(name) for name in portfolio.proposers]
            if proposer_names[0] != str(cfg.policy):
                raise ValueError(
                    'portfolio.proposers must start with cfg.policy'
                )
            labels = [
                str(name) for name in portfolio.get('labels', proposer_names)
            ]
            if len(labels) != len(proposer_names):
                raise ValueError(
                    'portfolio.labels must match portfolio.proposers'
                )
            seed_offsets = [
                int(value)
                for value in portfolio.get(
                    'seed_offsets',
                    [0] * len(proposer_names),
                )
            ]
            if len(seed_offsets) != len(proposer_names):
                raise ValueError(
                    'portfolio.seed_offsets must match portfolio.proposers'
                )
            proposer_models = [model]
            proposer_solvers = [solver]
            model_cache = {str(cfg.policy): model}
            for proposer_name in proposer_names[1:]:
                proposer = model_cache.get(proposer_name)
                if proposer is None:
                    proposer = swm.wm.utils.load_pretrained(proposer_name)
                    if cfg.get('bf16', False):
                        proposer = proposer.to(torch.bfloat16)
                    proposer = proposer.to('cuda').eval()
                    proposer.requires_grad_(False)
                    proposer.interpolate_pos_encoding = True
                    model_cache[proposer_name] = proposer
                proposer_models.append(proposer)
                proposer_solvers.append(
                    hydra.utils.instantiate(cfg.solver, model=proposer)
                )
            for proposer_solver, seed_offset in zip(
                proposer_solvers,
                seed_offsets,
                strict=True,
            ):
                proposer_solver.torch_gen.manual_seed(
                    int(cfg.seed) + seed_offset
                )
            solver = swm.solver.CEMPortfolioSolver(
                solvers=proposer_solvers,
                models=proposer_models,
                names=labels,
                steps=list(portfolio.steps),
                scorer_batch_size=int(portfolio.get('scorer_batch_size', 1)),
            )
            print(
                '[eval] CEM portfolio enabled: '
                f'proposers={proposer_names}, '
                f'labels={labels}, '
                f'seed_offsets={seed_offsets}, '
                f'steps={list(portfolio.steps)}'
            )
        elif cross_validate is not None:
            verifier_name = str(cross_validate.verifier)
            verifier = swm.wm.utils.load_pretrained(verifier_name)
            if cfg.get('bf16', False):
                verifier = verifier.to(torch.bfloat16)
            verifier = verifier.to('cuda').eval()
            verifier.requires_grad_(False)
            verifier.interpolate_pos_encoding = True
            solver = swm.solver.CrossValidatedCEMSolver(
                solver,
                verifier,
                steps=list(cross_validate.steps),
                selection_space=str(
                    cross_validate.get('selection_space', 'means')
                ),
                verifier_batch_size=int(
                    cross_validate.get('verifier_batch_size', 1)
                ),
                refit_topk=int(cross_validate.get('refit_topk', 30)),
            )
            print(
                '[eval] CEM cross-validation enabled: '
                f'verifier={verifier_name}, '
                f'steps={list(cross_validate.steps)}, '
                f'space={solver.selection_space}, '
                f'refit_topk={solver.refit_topk}'
            )
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(
            swm.data.utils.get_cache_dir(sub_folder='checkpoints'), cfg.policy
        ).parent
        if cfg.policy != 'random'
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {
        ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)
    }
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = (
        'episode_idx' if 'episode_idx' in dataset.column_names else 'ep_idx'
    )
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data('step_idx') <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), 'valid starting points found for evaluation.')

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)['step_idx']

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError(
            'Not enough episodes with sufficient length for evaluation.'
        )

    world.set_policy(policy)

    dataset_action_prior = cfg.get('dataset_action_prior')
    use_dataset_action_prior = bool(
        dataset_action_prior is not None
        and dataset_action_prior.get('enabled', True)
    )
    if use_dataset_action_prior:
        if cfg.policy == 'random' or solver is None:
            raise ValueError('dataset_action_prior requires a world-model policy')
        if 'action' not in process:
            raise ValueError('dataset_action_prior requires an action scaler')
        initial_action = build_dataset_initial_action_prior(
            dataset,
            eval_episodes,
            eval_start_idx,
            horizon=int(cfg.plan_config.horizon),
            action_block=int(cfg.plan_config.action_block),
            action_scaler=process['action'],
            alignment=str(dataset_action_prior.get('alignment', 'next')),
        )
        policy.set_initial_action(initial_action)
        print(
            '[eval] hidden-oracle dataset action prior enabled: '
            f'alignment={dataset_action_prior.get("alignment", "next")}, '
            f'shape={initial_action.shape}. This is diagnostic only.'
        )

    results_path.mkdir(parents=True, exist_ok=True)
    video_path = results_path if cfg.eval.get('video', True) else None
    if video_path is not None:
        print(
            f'[eval] saving videos to {video_path.resolve()} '
            '(one env_{i}.mp4 per env)'
        )
    else:
        print('[eval] video saving disabled')

    autocast_ctx = torch.autocast(
        device_type='cuda',
        dtype=torch.bfloat16,
        enabled=cfg.get('bf16', False),
    )

    if cfg.get('compile', False):
        print('Warming up compiled model...')
        warmup_autocast_ctx = torch.autocast(
            device_type='cuda',
            dtype=torch.bfloat16,
            enabled=cfg.get('bf16', False),
        )
        with warmup_autocast_ctx:
            n = world.num_envs
            world.evaluate(
                dataset=dataset,
                start_steps=eval_start_idx.tolist()[:n],
                goal_offset=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist()[:n],
                callables=OmegaConf.to_container(
                    cfg.eval.get('callables'), resolve=True
                ),
                options=to_container_or_none(cfg.eval.get('reset_options')),
                video=video_path,
            )
        print('Warmup done.')

    history_frames = int(cfg.plan_config.get('history_len', 1)) - 1

    start_time = time.time()
    with autocast_ctx:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(
                cfg.eval.get('callables'), resolve=True
            ),
            options=to_container_or_none(cfg.eval.get('reset_options')),
            video=video_path,
            history_frames=history_frames,
            history_frameskip=int(cfg.plan_config.action_block),
        )
    if use_dataset_action_prior:
        metrics['initial_action_prior'] = {
            'type': 'dataset_expert',
            'alignment': str(dataset_action_prior.get('alignment', 'next')),
            'hidden_oracle': True,
        }
    if solver is not None and hasattr(solver, 'selection_summary'):
        metrics['cross_validation'] = solver.selection_summary()
    end_time = time.time()

    metrics_out = cfg.eval.get('metrics_out')
    if metrics_out is not None:
        metrics_path = Path(str(metrics_out))
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': np.asarray(1),
            'dataset_rows': np.asarray(random_episode_indices, dtype=np.int64),
            'episodes': np.asarray(eval_episodes, dtype=np.int64),
            'starts': np.asarray(eval_start_idx, dtype=np.int64),
            'episode_successes': np.asarray(
                metrics['episode_successes'], dtype=bool
            ),
            'success_rate': np.asarray(metrics['success_rate'], dtype=np.float64),
            'elapsed_seconds': np.asarray(
                end_time - start_time, dtype=np.float64
            ),
            'seed': np.asarray(int(cfg.seed), dtype=np.int64),
            'policy': np.asarray(str(cfg.policy)),
            'metadata': np.asarray(
                json.dumps(
                    {
                        'dataset_action_prior': (
                            OmegaConf.to_container(
                                dataset_action_prior, resolve=True
                            )
                            if dataset_action_prior is not None
                            else None
                        ),
                        'horizon': int(cfg.plan_config.horizon),
                        'receding_horizon': int(
                            cfg.plan_config.receding_horizon
                        ),
                        'action_block': int(cfg.plan_config.action_block),
                        'goal_offset': int(cfg.eval.goal_offset_steps),
                        'eval_budget': int(cfg.eval.eval_budget),
                        'bf16': bool(cfg.get('bf16', False)),
                        'num_samples': int(cfg.solver.num_samples),
                        'cem_steps': int(cfg.solver.n_steps),
                        'topk': int(cfg.solver.topk),
                    },
                    sort_keys=True,
                )
            ),
        }
        for key in ('initial_task_distance', 'final_task_distance'):
            if key in metrics:
                payload[key] = np.asarray(metrics[key], dtype=np.float64)
        np.savez_compressed(metrics_path, **payload)
        print(f'[eval] structured metrics -> {metrics_path}')

    print(metrics)
    if video_path is not None:
        print(f'[eval] videos saved to {video_path.resolve()}')

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open('a') as f:
        f.write('\n')  # separate from previous runs

        f.write('==== CONFIG ====\n')
        f.write(OmegaConf.to_yaml(cfg))
        f.write('\n')

        f.write('==== RESULTS ====\n')
        f.write(f'metrics: {metrics}\n')
        f.write(f'evaluation_time: {end_time - start_time} seconds\n')


if __name__ == '__main__':
    run()
