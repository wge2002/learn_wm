#!/usr/bin/env python3
"""Post-hoc Gate A/G1/G2 audit for trained PushT LeWM checkpoints.

The audit uses a shared bank of physical trajectories for every checkpoint.
It separates four questions:

1. Gate A: does finite-horizon Jacobian shear fall, rather than only scale?
2. G1: are two latent representations related by a bidirectionally accurate
   non-orthogonal linear map (a lower bound on gauge-like reshaping)?
3. G2: does the coordinate-invariant action/residual generalized spectrum
   improve?  The reported ``logdet_I_plus_pencil`` is

       log det(W_r + W_u) - log det(W_r)
       = sum_i log(1 + lambda_i),

   where ``lambda_i`` are generalized eigenvalues of ``(W_u, W_r)``.
4. Sufficiency: do held-out physical-state probes and action sensitivity stay
   intact?

The dynamics are linearized along the same true latent trajectory for every
model.  The predictor consumes a three-frame context, so its Jacobian is taken
with respect to the full augmented context.  Residual and action perturbations
are then propagated through that time-varying linear system to the same
terminal step before forming ``W_r`` and ``W_u``.

This is an exploratory checkpoint audit, not a substitute for the final
multi-seed gate.  In particular, exact contact labels are unavailable in the
stored HDF5 file; ``block_moved`` is reported only as a contact/push proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.func import jacrev


def parse_policy(raw: str) -> tuple[str, str]:
    if '=' not in raw:
        raise argparse.ArgumentTypeError('--policy must be LABEL=CHECKPOINT')
    label, checkpoint = raw.split('=', 1)
    if not label or not checkpoint:
        raise argparse.ArgumentTypeError('--policy must be LABEL=CHECKPOINT')
    return label, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--policy', action='append', type=parse_policy, required=True
    )
    parser.add_argument('--reference', default=None)
    parser.add_argument(
        '--pair-id',
        default=None,
        help='Training-seed/pair identifier used by the cross-pair summarizer.',
    )
    parser.add_argument(
        '--training-seed',
        type=int,
        default=None,
        help='Global RNG seed used to train both checkpoints in this pair.',
    )
    parser.add_argument(
        '--checkpoint-epoch',
        type=int,
        default=None,
        help='Epoch shared by the two checkpoint arguments.',
    )
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num-samples', type=int, default=320)
    parser.add_argument('--jacobian-samples', type=int, default=20)
    parser.add_argument('--history', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--frameskip', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--ridge-alpha', type=float, default=1e-3)
    parser.add_argument('--block-motion-threshold', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=20260810)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--skip-jacobians', action='store_true')
    return parser.parse_args()


def wrap_angle(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def take_rows(dataset, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows through a sorted unique gather."""

    unique, inverse = np.unique(indices.reshape(-1), return_inverse=True)
    gathered = np.asarray(dataset[unique])
    return gathered[inverse].reshape(*indices.shape, *dataset.shape[1:])


def sample_bank(args: argparse.Namespace) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    total_frames = args.history + args.horizon
    span = total_frames * args.frameskip

    with h5py.File(args.dataset, 'r') as handle:
        lengths = np.asarray(handle['ep_len'], dtype=np.int64)
        offsets = np.asarray(handle['ep_offset'], dtype=np.int64)
        valid_counts = np.maximum(lengths - span + 1, 0)
        valid_episodes = np.flatnonzero(valid_counts > 0)
        if len(valid_episodes) == 0:
            raise ValueError(f'no episode is long enough for span={span}')
        weights = valid_counts[valid_episodes].astype(np.float64)
        weights /= weights.sum()
        episodes = rng.choice(
            valid_episodes,
            size=args.num_samples,
            replace=True,
            p=weights,
        )
        starts = np.asarray(
            [rng.integers(valid_counts[ep]) for ep in episodes],
            dtype=np.int64,
        )
        bases = offsets[episodes] + starts
        frame_indices = bases[:, None] + (
            np.arange(total_frames, dtype=np.int64)[None, :] * args.frameskip
        )
        action_indices = frame_indices[:, :, None] + np.arange(
            args.frameskip, dtype=np.int64
        )[None, None, :]

        print('[audit] reading shared pixel/state/action bank', flush=True)
        pixels = take_rows(handle['pixels'], frame_indices)
        states = take_rows(handle['state'], frame_indices).astype(np.float32)
        actions = take_rows(handle['action'], action_indices).astype(np.float32)
        all_actions = np.asarray(handle['action'], dtype=np.float32)

    valid_actions = all_actions[~np.isnan(all_actions).any(axis=1)]
    action_mean = valid_actions.mean(axis=0, dtype=np.float64)
    action_std = valid_actions.std(axis=0, dtype=np.float64)
    normalized = (actions - action_mean) / np.maximum(action_std, 1e-8)
    normalized = normalized.reshape(args.num_samples, total_frames, -1)

    block_delta = states[:, args.history :, 2:4] - states[
        :, args.history - 1 : -1, 2:4
    ]
    angle_delta = wrap_angle(
        states[:, args.history :, 4]
        - states[:, args.history - 1 : -1, 4]
    )
    block_motion = np.sqrt(
        np.square(block_delta).sum(axis=-1) + np.square(angle_delta)
    )
    block_moved = block_motion.max(axis=1) > args.block_motion_threshold

    order = rng.permutation(args.num_samples)
    split = max(1, int(0.75 * args.num_samples))
    train_idx = np.sort(order[:split])
    test_idx = np.sort(order[split:])
    return {
        'pixels': pixels,
        'states': states,
        'actions': normalized.astype(np.float32),
        'episodes': episodes,
        'starts': starts,
        'block_motion': block_motion.astype(np.float32),
        'block_moved': block_moved,
        'train_idx': train_idx,
        'test_idx': test_idx,
        'action_mean': action_mean,
        'action_std': action_std,
    }


def images_to_tensor(frames: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float().div_(255.0)
    stats = spt.data.dataset_stats.ImageNet
    mean = torch.as_tensor(stats['mean'], dtype=tensor.dtype).view(
        1, 1, 3, 1, 1
    )
    std = torch.as_tensor(stats['std'], dtype=tensor.dtype).view(
        1, 1, 3, 1, 1
    )
    return (tensor - mean) / std


@torch.inference_mode()
def encode_bank(
    model,
    pixels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = []
    dtype = next(model.parameters()).dtype
    for start in range(0, len(pixels), batch_size):
        batch = images_to_tensor(pixels[start : start + batch_size]).to(
            device=device, dtype=dtype
        )
        encoded = model.encode({'pixels': batch})['emb']
        rows.append(encoded.float().cpu().numpy())
    return np.concatenate(rows, axis=0)


@torch.inference_mode()
def residual_covariances(
    model,
    z: np.ndarray,
    actions: np.ndarray,
    *,
    history: int,
    horizon: int,
    device: torch.device,
) -> tuple[list[np.ndarray], dict[str, float]]:
    zt = torch.from_numpy(z).to(device)
    at = torch.from_numpy(actions).to(device)
    action_embeddings = model.action_encoder(at)
    residuals = []
    for step in range(horizon):
        pred = model.predict(
            zt[:, step : step + history],
            action_embeddings[:, step : step + history],
        )[:, -1]
        residuals.append(
            (zt[:, step + history] - pred).float().cpu().numpy()
        )

    covariances = []
    for residual in residuals:
        centered = residual - residual.mean(axis=0, keepdims=True)
        covariances.append(centered.T @ centered / max(1, len(centered) - 1))
    stacked = np.stack(residuals, axis=1)
    return covariances, {
        'one_step_residual_l2_median': float(
            np.median(np.linalg.norm(stacked[:, 0], axis=-1))
        ),
        'residual_l2_terminal_median': float(
            np.median(np.linalg.norm(stacked[:, -1], axis=-1))
        ),
        'residual_mse_mean': float(np.mean(np.square(stacked))),
    }


def effective_rank(values: np.ndarray) -> dict[str, float | int]:
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eig = np.linalg.eigvalsh(covariance)[::-1].clip(min=0.0)
    total = eig.sum()
    fraction = eig / max(total, 1e-30)
    cumulative = np.cumsum(fraction)
    entropy = -np.sum(fraction * np.log(fraction + 1e-30))
    return {
        'covariance_trace': float(total),
        'participation_rank': float(total**2 / max(np.square(eig).sum(), 1e-30)),
        'entropy_effective_rank': float(np.exp(entropy)),
        'pca_rank_99': int(np.searchsorted(cumulative, 0.99) + 1),
    }


def local_dimension_summary(
    values: np.ndarray, *, seed: int
) -> dict[str, float | int]:
    """Estimate support dimension; exact full-D Gaussian rigidity needs D."""

    scaler = StandardScaler().fit(values)
    normalized = scaler.transform(values)
    normalized = normalized[:, scaler.var_ > 1e-20]
    normalized = np.unique(normalized, axis=0)
    unique_count = len(normalized)
    rng = np.random.default_rng(seed)
    if len(normalized) > 4_000:
        normalized = normalized[
            rng.choice(len(normalized), size=4_000, replace=False)
        ]
    neighbors = min(65, len(normalized))
    if neighbors < 21:
        return {'status': 'too_few_unique_samples', 'unique_samples': len(normalized)}

    nn = NearestNeighbors(n_neighbors=neighbors, n_jobs=-1).fit(normalized)
    distances, indices = nn.kneighbors(normalized, return_distance=True)
    distances = distances[:, 1:]
    knn = 20
    local = distances[:, :knn]
    valid = (local[:, 0] > 1e-12) & (local[:, -1] > local[:, 0])
    local = local[valid]
    mle = (knn - 1) / np.maximum(
        np.log(local[:, -1, None] / local[:, :-1]).sum(axis=1),
        1e-12,
    )
    mle = mle[np.isfinite(mle)]

    anchor_count = min(256, len(normalized))
    anchors = rng.choice(len(normalized), size=anchor_count, replace=False)
    local_ranks = []
    for anchor in anchors:
        neighborhood = normalized[indices[anchor, 1:65]].copy()
        neighborhood -= neighborhood.mean(axis=0, keepdims=True)
        singular = np.linalg.svd(
            neighborhood, full_matrices=False, compute_uv=False
        )
        energy = np.square(singular)
        cumulative = np.cumsum(energy) / max(energy.sum(), 1e-30)
        local_ranks.append(int(np.searchsorted(cumulative, 0.95) + 1))
    return {
        'unique_samples_total': int(unique_count),
        'estimator_samples': int(len(normalized)),
        'duplicate_fraction': float(1.0 - unique_count / len(values)),
        'mle_knn': knn,
        'mle_id_median': float(np.median(mle)),
        'mle_id_trimmed_mean': float(
            np.mean(np.clip(mle, np.percentile(mle, 5), np.percentile(mle, 95)))
        ),
        'local_pca_neighbors': 64,
        'local_pca_rank95_median': float(np.median(local_ranks)),
        'local_pca_rank95_p25': float(np.percentile(local_ranks, 25)),
        'local_pca_rank95_p75': float(np.percentile(local_ranks, 75)),
    }


def standardized_ridge_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    regressor = Ridge(alpha=alpha)
    regressor.fit(x_scaler.transform(x_train), y_scaler.transform(y_train))
    prediction_scaled = regressor.predict(x_scaler.transform(x_test))
    prediction = y_scaler.inverse_transform(prediction_scaled)
    return {
        'r2_uniform': float(r2_score(y_test, prediction)),
        'r2_by_coordinate': np.asarray(
            r2_score(y_test, prediction, multioutput='raw_values')
        ).tolist(),
    }


def sufficiency_probes(
    z: np.ndarray,
    states: np.ndarray,
    block_motion: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
    block_motion_threshold: float,
) -> dict[str, object]:
    z_train = z[train_idx].reshape(-1, z.shape[-1])
    z_test = z[test_idx].reshape(-1, z.shape[-1])
    state_train = states[train_idx].reshape(-1, states.shape[-1])
    state_test = states[test_idx].reshape(-1, states.shape[-1])
    targets = {
        'agent_xy': (state_train[:, :2], state_test[:, :2]),
        'block_xy': (state_train[:, 2:4], state_test[:, 2:4]),
        'block_angle_sincos': (
            np.column_stack((np.sin(state_train[:, 4]), np.cos(state_train[:, 4]))),
            np.column_stack((np.sin(state_test[:, 4]), np.cos(state_test[:, 4]))),
        ),
        'agent_velocity': (state_train[:, 5:7], state_test[:, 5:7]),
    }
    result = {
        name: standardized_ridge_probe(
            z_train, y_train, z_test, y_test, alpha
        )
        for name, (y_train, y_test) in targets.items()
    }

    # The HDF5 file has no exact contact bit.  Predict whether the block moves
    # during the next model step as a conservative control/task proxy.
    x_motion = z[:, 2:-1].reshape(-1, z.shape[-1])
    y_motion = (block_motion > block_motion_threshold).reshape(-1).astype(
        np.int64
    )
    trajectory_ids = np.repeat(np.arange(len(z)), z.shape[1] - 3)
    train_mask = np.isin(trajectory_ids, train_idx)
    test_mask = np.isin(trajectory_ids, test_idx)
    if len(np.unique(y_motion[train_mask])) == 2 and len(
        np.unique(y_motion[test_mask])
    ) == 2:
        scaler = StandardScaler().fit(x_motion[train_mask])
        classifier = LogisticRegression(
            max_iter=500, class_weight='balanced', random_state=0
        )
        classifier.fit(scaler.transform(x_motion[train_mask]), y_motion[train_mask])
        probability = classifier.predict_proba(
            scaler.transform(x_motion[test_mask])
        )[:, 1]
        prediction = probability >= 0.5
        result['block_motion_probe'] = {
            'positive_rate_test': float(y_motion[test_mask].mean()),
            'balanced_accuracy': float(
                balanced_accuracy_score(y_motion[test_mask], prediction)
            ),
            'roc_auc': float(roc_auc_score(y_motion[test_mask], probability)),
        }
    else:
        result['block_motion_probe'] = {'status': 'single_class'}
    return result


def orthogonal_map(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd(
        (x_train - x_mean).T @ (y_train - y_mean), full_matrices=False
    )
    rotation = u @ vt
    prediction = (x_test - x_mean) @ rotation + y_mean
    return {
        'r2': float(r2_score(y_test, prediction)),
        'relative_frobenius_error': float(
            np.linalg.norm(prediction - y_test)
            / max(np.linalg.norm(y_test - y_test.mean(axis=0)), 1e-30)
        ),
    }


def g1_bidirectional_maps(
    source: np.ndarray,
    target: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    def flatten(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return values[indices].reshape(-1, values.shape[-1])

    sx, sy = flatten(source, train_idx), flatten(target, train_idx)
    tx, ty = flatten(source, test_idx), flatten(target, test_idx)

    sx_scaler = StandardScaler().fit(sx)
    sy_scaler = StandardScaler().fit(sy)
    sx_train = sx_scaler.transform(sx)
    sy_train = sy_scaler.transform(sy)
    tx_scaled = sx_scaler.transform(tx)
    ty_scaled = sy_scaler.transform(ty)
    forward = Ridge(alpha=alpha).fit(sx_train, sy_train)
    reverse = Ridge(alpha=alpha).fit(sy_train, sx_train)

    forward_prediction_scaled = forward.predict(tx_scaled)
    reverse_prediction_scaled = reverse.predict(ty_scaled)
    forward_prediction = sy_scaler.inverse_transform(
        forward_prediction_scaled
    )
    reverse_prediction = sx_scaler.inverse_transform(
        reverse_prediction_scaled
    )
    cycle_source = sx_scaler.inverse_transform(
        reverse.predict(forward_prediction_scaled)
    )
    cycle_target = sy_scaler.inverse_transform(
        forward.predict(reverse_prediction_scaled)
    )

    def prediction_summary(truth, prediction):
        return {
            'r2_uniform': float(r2_score(truth, prediction)),
            'relative_frobenius_error': float(
                np.linalg.norm(prediction - truth)
                / max(np.linalg.norm(truth - truth.mean(axis=0)), 1e-30)
            ),
        }

    def coefficient_summary(regressor):
        singular = np.linalg.svd(regressor.coef_, compute_uv=False)
        tolerance = singular[0] * 1e-6
        return {
            'numerical_rank_1e-6': int(np.sum(singular > tolerance)),
            'max_singular_value': float(singular[0]),
            'min_singular_value': float(singular[-1]),
            'condition_number': float(
                singular[0] / max(singular[-1], 1e-30)
            ),
        }

    forward_linear = prediction_summary(ty, forward_prediction)
    forward_linear['coefficient'] = coefficient_summary(forward)
    reverse_linear = prediction_summary(tx, reverse_prediction)
    reverse_linear['coefficient'] = coefficient_summary(reverse)
    return {
        'forward': {
            'orthogonal': orthogonal_map(sx, sy, tx, ty),
            'linear': forward_linear,
        },
        'reverse': {
            'orthogonal': orthogonal_map(sy, sx, ty, tx),
            'linear': reverse_linear,
        },
        'cycle': {
            'source_to_target_to_source': prediction_summary(tx, cycle_source),
            'target_to_source_to_target': prediction_summary(ty, cycle_target),
        },
        'nonlinear_invertible_status': 'not_run_in_exploratory_audit',
    }


def spectrum_summary(matrix: np.ndarray) -> dict[str, float]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    floor = max(float(singular[0]) * 1e-12, 1e-30)
    log_singular = np.log(np.maximum(singular, floor))
    return {
        'max_gain': float(singular[0]),
        'median_gain': float(np.median(singular)),
        'log_scale_mean': float(log_singular.mean()),
        'log_shear_rms': float(log_singular.std()),
        'log_spread_p95_p05': float(
            np.percentile(log_singular, 95) - np.percentile(log_singular, 5)
        ),
    }


def propagate_context(jacobians: list[np.ndarray], initial: np.ndarray) -> np.ndarray:
    dim = jacobians[0].shape[0] if jacobians else initial.shape[1]
    context_map = initial
    output = context_map[-dim:]
    for jacobian in jacobians:
        output = jacobian @ context_map
        context_map = np.vstack((context_map[dim:], output))
    return output


def residual_terminal_maps(
    context_jacobians: list[np.ndarray], dim: int
) -> list[np.ndarray]:
    result = []
    history = context_jacobians[0].shape[1] // dim
    injected = np.vstack(
        (np.zeros(((history - 1) * dim, dim)), np.eye(dim))
    )
    for source in range(len(context_jacobians)):
        if source == len(context_jacobians) - 1:
            result.append(np.eye(dim))
        else:
            result.append(
                propagate_context(context_jacobians[source + 1 :], injected)
            )
    return result


def pencil_summary(w_residual: np.ndarray, w_action: np.ndarray) -> dict[str, object]:
    wr = 0.5 * (w_residual + w_residual.T)
    wu = 0.5 * (w_action + w_action.T)
    wr_eig, wr_vec = np.linalg.eigh(wr)
    maximum = max(float(wr_eig[-1]), 1e-30)
    floor = maximum * 1e-12
    clipped = np.maximum(wr_eig, floor)
    whitening = wr_vec @ np.diag(1.0 / np.sqrt(clipped))
    pencil = whitening.T @ wu @ whitening
    pencil = 0.5 * (pencil + pencil.T)
    eigenvalues = np.linalg.eigvalsh(pencil)[::-1].clip(min=0.0)
    positive = eigenvalues[eigenvalues > max(eigenvalues[0] * 1e-10, 1e-14)]
    return {
        'wr_condition_clipped': float(clipped[-1] / clipped[0]),
        'wr_clipped_directions': int(np.sum(wr_eig < floor)),
        'action_rank': int(len(positive)),
        'lambda_top': float(eigenvalues[0]),
        'lambda_positive_median': float(np.median(positive)) if len(positive) else 0.0,
        'trace_pencil': float(eigenvalues.sum()),
        'logdet_I_plus_pencil': float(np.log1p(eigenvalues).sum()),
        'lambda_top10': eigenvalues[:10].tolist(),
    }


def local_jacobians(
    model,
    z: np.ndarray,
    actions: np.ndarray,
    *,
    history: int,
    horizon: int,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    context_jacobians = []
    action_jacobians = []
    for step in range(horizon):
        context = torch.from_numpy(
            z[step : step + history].reshape(-1)
        ).to(device)
        prior_actions = torch.from_numpy(
            actions[step : step + history - 1]
        ).to(device)
        last_action = torch.from_numpy(
            actions[step + history - 1]
        ).to(device)

        def transition(flat_context, current_action):
            ctx = flat_context.reshape(history, -1)
            action_window = torch.cat(
                (prior_actions, current_action.unsqueeze(0)), dim=0
            )
            action_embedding = model.action_encoder(action_window.unsqueeze(0))
            return model.predict(ctx.unsqueeze(0), action_embedding)[0, -1]

        jac_context, jac_action = jacrev(
            transition, argnums=(0, 1)
        )(context, last_action)
        context_jacobians.append(jac_context.detach().float().cpu().numpy())
        action_jacobians.append(jac_action.detach().float().cpu().numpy())
    return context_jacobians, action_jacobians


def one_trajectory_metrics(
    context_jacobians: list[np.ndarray],
    action_jacobians: list[np.ndarray],
    residual_covariances_: list[np.ndarray],
) -> dict[str, object]:
    dim = context_jacobians[0].shape[0]
    history = context_jacobians[0].shape[1] // dim
    initial_last = np.vstack(
        (np.zeros(((history - 1) * dim, dim)), np.eye(dim))
    )
    horizon_map = propagate_context(context_jacobians, initial_last)
    residual_maps = residual_terminal_maps(context_jacobians, dim)

    wr = np.zeros((dim, dim), dtype=np.float64)
    wu = np.zeros((dim, dim), dtype=np.float64)
    for residual_map, action_jacobian, covariance in zip(
        residual_maps,
        action_jacobians,
        residual_covariances_,
        strict=True,
    ):
        residual_map = residual_map.astype(np.float64)
        action_map = residual_map @ action_jacobian.astype(np.float64)
        wr += residual_map @ covariance.astype(np.float64) @ residual_map.T
        wu += action_map @ action_map.T

    action_energy = float(np.trace(wu))
    residual_energy = float(np.trace(wr))
    return {
        'one_step': spectrum_summary(
            context_jacobians[0][:, -dim:].astype(np.float64)
        ),
        'horizon': spectrum_summary(horizon_map.astype(np.float64)),
        'action_energy_trace': action_energy,
        'residual_energy_trace': residual_energy,
        'action_to_residual_trace_ratio': action_energy / max(residual_energy, 1e-30),
        'pencil': pencil_summary(wr, wu),
    }


def aggregate_trajectory_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    paths = {
        'one_step.max_gain': lambda r: r['one_step']['max_gain'],
        'one_step.log_scale_mean': lambda r: r['one_step']['log_scale_mean'],
        'one_step.log_shear_rms': lambda r: r['one_step']['log_shear_rms'],
        'horizon.max_gain': lambda r: r['horizon']['max_gain'],
        'horizon.log_scale_mean': lambda r: r['horizon']['log_scale_mean'],
        'horizon.log_shear_rms': lambda r: r['horizon']['log_shear_rms'],
        'horizon.log_spread_p95_p05': lambda r: r['horizon']['log_spread_p95_p05'],
        'action_energy_trace': lambda r: r['action_energy_trace'],
        'residual_energy_trace': lambda r: r['residual_energy_trace'],
        'action_to_residual_trace_ratio': lambda r: r['action_to_residual_trace_ratio'],
        'pencil.lambda_top': lambda r: r['pencil']['lambda_top'],
        'pencil.trace_pencil': lambda r: r['pencil']['trace_pencil'],
        'pencil.logdet_I_plus_pencil': lambda r: r['pencil']['logdet_I_plus_pencil'],
    }
    aggregate = {}
    for name, getter in paths.items():
        values = np.asarray([getter(row) for row in rows], dtype=np.float64)
        aggregate[name] = {
            'mean': float(values.mean()),
            'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
        }
    return aggregate


def paired_differences(
    reference: list[dict[str, object]],
    candidate: list[dict[str, object]],
    *,
    seed: int,
) -> dict[str, object]:
    paths = {
        'horizon.max_gain': lambda r: r['horizon']['max_gain'],
        'horizon.log_scale_mean': lambda r: r['horizon']['log_scale_mean'],
        'horizon.log_shear_rms': lambda r: r['horizon']['log_shear_rms'],
        'action_energy_trace': lambda r: r['action_energy_trace'],
        'residual_energy_trace': lambda r: r['residual_energy_trace'],
        'pencil.trace_pencil': lambda r: r['pencil']['trace_pencil'],
        'pencil.logdet_I_plus_pencil': lambda r: r['pencil']['logdet_I_plus_pencil'],
    }
    rng = np.random.default_rng(seed)
    result = {}
    for name, getter in paths.items():
        ref = np.asarray([getter(row) for row in reference], dtype=np.float64)
        cand = np.asarray([getter(row) for row in candidate], dtype=np.float64)
        diff = cand - ref
        bootstrap = np.empty(2_000, dtype=np.float64)
        for index in range(len(bootstrap)):
            chosen = rng.integers(0, len(diff), size=len(diff))
            bootstrap[index] = diff[chosen].mean()
        result[name] = {
            'mean_difference': float(diff.mean()),
            'median_difference': float(np.median(diff)),
            'mean_difference_ci95': np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            'candidate_over_reference_median': float(
                np.median(cand / np.maximum(np.abs(ref), 1e-30))
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if len(args.policy) < 1:
        raise ValueError('at least one --policy is required')
    labels = [label for label, _ in args.policy]
    if len(labels) != len(set(labels)):
        raise ValueError('policy labels must be unique')
    reference = args.reference or labels[0]
    if reference not in labels:
        raise ValueError(f'unknown --reference {reference!r}')
    if args.jacobian_samples > args.num_samples:
        raise ValueError('--jacobian-samples cannot exceed --num-samples')

    started = time.time()
    bank = sample_bank(args)
    device = torch.device(args.device)
    embeddings: dict[str, np.ndarray] = {}
    model_results: dict[str, object] = {}
    trajectory_results: dict[str, list[dict[str, object]]] = {}
    rng = np.random.default_rng(args.seed + 17)
    jacobian_indices = np.sort(
        rng.choice(
            args.num_samples,
            size=args.jacobian_samples,
            replace=False,
        )
    )

    for label, checkpoint in args.policy:
        print(f'[audit] loading {label}={checkpoint}', flush=True)
        model = swm.wm.utils.load_pretrained(checkpoint).to(device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        state_tensors = list(model.state_dict().values())
        nonfinite_parameters = sum(
            int((~torch.isfinite(value)).sum().item())
            for value in state_tensors
            if value.is_floating_point()
        )
        if nonfinite_parameters:
            raise ValueError(
                f'{label} checkpoint contains {nonfinite_parameters} '
                'non-finite state values'
            )
        print(f'[audit] encoding {label}', flush=True)
        z = encode_bank(
            model,
            bank['pixels'],
            batch_size=args.batch_size,
            device=device,
        )
        if not np.isfinite(z).all():
            raise ValueError(f'{label} encoder produced non-finite embeddings')
        embeddings[label] = z
        covariance, residual_summary = residual_covariances(
            model,
            z,
            bank['actions'],
            history=args.history,
            horizon=args.horizon,
            device=device,
        )
        result = {
            'checkpoint': checkpoint,
            'latent': {
                **effective_rank(z.reshape(-1, z.shape[-1])),
                'local_dimension': local_dimension_summary(
                    z.reshape(-1, z.shape[-1]), seed=args.seed
                ),
            },
            'prediction': residual_summary,
            'sufficiency': sufficiency_probes(
                z,
                bank['states'],
                bank['block_motion'],
                bank['train_idx'],
                bank['test_idx'],
                args.ridge_alpha,
                args.block_motion_threshold,
            ),
        }

        if not args.skip_jacobians:
            rows = []
            for count, sample_index in enumerate(jacobian_indices, start=1):
                print(
                    f'[audit] {label} Jacobian {count}/{len(jacobian_indices)} '
                    f'sample={sample_index}',
                    flush=True,
                )
                context_jacobians, action_jacobians = local_jacobians(
                    model,
                    z[sample_index],
                    bank['actions'][sample_index],
                    history=args.history,
                    horizon=args.horizon,
                    device=device,
                )
                row = one_trajectory_metrics(
                    context_jacobians, action_jacobians, covariance
                )
                row['sample_index'] = int(sample_index)
                row['block_moved'] = bool(bank['block_moved'][sample_index])
                rows.append(row)
            result['controlled_metric'] = {
                'aggregate': aggregate_trajectory_metrics(rows),
                'per_trajectory': rows,
                'block_moved_proxy_rate': float(
                    bank['block_moved'][jacobian_indices].mean()
                ),
            }
            trajectory_results[label] = rows
        model_results[label] = result
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    g1 = {}
    for label in labels:
        if label == reference:
            continue
        g1[f'{reference}_vs_{label}'] = g1_bidirectional_maps(
            embeddings[reference],
            embeddings[label],
            bank['train_idx'],
            bank['test_idx'],
            args.ridge_alpha,
        )

    paired = {}
    if not args.skip_jacobians:
        for label in labels:
            if label == reference:
                continue
            paired[f'{label}_minus_{reference}'] = paired_differences(
                trajectory_results[reference],
                trajectory_results[label],
                seed=args.seed,
            )

    output = {
        'config': {
            key: (
                [list(item) for item in value]
                if key == 'policy'
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        'reference': reference,
        'metadata': {
            'elapsed_seconds': time.time() - started,
            'episodes': bank['episodes'].tolist(),
            'starts': bank['starts'].tolist(),
            'train_indices': bank['train_idx'].tolist(),
            'test_indices': bank['test_idx'].tolist(),
            'jacobian_indices': jacobian_indices.tolist(),
            'block_moved_proxy_rate': float(bank['block_moved'].mean()),
            'action_mean': bank['action_mean'].tolist(),
            'action_std': bank['action_std'].tolist(),
            'limitations': [
                'block_moved is a proxy, not an exact contact label',
                (
                    'G1 nonlinear invertible map is not included in this '
                    'exploratory pass'
                ),
                (
                    'action covariance is identity in normalized '
                    'five-action-block coordinates'
                ),
                (
                    'the intrinsic CP_H certificate still requires a '
                    'matched physical-perturbation W_0'
                ),
            ],
        },
        'models': model_results,
        'g1_bidirectional_maps': g1,
        'paired_controlled_metric': paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'[audit] wrote {args.output} in {time.time() - started:.1f}s')


if __name__ == '__main__':
    main()
