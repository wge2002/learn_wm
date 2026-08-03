#!/usr/bin/env python3
"""Select a dynamics-compatible geometry inside an exact Gaussian gauge.

The observed state is a nonlinear, Gaussian-measure-preserving radial twist of
a simple controlled OU process.  Every candidate encoder is another radial
twist, so every candidate latent is *exactly* standard Gaussian.  The scan asks
which candidate makes the controlled dynamics easiest to fit and least
geometrically strained.

This isolates the positive question that marginal regularization leaves open:
which Gaussian-preserving gauge gives the best dynamics?
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


Array = np.ndarray


def rotate(points: Array, angles: Array) -> Array:
    cosine = np.cos(angles)
    sine = np.sin(angles)
    x = cosine * points[:, 0] - sine * points[:, 1]
    y = sine * points[:, 0] + cosine * points[:, 1]
    return np.column_stack((x, y))


def radial_twist(points: Array, strength: float) -> Array:
    radius_squared = np.square(points).sum(axis=1)
    return rotate(points, strength * radius_squared)


def radial_twist_jacobian(points: Array, strength: float) -> Array:
    """Jacobian of ``T_c(x) = R_{c ||x||^2} x`` for a batch of points."""

    radius_squared = np.square(points).sum(axis=1)
    angles = strength * radius_squared
    cosine = np.cos(angles)
    sine = np.sin(angles)

    rotation = np.empty((len(points), 2, 2))
    rotation[:, 0, 0] = cosine
    rotation[:, 0, 1] = -sine
    rotation[:, 1, 0] = sine
    rotation[:, 1, 1] = cosine

    rotated_tangent = np.column_stack((-points[:, 1], points[:, 0]))
    shear = np.eye(2)[None, :, :] + 2.0 * strength * (
        rotated_tangent[:, :, None] * points[:, None, :]
    )
    return rotation @ shear


def reflection_matrices(actions: Array, axis_angle: float) -> Array:
    """Reflection across axes at ``+/- axis_angle`` selected by the action."""

    cosine = math.cos(2.0 * axis_angle)
    sine = math.sin(2.0 * axis_angle)
    matrices = np.empty((len(actions), 2, 2))
    matrices[:, 0, 0] = cosine
    matrices[:, 0, 1] = actions * sine
    matrices[:, 1, 0] = actions * sine
    matrices[:, 1, 1] = -cosine
    return matrices


def apply_matrices(matrices: Array, points: Array) -> Array:
    return np.einsum('nij,nj->ni', matrices, points)


def singular_values_2x2(matrices: Array) -> tuple[Array, Array]:
    frobenius_squared = np.square(matrices).sum(axis=(1, 2))
    determinant = np.abs(
        matrices[:, 0, 0] * matrices[:, 1, 1]
        - matrices[:, 0, 1] * matrices[:, 1, 0]
    )
    discriminant = np.maximum(
        frobenius_squared**2 - 4.0 * determinant**2,
        0.0,
    )
    largest_eigenvalue = 0.5 * (
        frobenius_squared + np.sqrt(discriminant)
    )
    largest = np.sqrt(largest_eigenvalue)
    smallest = determinant / np.maximum(largest, 1e-15)
    return largest, smallest


def marginal_covariance_error(samples: Array) -> float:
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    return float(
        np.linalg.norm(covariance - np.eye(2), ord='fro') / math.sqrt(2.0)
    )


def fit_bilinear_predictor(
    z: Array,
    actions: Array,
    z_next: Array,
) -> tuple[float, float]:
    features = np.column_stack((z, actions[:, None] * z, np.ones(len(z))))
    coefficients, *_ = np.linalg.lstsq(features, z_next, rcond=None)
    prediction = features @ coefficients
    mse = float(np.square(z_next - prediction).mean())
    residual = np.square(z_next - prediction).sum()
    total = np.square(z_next - z_next.mean(axis=0)).sum()
    r_squared = float(1.0 - residual / total)
    return mse, r_squared


def one_step_skeleton_jacobian(
    base_state: Array,
    actions: Array,
    gauge_strength: float,
    rho: float,
    axis_angle: float,
) -> Array:
    """Jacobian in the candidate Gaussian gauge, excluding process noise."""

    z = radial_twist(base_state, gauge_strength)
    reflections = reflection_matrices(actions, axis_angle)
    next_base = rho * apply_matrices(reflections, base_state)
    left = radial_twist_jacobian(next_base, gauge_strength)
    right = radial_twist_jacobian(z, -gauge_strength)
    linear = rho * reflections
    return left @ linear @ right


def metric_statistics(base_state: Array, gauge_strength: float) -> dict[str, float]:
    jacobian = radial_twist_jacobian(base_state, gauge_strength)
    largest, smallest = singular_values_2x2(jacobian)
    metric_log_condition = 2.0 * np.log(largest / smallest)
    return {
        'metric_log_condition_p50': float(
            np.quantile(metric_log_condition, 0.50)
        ),
        'metric_log_condition_p95': float(
            np.quantile(metric_log_condition, 0.95)
        ),
        'metric_log_condition_p99': float(
            np.quantile(metric_log_condition, 0.99)
        ),
    }


def one_step_strain_statistics(jacobian: Array, rho: float) -> dict[str, float]:
    largest, smallest = singular_values_2x2(jacobian)
    log_shear = np.log(largest / smallest)
    normalized_gain = largest / rho
    return {
        'one_step_log_shear_mean': float(log_shear.mean()),
        'one_step_log_shear_p95': float(np.quantile(log_shear, 0.95)),
        'one_step_gain_over_rho_p50': float(
            np.quantile(normalized_gain, 0.50)
        ),
        'one_step_gain_over_rho_p95': float(
            np.quantile(normalized_gain, 0.95)
        ),
        'one_step_gain_over_rho_p99': float(
            np.quantile(normalized_gain, 0.99)
        ),
    }


def product_gain_statistics(
    initial_base_state: Array,
    action_sequence: Array,
    gauge_strength: float,
    rho: float,
    axis_angle: float,
) -> dict[str, float]:
    state = initial_base_state.copy()
    product = np.broadcast_to(np.eye(2), (len(state), 2, 2)).copy()

    for step in range(action_sequence.shape[1]):
        actions = action_sequence[:, step]
        jacobian = one_step_skeleton_jacobian(
            state,
            actions,
            gauge_strength,
            rho,
            axis_angle,
        )
        product = jacobian @ product
        reflections = reflection_matrices(actions, axis_angle)
        state = rho * apply_matrices(reflections, state)

    largest, smallest = singular_values_2x2(product)
    baseline_gain = rho ** action_sequence.shape[1]
    normalized_gain = largest / baseline_gain
    log_shear = np.log(largest / smallest)
    return {
        'product_gain_over_rho_h_p50': float(
            np.quantile(normalized_gain, 0.50)
        ),
        'product_gain_over_rho_h_p95': float(
            np.quantile(normalized_gain, 0.95)
        ),
        'product_gain_over_rho_h_p99': float(
            np.quantile(normalized_gain, 0.99)
        ),
        'product_log_shear_p95': float(np.quantile(log_shear, 0.95)),
    }


def scan(args: argparse.Namespace) -> dict[str, object]:
    rng = np.random.default_rng(args.seed)
    base = rng.normal(size=(args.samples, 2))
    actions = rng.choice(np.array([-1.0, 1.0]), size=args.samples)
    noise = rng.normal(size=base.shape)
    reflections = reflection_matrices(actions, args.axis_angle)
    next_base = args.rho * apply_matrices(reflections, base)
    next_base += math.sqrt(1.0 - args.rho**2) * noise

    geometry_count = min(args.geometry_samples, args.samples)
    geometry_base = base[:geometry_count]
    geometry_actions = actions[:geometry_count]
    action_sequence = rng.choice(
        np.array([-1.0, 1.0]),
        size=(geometry_count, args.product_horizon),
    )

    beta_values = np.linspace(
        args.beta_min,
        args.beta_max,
        args.beta_steps,
    )
    rows = []
    for beta in beta_values:
        effective_gauge = args.observation_twist + float(beta)
        z = radial_twist(base, effective_gauge)
        z_next = radial_twist(next_base, effective_gauge)
        linear_mse, linear_r_squared = fit_bilinear_predictor(
            z,
            actions,
            z_next,
        )
        jacobian = one_step_skeleton_jacobian(
            geometry_base,
            geometry_actions,
            effective_gauge,
            args.rho,
            args.axis_angle,
        )
        metrics: dict[str, float] = {
            'encoder_beta': float(beta),
            'effective_gauge': effective_gauge,
            'marginal_covariance_error': marginal_covariance_error(z),
            'mean_radius_squared': float(np.square(z).sum(axis=1).mean()),
            'bilinear_prediction_mse': linear_mse,
            'bilinear_prediction_r2': linear_r_squared,
        }
        metrics.update(metric_statistics(geometry_base, effective_gauge))
        metrics.update(one_step_strain_statistics(jacobian, args.rho))
        metrics.update(
            product_gain_statistics(
                geometry_base,
                action_sequence,
                effective_gauge,
                args.rho,
                args.axis_angle,
            )
        )
        rows.append(metrics)

    best_prediction = min(rows, key=lambda row: row['bilinear_prediction_mse'])
    best_strain = min(rows, key=lambda row: row['one_step_log_shear_mean'])
    best_product = min(rows, key=lambda row: row['product_gain_over_rho_h_p95'])
    return {
        'config': {
            key: value
            for key, value in vars(args).items()
            if key != 'output'
        },
        'analytic_optimum_beta': -args.observation_twist,
        'base_bayes_mse_per_coordinate': 1.0 - args.rho**2,
        'best_by_bilinear_prediction_mse': best_prediction['encoder_beta'],
        'best_by_one_step_strain': best_strain['encoder_beta'],
        'best_by_horizon_product_gain_p95': best_product['encoder_beta'],
        'rows': rows,
    }


def print_summary(results: dict[str, object]) -> None:
    print(
        'beta'.rjust(7),
        'gauge'.rjust(7),
        'pred-MSE'.rjust(10),
        'pred-R2'.rjust(9),
        'metric-k95'.rjust(11),
        'strain95'.rjust(9),
        'gain1-95'.rjust(10),
        'gainH-95'.rjust(10),
    )
    rows = results['rows']
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        print(
            f"{row['encoder_beta']:.2f}".rjust(7),
            f"{row['effective_gauge']:.2f}".rjust(7),
            f"{row['bilinear_prediction_mse']:.4f}".rjust(10),
            f"{row['bilinear_prediction_r2']:.3f}".rjust(9),
            f"{row['metric_log_condition_p95']:.2f}".rjust(11),
            f"{row['one_step_log_shear_p95']:.2f}".rjust(9),
            f"{row['one_step_gain_over_rho_p95']:.2f}".rjust(10),
            f"{row['product_gain_over_rho_h_p95']:.2f}".rjust(10),
        )
    print()
    print('Analytic optimum beta:', results['analytic_optimum_beta'])
    print(
        'Best beta by prediction / strain / product:',
        results['best_by_bilinear_prediction_mse'],
        results['best_by_one_step_strain'],
        results['best_by_horizon_product_gain_p95'],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260801)
    parser.add_argument('--samples', type=int, default=100000)
    parser.add_argument('--geometry-samples', type=int, default=20000)
    parser.add_argument('--rho', type=float, default=0.85)
    parser.add_argument('--axis-angle', type=float, default=math.pi / 8.0)
    parser.add_argument('--observation-twist', type=float, default=1.25)
    parser.add_argument('--beta-min', type=float, default=-2.5)
    parser.add_argument('--beta-max', type=float, default=0.0)
    parser.add_argument('--beta-steps', type=int, default=11)
    parser.add_argument('--product-horizon', type=int, default=5)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            'docs/knowledge/dynamics_compatible_gaussian_geometry_20260801/'
            'results.json'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = scan(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print_summary(results)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
