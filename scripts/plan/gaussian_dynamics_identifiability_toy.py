#!/usr/bin/env python3
"""Audit what a Gaussian marginal does and does not identify.

The script has two deliberately separate experiments.

1. ``distribution_sketch`` compares a true 192-D Gaussian with a linear
   3-D sheet, a nonlinear 3-D Fourier sheet, and a 12-bit codebook.  It
   evaluates the exact finite-batch SIGReg statistic used in this repository.
2. ``dynamics`` constructs several transition laws that all preserve an
   exact 2-D standard Gaussian marginal, while having very different action
   sensitivity, predictability, and local Jacobian gain.
3. ``horizon_allocation`` solves a two-factor Gaussian model exactly and
   measures when a longer prediction horizon increasingly favors a persistent
   shortcut over a less persistent task factor.

Only NumPy is required.  The output is deterministic for a fixed seed and is
written as JSON so the reported numbers can be regenerated rather than copied
by hand.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np


Array = np.ndarray
Sampler = Callable[[int, np.random.Generator], Array]


def sigreg_statistic(
    samples: Array,
    rng: np.random.Generator,
    *,
    knots: int,
    num_proj: int,
) -> float:
    """Match ``stable_worldmodel.wm.loss.SIGReg`` in NumPy.

    ``samples`` has shape ``(batch, dimension)``.  The repository's torch
    implementation accepts an extra time dimension, but its statistic is the
    same for a single time slice.
    """

    batch, dimension = samples.shape
    directions = rng.normal(size=(dimension, num_proj))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    projected = samples @ directions

    t = np.linspace(0.0, 3.0, knots)
    dt = 3.0 / (knots - 1)
    trapezoid = np.full(knots, 2.0 * dt)
    trapezoid[[0, -1]] = dt
    phi = np.exp(-(t**2) / 2.0)
    weights = trapezoid * phi

    statistic = np.zeros(num_proj)
    for knot, target, weight in zip(t, phi, weights, strict=True):
        phase = projected * knot
        real_error = np.cos(phase).mean(axis=0) - target
        imag_error = np.sin(phase).mean(axis=0)
        statistic += (real_error**2 + imag_error**2) * weight
    return float((statistic * batch).mean())


def effective_rank(samples: Array) -> float:
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    return float(eigenvalues.sum() ** 2 / np.square(eigenvalues).sum())


def marginal_moments(samples: Array) -> tuple[float, float]:
    centered = samples - samples.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / (len(centered) - 1)
    identity = np.eye(samples.shape[1])
    mean_norm = np.linalg.norm(samples.mean(axis=0))
    covariance_error = np.linalg.norm(covariance - identity, ord='fro')
    covariance_error /= np.linalg.norm(identity, ord='fro')
    return float(mean_norm), float(covariance_error)


def make_distribution_samplers(
    dimension: int,
    key_bits: int,
    seed: int,
) -> dict[str, tuple[Sampler, int, str]]:
    construction_rng = np.random.default_rng(seed)
    intrinsic_dimension = 3

    basis, _ = np.linalg.qr(
        construction_rng.normal(size=(dimension, intrinsic_dimension))
    )

    frequency_limit = 47
    frequencies: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    while len(frequencies) < dimension:
        frequency = tuple(
            int(value)
            for value in construction_rng.integers(
                -frequency_limit,
                frequency_limit + 1,
                size=intrinsic_dimension,
            )
        )
        if frequency == (0, 0, 0):
            continue
        first_nonzero = next(value for value in frequency if value != 0)
        canonical = frequency
        if first_nonzero < 0:
            canonical = tuple(-value for value in frequency)
        if canonical in seen:
            continue
        seen.add(canonical)
        frequencies.append(frequency)
    frequency_array = np.asarray(frequencies, dtype=float)
    phases = construction_rng.uniform(0.0, 1.0, size=dimension)

    codebook_size = 2**key_bits
    codebook = construction_rng.normal(size=(codebook_size, dimension))
    codebook -= codebook.mean(axis=0, keepdims=True)
    covariance = codebook.T @ codebook / codebook_size
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening = eigenvectors @ np.diag(eigenvalues**-0.5) @ eigenvectors.T
    codebook = codebook @ whitening

    def gaussian(count: int, rng: np.random.Generator) -> Array:
        return rng.normal(size=(count, dimension))

    def linear_sheet(count: int, rng: np.random.Generator) -> Array:
        coordinates = rng.normal(size=(count, intrinsic_dimension))
        scale = math.sqrt(dimension / intrinsic_dimension)
        return scale * coordinates @ basis.T

    def fourier_sheet(count: int, rng: np.random.Generator) -> Array:
        coordinates = rng.uniform(
            0.0,
            1.0,
            size=(count, intrinsic_dimension),
        )
        angles = 2.0 * math.pi * (
            coordinates @ frequency_array.T + phases
        )
        return math.sqrt(2.0) * np.cos(angles)

    def discrete_codebook(count: int, rng: np.random.Generator) -> Array:
        keys = rng.integers(0, codebook_size, size=count)
        return codebook[keys]

    return {
        'true_gaussian': (gaussian, dimension, 'full-dimensional density'),
        'linear_3d_sheet': (
            linear_sheet,
            intrinsic_dimension,
            'rank-3 linear support',
        ),
        'fourier_3d_sheet': (
            fourier_sheet,
            intrinsic_dimension,
            'smooth nonlinear image of a 3-torus',
        ),
        f'codebook_{key_bits}bit': (
            discrete_codebook,
            0,
            f'{codebook_size} atoms',
        ),
    }


def audit_distribution_sketch(
    args: argparse.Namespace,
) -> dict[str, dict[str, float | int | str]]:
    samplers = make_distribution_samplers(
        args.latent_dim,
        args.key_bits,
        args.seed + 11,
    )
    results: dict[str, dict[str, float | int | str]] = {}
    sequence = np.random.SeedSequence(args.seed + 101)
    children = iter(sequence.spawn(len(samplers) * (args.sigreg_repeats + 1)))

    for name, (sampler, support_dimension, support_description) in samplers.items():
        scores = []
        for _ in range(args.sigreg_repeats):
            rng = np.random.default_rng(next(children))
            batch = sampler(args.batch_size, rng)
            scores.append(
                sigreg_statistic(
                    batch,
                    rng,
                    knots=args.knots,
                    num_proj=args.num_proj,
                )
            )

        audit_rng = np.random.default_rng(next(children))
        audit_samples = sampler(args.audit_samples, audit_rng)
        mean_norm, covariance_error = marginal_moments(audit_samples)
        results[name] = {
            'sigreg_mean': float(np.mean(scores)),
            'sigreg_std': float(np.std(scores, ddof=1)),
            'mean_norm': mean_norm,
            'covariance_relative_frobenius_error': covariance_error,
            'covariance_participation_rank': effective_rank(audit_samples),
            'support_dimension_upper_bound': support_dimension,
            'support_description': support_description,
        }
    return results


def rotate(z: Array, angle: Array) -> Array:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    x = cosine * z[:, 0] - sine * z[:, 1]
    y = sine * z[:, 0] + cosine * z[:, 1]
    return np.column_stack((x, y))


def fit_r_squared(features: Array, targets: Array) -> float:
    design = np.column_stack((features, np.ones(len(features))))
    coefficients, *_ = np.linalg.lstsq(design, targets, rcond=None)
    prediction = design @ coefficients
    residual = np.square(targets - prediction).sum()
    total = np.square(targets - targets.mean(axis=0)).sum()
    return float(1.0 - residual / total)


def shear_singular_value(shear: Array) -> Array:
    square = shear**2
    return np.sqrt((square + 2.0 + np.abs(shear) * np.sqrt(square + 4.0)) / 2.0)


def audit_dynamics(
    args: argparse.Namespace,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(args.seed + 1001)
    z = rng.normal(size=(args.dynamics_samples, 2))
    action = rng.choice(np.array([-1.0, 1.0]), size=args.dynamics_samples)
    noise = rng.normal(size=z.shape)

    rotation_angle = math.pi / 3.0
    twist_strength = 1.25
    rho = 0.85

    radius_squared = np.square(z).sum(axis=1)
    transitions = {
        'identity': z.copy(),
        'controlled_rotation': rotate(z, action * rotation_angle),
        'controlled_radial_twist': rotate(
            z,
            action * twist_strength * radius_squared,
        ),
        'ou_rho_0.85': rho * z + math.sqrt(1.0 - rho**2) * noise,
        'independent_refresh': noise.copy(),
    }

    oracle_mse = {
        'identity': 0.0,
        'controlled_rotation': 0.0,
        'controlled_radial_twist': 0.0,
        'ou_rho_0.85': 1.0 - rho**2,
        'independent_refresh': 1.0,
    }

    paired_action_effect = {
        'identity': 0.0,
        'controlled_rotation': float(
            np.square(
                rotate(z, np.full(len(z), rotation_angle))
                - rotate(z, np.full(len(z), -rotation_angle))
            ).mean()
        ),
        'controlled_radial_twist': float(
            np.square(
                rotate(z, twist_strength * radius_squared)
                - rotate(z, -twist_strength * radius_squared)
            ).mean()
        ),
        'ou_rho_0.85': 0.0,
        'independent_refresh': 0.0,
    }

    local_gain = {
        'identity': np.ones(len(z)),
        'controlled_rotation': np.ones(len(z)),
        'controlled_radial_twist': shear_singular_value(
            2.0 * twist_strength * radius_squared
        ),
        'ou_rho_0.85': np.full(len(z), rho),
        'independent_refresh': np.zeros(len(z)),
    }

    results: dict[str, dict[str, float]] = {}
    for index, (name, z_next) in enumerate(transitions.items()):
        mean_norm, covariance_error = marginal_moments(z_next)
        state_features = z
        bilinear_features = np.column_stack((z, action[:, None] * z))
        score_sequence = np.random.SeedSequence(
            args.seed + 2001 + index
        ).spawn(args.dynamics_sigreg_repeats)
        marginal_scores = []
        for score_seed in score_sequence:
            score_rng = np.random.default_rng(score_seed)
            score_indices = score_rng.choice(
                len(z_next),
                size=args.batch_size,
                replace=False,
            )
            marginal_scores.append(
                sigreg_statistic(
                    z_next[score_indices],
                    score_rng,
                    knots=args.knots,
                    num_proj=args.num_proj,
                )
            )
        gains = local_gain[name]
        results[name] = {
            'next_marginal_sigreg_mean': float(
                np.mean(marginal_scores)
            ),
            'next_marginal_sigreg_std': float(
                np.std(marginal_scores, ddof=1)
            ),
            'next_mean_norm': mean_norm,
            'next_covariance_relative_frobenius_error': covariance_error,
            'copy_predictor_mse_per_coordinate': float(
                np.square(z_next - z).mean()
            ),
            'state_only_linear_r2': fit_r_squared(state_features, z_next),
            'action_bilinear_linear_r2': fit_r_squared(
                bilinear_features,
                z_next,
            ),
            'oracle_bayes_mse_per_coordinate': oracle_mse[name],
            'paired_action_effect_per_coordinate': paired_action_effect[name],
            'local_conditional_mean_gain_p50': float(
                np.quantile(gains, 0.50)
            ),
            'local_conditional_mean_gain_p95': float(
                np.quantile(gains, 0.95)
            ),
            'local_conditional_mean_gain_p99': float(
                np.quantile(gains, 0.99)
            ),
        }
    return results


def average_k_step_bayes_risk(rho: float, horizon: int) -> float:
    steps = np.arange(1, horizon + 1, dtype=float)
    return float(np.mean(1.0 - rho ** (2.0 * steps)))


def audit_horizon_allocation() -> dict[str, object]:
    """Exact risk comparison for two independent Gaussian AR(1) factors.

    Let ``s`` be a shortcut and ``x`` a task factor, both marginally standard
    Gaussian.  Every scalar mixture

        z(theta) = cos(theta) s + sin(theta) x

    is also exactly standard Gaussian.  For one pure factor with lag
    correlation ``rho``, the Bayes k-step MSE is ``1 - rho**(2*k)``.  The
    reported advantage is task risk minus shortcut risk, so positive values
    mean that prediction loss favors the shortcut.
    """

    horizons = [1, 2, 3, 5, 10]
    task_rho = 0.85
    shortcut_rhos = [0.50, 0.85, 0.90, 0.99, 1.00]
    rows = []
    for shortcut_rho in shortcut_rhos:
        advantages = {}
        for horizon in horizons:
            task_risk = average_k_step_bayes_risk(task_rho, horizon)
            shortcut_risk = average_k_step_bayes_risk(
                shortcut_rho,
                horizon,
            )
            advantages[str(horizon)] = task_risk - shortcut_risk
        rows.append(
            {
                'shortcut_rho': shortcut_rho,
                'shortcut_advantage_by_horizon': advantages,
            }
        )
    return {
        'task_rho': task_rho,
        'horizons': horizons,
        'definition': (
            'task Bayes risk minus shortcut Bayes risk; positive means '
            'prediction favors the shortcut'
        ),
        'rows': rows,
    }


def print_summary(results: dict[str, object]) -> None:
    print('Distribution sketch audit')
    print(
        'name'.ljust(24),
        'SIGReg'.rjust(10),
        'cov-rank'.rjust(10),
        'support-dim'.rjust(12),
    )
    distributions = results['distribution_sketch']
    assert isinstance(distributions, dict)
    for name, raw_metrics in distributions.items():
        metrics = dict(raw_metrics)
        print(
            name.ljust(24),
            f"{metrics['sigreg_mean']:.3f}".rjust(10),
            f"{metrics['covariance_participation_rank']:.1f}".rjust(10),
            str(metrics['support_dimension_upper_bound']).rjust(12),
        )

    print('\nSame Gaussian marginal, different dynamics')
    print(
        'name'.ljust(26),
        'SIGReg'.rjust(9),
        'copy-MSE'.rjust(10),
        'bilin-R2'.rjust(10),
        'gain-p95'.rjust(10),
    )
    dynamics = results['dynamics']
    assert isinstance(dynamics, dict)
    for name, raw_metrics in dynamics.items():
        metrics = dict(raw_metrics)
        print(
            name.ljust(26),
            f"{metrics['next_marginal_sigreg_mean']:.3f}".rjust(9),
            f"{metrics['copy_predictor_mse_per_coordinate']:.3f}".rjust(10),
            f"{metrics['action_bilinear_linear_r2']:.3f}".rjust(10),
            f"{metrics['local_conditional_mean_gain_p95']:.2f}".rjust(10),
        )

    print('\nHorizon allocation: shortcut advantage over task rho=0.85')
    allocation = results['horizon_allocation']
    assert isinstance(allocation, dict)
    horizons = allocation['horizons']
    assert isinstance(horizons, list)
    print('shortcut-rho'.ljust(14), *(f'K={k}'.rjust(9) for k in horizons))
    rows = allocation['rows']
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        advantages = row['shortcut_advantage_by_horizon']
        assert isinstance(advantages, dict)
        print(
            f"{row['shortcut_rho']:.2f}".ljust(14),
            *(f"{advantages[str(k)]:.3f}".rjust(9) for k in horizons),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20260801)
    parser.add_argument('--latent-dim', type=int, default=192)
    parser.add_argument('--key-bits', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--audit-samples', type=int, default=8192)
    parser.add_argument('--dynamics-samples', type=int, default=100000)
    parser.add_argument('--sigreg-repeats', type=int, default=6)
    parser.add_argument('--dynamics-sigreg-repeats', type=int, default=32)
    parser.add_argument('--knots', type=int, default=17)
    parser.add_argument('--num-proj', type=int, default=1024)
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(
            'docs/knowledge/gaussian_dynamics_identifiability_20260801/'
            'results.json'
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        'config': {
            key: value
            for key, value in vars(args).items()
            if key != 'output'
        },
        'distribution_sketch': audit_distribution_sketch(args),
        'dynamics': audit_dynamics(args),
        'horizon_allocation': audit_horizon_allocation(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print_summary(results)
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
