#!/usr/bin/env python3
"""Summarize multi-seed paired controlled-metric checkpoint audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TRAJECTORY_METRICS = {
    'pencil.logdet_I_plus_pencil': ('pencil', 'logdet_I_plus_pencil'),
    'pencil.trace_pencil': ('pencil', 'trace_pencil'),
    'horizon.log_shear_rms': ('horizon', 'log_shear_rms'),
    'horizon.log_scale_mean': ('horizon', 'log_scale_mean'),
    'horizon.max_gain': ('horizon', 'max_gain'),
    'action_to_residual_trace_ratio': ('action_to_residual_trace_ratio',),
    'action_energy_trace': ('action_energy_trace',),
    'residual_energy_trace': ('residual_energy_trace',),
}
SUFFICIENCY_PROBES = (
    'agent_xy',
    'block_xy',
    'block_angle_sincos',
    'agent_velocity',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--protocol-proof', type=Path, default=None)
    parser.add_argument('--decision-epoch', type=int, default=30)
    parser.add_argument('--min-pairs', type=int, default=3)
    parser.add_argument('--bootstraps', type=int, default=20_000)
    parser.add_argument('--seed', type=int, default=20260810)
    parser.add_argument('--pose-tolerance', type=float, default=-0.05)
    parser.add_argument('--motion-tolerance', type=float, default=-0.03)
    parser.add_argument('--g1-linear-r2', type=float, default=0.75)
    parser.add_argument('--g1-cycle-r2', type=float, default=0.90)
    parser.add_argument('--g1-nonorthogonal-gap', type=float, default=0.10)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--output-md', type=Path, required=True)
    return parser.parse_args()


def nested(row: dict[str, object], path: tuple[str, ...]) -> float:
    value: object = row
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


def bank_signature(audit: dict[str, object]) -> str:
    metadata = audit['metadata']
    config = audit['config']
    payload = {
        'config': {
            key: config[key]
            for key in (
                'num_samples',
                'jacobian_samples',
                'history',
                'horizon',
                'frameskip',
                'ridge_alpha',
                'block_motion_threshold',
                'seed',
                'dataset',
            )
        },
        'metadata': {
            key: metadata[key]
            for key in (
                'episodes',
                'starts',
                'train_indices',
                'test_indices',
                'jacobian_indices',
                'action_mean',
                'action_std',
            )
        },
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(',', ':')
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def paired_trajectory_deltas(
    audit: dict[str, object],
) -> dict[str, np.ndarray]:
    models = audit['models']
    reference = {
        int(row['sample_index']): row
        for row in models['K1']['controlled_metric']['per_trajectory']
    }
    candidate = {
        int(row['sample_index']): row
        for row in models['K5']['controlled_metric']['per_trajectory']
    }
    if reference.keys() != candidate.keys():
        raise ValueError('K1/K5 trajectory indices are not exactly paired')
    indices = sorted(reference)
    return {
        metric: np.asarray(
            [
                nested(candidate[index], path) - nested(reference[index], path)
                for index in indices
            ],
            dtype=np.float64,
        )
        for metric, path in TRAJECTORY_METRICS.items()
    }


def sufficiency_deltas(audit: dict[str, object]) -> dict[str, float | None]:
    models = audit['models']
    result: dict[str, float | None] = {}
    for probe in SUFFICIENCY_PROBES:
        result[probe] = float(
            models['K5']['sufficiency'][probe]['r2_uniform']
            - models['K1']['sufficiency'][probe]['r2_uniform']
        )
    k1_motion = models['K1']['sufficiency']['block_motion_probe']
    k5_motion = models['K5']['sufficiency']['block_motion_probe']
    if 'roc_auc' in k1_motion and 'roc_auc' in k5_motion:
        result['block_motion_roc_auc'] = float(
            k5_motion['roc_auc'] - k1_motion['roc_auc']
        )
    else:
        result['block_motion_roc_auc'] = None
    return result


def g1_summary(audit: dict[str, object]) -> dict[str, float]:
    maps = audit['g1_bidirectional_maps']['K1_vs_K5']
    forward_linear = float(maps['forward']['linear']['r2_uniform'])
    reverse_linear = float(maps['reverse']['linear']['r2_uniform'])
    forward_orthogonal = float(maps['forward']['orthogonal']['r2'])
    reverse_orthogonal = float(maps['reverse']['orthogonal']['r2'])
    return {
        'forward_linear_r2': forward_linear,
        'reverse_linear_r2': reverse_linear,
        'source_cycle_r2': float(
            maps['cycle']['source_to_target_to_source']['r2_uniform']
        ),
        'target_cycle_r2': float(
            maps['cycle']['target_to_source_to_target']['r2_uniform']
        ),
        'forward_linear_minus_orthogonal_r2': (
            forward_linear - forward_orthogonal
        ),
        'reverse_linear_minus_orthogonal_r2': (
            reverse_linear - reverse_orthogonal
        ),
    }


def hierarchical_summary(
    cells: list[np.ndarray],
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    pair_means = np.asarray([cell.mean() for cell in cells], dtype=np.float64)
    pair_count = len(cells)
    selected_pairs = rng.integers(
        0, pair_count, size=(bootstraps, pair_count)
    )
    draws = np.empty_like(selected_pairs, dtype=np.float64)
    for slot in range(pair_count):
        for source, values in enumerate(cells):
            mask = selected_pairs[:, slot] == source
            count = int(mask.sum())
            if count:
                selected_rows = rng.integers(
                    0, len(values), size=(count, len(values))
                )
                draws[mask, slot] = values[selected_rows].mean(axis=1)
    bootstrap = draws.mean(axis=1)
    return {
        'mean_delta_k5_minus_k1': float(pair_means.mean()),
        'hierarchical_bootstrap_ci95': np.percentile(
            bootstrap, [2.5, 97.5]
        ).tolist(),
        'pair_mean_deltas': pair_means.tolist(),
        'all_pairs_positive': bool(np.all(pair_means > 0)),
        'all_pairs_negative': bool(np.all(pair_means < 0)),
    }


def seed_summary(
    values: list[float],
    *,
    bootstraps: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    selected = rng.integers(0, len(array), size=(bootstraps, len(array)))
    means = array[selected].mean(axis=1)
    return {
        'mean_delta_k5_minus_k1': float(array.mean()),
        'pair_bootstrap_ci95': np.percentile(means, [2.5, 97.5]).tolist(),
        'pair_deltas': array.tolist(),
        'minimum_pair_delta': float(array.min()),
    }


def load_cells(paths: list[Path]) -> tuple[dict[int, list[dict]], str]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    signatures = set()
    identities = set()
    for path in paths:
        audit = json.loads(path.read_text())
        if (
            audit.get('reference') != 'K1'
            or set(audit['models']) != {'K1', 'K5'}
        ):
            raise ValueError(
                f'{path}: expected exactly K1/K5 with K1 reference'
            )
        config = audit['config']
        pair_id = config.get('pair_id')
        training_seed = config.get('training_seed')
        epoch = config.get('checkpoint_epoch')
        if pair_id is None or training_seed is None or epoch is None:
            raise ValueError(f'{path}: missing pair/seed/epoch audit metadata')
        identity = (int(epoch), str(pair_id))
        if identity in identities:
            raise ValueError(f'duplicate audit cell {identity}')
        identities.add(identity)
        signature = bank_signature(audit)
        signatures.add(signature)
        grouped[int(epoch)].append(
            {
                'path': str(path.resolve()),
                'pair_id': str(pair_id),
                'training_seed': int(training_seed),
                'trajectory_deltas': paired_trajectory_deltas(audit),
                'sufficiency_deltas': sufficiency_deltas(audit),
                'g1': g1_summary(audit),
                'checkpoints': {
                    label: str(
                        Path(
                            audit['models'][label]['checkpoint']
                        ).resolve()
                    )
                    for label in ('K1', 'K5')
                },
                'bank_signature': signature,
            }
        )
    if len(signatures) != 1:
        raise ValueError('audit files do not use one identical physical bank')
    for cells in grouped.values():
        cells.sort(key=lambda cell: cell['training_seed'])
    return dict(sorted(grouped.items())), signatures.pop()


def summarize_epoch(
    cells: list[dict], args: argparse.Namespace, rng: np.random.Generator
) -> dict[str, object]:
    metrics = {
        metric: hierarchical_summary(
            [cell['trajectory_deltas'][metric] for cell in cells],
            bootstraps=args.bootstraps,
            rng=rng,
        )
        for metric in TRAJECTORY_METRICS
    }
    sufficiency = {}
    for probe in (*SUFFICIENCY_PROBES, 'block_motion_roc_auc'):
        values = [cell['sufficiency_deltas'][probe] for cell in cells]
        sufficiency[probe] = (
            None
            if any(value is None for value in values)
            else seed_summary(
                [float(value) for value in values],
                bootstraps=args.bootstraps,
                rng=rng,
            )
        )

    g1_per_pair = []
    for cell in cells:
        values = cell['g1']
        passed = (
            min(values['forward_linear_r2'], values['reverse_linear_r2'])
            >= args.g1_linear_r2
            and min(values['source_cycle_r2'], values['target_cycle_r2'])
            >= args.g1_cycle_r2
            and min(
                values['forward_linear_minus_orthogonal_r2'],
                values['reverse_linear_minus_orthogonal_r2'],
            )
            >= args.g1_nonorthogonal_gap
        )
        g1_per_pair.append(
            {'pair_id': cell['pair_id'], **values, 'pass': bool(passed)}
        )

    return {
        'num_pairs': len(cells),
        'pair_ids': [cell['pair_id'] for cell in cells],
        'training_seeds': [cell['training_seed'] for cell in cells],
        'inputs': [cell['path'] for cell in cells],
        'trajectory_metrics': metrics,
        'sufficiency': sufficiency,
        'g1': {
            'thresholds': {
                'linear_r2_min': args.g1_linear_r2,
                'cycle_r2_min': args.g1_cycle_r2,
                'linear_minus_orthogonal_r2_min': args.g1_nonorthogonal_gap,
            },
            'per_pair': g1_per_pair,
            'all_pairs_pass': all(row['pass'] for row in g1_per_pair),
        },
    }


def decision(
    epoch: dict[str, object],
    *,
    proof_ok: bool,
    args: argparse.Namespace,
) -> dict[str, object]:
    metrics = epoch['trajectory_metrics']
    primary = metrics['pencil.logdet_I_plus_pencil']
    shear = metrics['horizon.log_shear_rms']
    gates: dict[str, dict[str, object]] = {
        'protocol_pairing_proof': {
            'pass': proof_ok,
            'rule': 'machine-checked init/config/split/order/update parity',
        },
        'minimum_independent_pairs': {
            'pass': epoch['num_pairs'] >= args.min_pairs,
            'rule': f'at least {args.min_pairs} training-seed pairs',
        },
        'g2_generalized_pencil': {
            'pass': (
                primary['hierarchical_bootstrap_ci95'][0] > 0
                and primary['all_pairs_positive']
            ),
            'rule': 'K5-K1 logdet(I+Wr^-1 Wu) CI low > 0 and every seed > 0',
        },
        'gate_a_shear': {
            'pass': (
                shear['hierarchical_bootstrap_ci95'][1] < 0
                and shear['all_pairs_negative']
            ),
            'rule': 'K5-K1 horizon log-shear CI high < 0 and every seed < 0',
        },
        'g1_bidirectional_nonorthogonal_linear_map': {
            'pass': epoch['g1']['all_pairs_pass'],
            'rule': (
                'every seed clears bidirectional linear/cycle/'
                'non-orthogonal thresholds'
            ),
        },
    }

    sufficiency_ok = True
    sufficiency_details = {}
    for probe in SUFFICIENCY_PROBES:
        summary = epoch['sufficiency'][probe]
        passed = (
            summary is not None
            and summary['pair_bootstrap_ci95'][0] >= args.pose_tolerance
            and summary['minimum_pair_delta'] >= args.pose_tolerance
        )
        sufficiency_ok &= passed
        sufficiency_details[probe] = passed
    motion = epoch['sufficiency']['block_motion_roc_auc']
    motion_ok = (
        motion is not None
        and motion['pair_bootstrap_ci95'][0] >= args.motion_tolerance
        and motion['minimum_pair_delta'] >= args.motion_tolerance
    )
    sufficiency_ok &= motion_ok
    sufficiency_details['block_motion_roc_auc'] = motion_ok
    gates['sufficiency_noninferiority'] = {
        'pass': bool(sufficiency_ok),
        'rule': (
            f'pose R2 delta >= {args.pose_tolerance} and motion AUC delta '
            f'>= {args.motion_tolerance}, both CI and every seed'
        ),
        'components': sufficiency_details,
    }

    structural = (
        gates['protocol_pairing_proof']['pass']
        and gates['minimum_independent_pairs']['pass']
    )
    all_scientific = all(gate['pass'] for gate in gates.values())
    if not structural:
        status = 'INCOMPLETE'
    elif all_scientific:
        status = 'PROVISIONAL_PASS'
    else:
        status = 'KILL_OR_DOWNGRADE'
    return {
        'status': status,
        'gates': gates,
        'claim_scope': (
            'PROVISIONAL_PASS is not the intrinsic CP_H certificate: exact '
            'contact labels, physical-perturbation W_0, and the nonlinear G1 '
            'map remain required.'
        ),
    }


def fmt(value: float) -> str:
    return f'{value:+.4g}'


def render_markdown(output: dict[str, object]) -> str:
    lines = [
        '# Controlled-metric paired summary',
        '',
        f'- Decision epoch: `{output["decision_epoch"]}`',
        f'- Status: **{output["decision"]["status"]}**',
        f'- Shared physical-bank SHA-256: `{output["bank_sha256"]}`',
        f'- Pairing proof: `{output["protocol_proof_status"]}`',
        '',
        '## Checkpoint trajectory',
        '',
        (
            '| epoch | pairs | Δ pencil logdet | 95% CI | '
            'Δ log-shear | 95% CI |'
        ),
        '| ---: | ---: | ---: | --- | ---: | --- |',
    ]
    for epoch, summary in output['epochs'].items():
        primary = summary['trajectory_metrics']['pencil.logdet_I_plus_pencil']
        shear = summary['trajectory_metrics']['horizon.log_shear_rms']
        lines.append(
            f'| {epoch} | {summary["num_pairs"]} | '
            f'{fmt(primary["mean_delta_k5_minus_k1"])} | '
            f'[{fmt(primary["hierarchical_bootstrap_ci95"][0])}, '
            f'{fmt(primary["hierarchical_bootstrap_ci95"][1])}] | '
            f'{fmt(shear["mean_delta_k5_minus_k1"])} | '
            f'[{fmt(shear["hierarchical_bootstrap_ci95"][0])}, '
            f'{fmt(shear["hierarchical_bootstrap_ci95"][1])}] |'
        )
    lines.extend(
        [
            '',
            '## Preregistered decision gates',
            '',
            '| gate | pass | rule |',
            '| --- | :---: | --- |',
        ]
    )
    for name, gate in output['decision']['gates'].items():
        lines.append(
            f'| `{name}` | {"yes" if gate["pass"] else "no"} | '
            f'{gate["rule"]} |'
        )
    lines.extend(
        [
            '',
            '## Interpretation boundary',
            '',
            output['decision']['claim_scope'],
            '',
            'All deltas are `K5 - K1`. Training seeds are the independent '
            'units; the controlled-metric interval is hierarchical (seed pair, '
            'then matched physical trajectory).',
            '',
        ]
    )
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    if args.bootstraps < 1000:
        raise ValueError('--bootstraps must be at least 1000')
    grouped, signature = load_cells(args.input)
    if args.decision_epoch not in grouped:
        raise ValueError(f'no audits for decision epoch {args.decision_epoch}')

    proof = None
    proof_ok = False
    if args.protocol_proof is not None:
        proof = json.loads(args.protocol_proof.read_text())
        expected_seeds = sorted(
            cell['training_seed'] for cell in grouped[args.decision_epoch]
        )
        proof_seeds = sorted(pair['training_seed'] for pair in proof['pairs'])
        proof_pairs = {pair['pair_id']: pair for pair in proof['pairs']}
        checkpoint_match = all(
            cell['pair_id'] in proof_pairs
            and all(
                cell['checkpoints'][label]
                == str(
                    Path(
                        proof_pairs[cell['pair_id']]['arms'][label.lower()][
                            'checkpoint'
                        ]
                    ).resolve()
                )
                for label in ('K1', 'K5')
            )
            for cell in grouped[args.decision_epoch]
        )
        proof_ok = (
            proof.get('status') == 'PASS'
            and proof.get('epochs') == args.decision_epoch
            and proof_seeds == expected_seeds
            and checkpoint_match
        )

    rng = np.random.default_rng(args.seed)
    epochs = {
        str(epoch): summarize_epoch(cells, args, rng)
        for epoch, cells in grouped.items()
    }
    final_epoch = epochs[str(args.decision_epoch)]
    final_decision = decision(final_epoch, proof_ok=proof_ok, args=args)
    output = {
        'config': {
            key: (
                [str(path) for path in value]
                if key == 'input'
                else str(value)
                if isinstance(value, Path)
                else value
            )
            for key, value in vars(args).items()
        },
        'bank_sha256': signature,
        'protocol_proof_status': (
            proof.get('status') if proof is not None else 'NOT_PROVIDED'
        ),
        'decision_epoch': args.decision_epoch,
        'epochs': epochs,
        'decision': final_decision,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, allow_nan=False) + '\n'
    )
    args.output_md.write_text(render_markdown(output))
    print(
        f'[paired-summary] {final_decision["status"]}: '
        f'wrote {args.output_json} and {args.output_md}'
    )


if __name__ == '__main__':
    main()
