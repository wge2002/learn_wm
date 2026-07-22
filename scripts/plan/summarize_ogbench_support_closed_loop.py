"""Summarize paired OGBench closed-loop support-intervention runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


CELLS = ('zero_k1', 'zero_k5', 'prior_k1', 'prior_k5')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for cell in CELLS:
        parser.add_argument(
            f'--{cell.replace("_", "-")}',
            nargs='+',
            required=True,
            type=Path,
        )
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=20260722)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scalar(value):
    value = np.asarray(value)
    return value.item() if value.ndim == 0 else value


def load_cell(paths: list[Path]) -> tuple[dict, list[dict]]:
    shards = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            item = {key: np.asarray(archive[key]) for key in archive.files}
        item['metadata_dict'] = json.loads(str(scalar(item['metadata'])))
        item['source'] = path
        shards.append(item)
    shards.sort(key=lambda item: int(scalar(item['seed'])))
    merged = {
        field: np.concatenate([item[field] for item in shards], axis=0)
        for field in (
            'dataset_rows',
            'episodes',
            'starts',
            'episode_successes',
            'initial_task_distance',
            'final_task_distance',
        )
    }
    merged['seed_per_state'] = np.concatenate(
        [
            np.full(len(item['dataset_rows']), int(scalar(item['seed'])))
            for item in shards
        ]
    )
    merged['policy_per_state'] = np.concatenate(
        [
            np.full(len(item['dataset_rows']), str(scalar(item['policy'])))
            for item in shards
        ]
    )
    merged['shard_elapsed_seconds'] = [
        float(scalar(item['elapsed_seconds'])) for item in shards
    ]
    return merged, shards


def assert_protocol(cells: dict[str, tuple[dict, list[dict]]]) -> None:
    reference, reference_shards = cells['zero_k1']
    seeds = [int(scalar(item['seed'])) for item in reference_shards]
    for cell, (merged, shards) in cells.items():
        actual_seeds = [int(scalar(item['seed'])) for item in shards]
        if actual_seeds != seeds:
            raise ValueError(f'{cell}: seeds {actual_seeds} != {seeds}')
        for field in ('dataset_rows', 'episodes', 'starts', 'initial_task_distance'):
            if not np.array_equal(merged[field], reference[field]):
                raise ValueError(f'{cell}: pairing field {field!r} differs')
        for shard, reference_shard in zip(shards, reference_shards, strict=True):
            metadata = shard['metadata_dict']
            reference_metadata = reference_shard['metadata_dict']
            for field in (
                'horizon',
                'receding_horizon',
                'action_block',
                'goal_offset',
                'eval_budget',
                'bf16',
                'num_samples',
                'cem_steps',
                'topk',
            ):
                if metadata[field] != reference_metadata[field]:
                    raise ValueError(f'{cell}: metadata field {field!r} differs')

    for cell in ('zero_k1', 'zero_k5'):
        for shard in cells[cell][1]:
            if shard['metadata_dict']['dataset_action_prior'] is not None:
                raise ValueError(f'{cell}: zero condition contains a prior')
    for cell in ('prior_k1', 'prior_k5'):
        for shard in cells[cell][1]:
            prior = shard['metadata_dict']['dataset_action_prior']
            if not prior or not prior.get('enabled', True):
                raise ValueError(f'{cell}: prior condition is not enabled')

    if len(np.unique(reference['dataset_rows'])) != len(reference['dataset_rows']):
        raise ValueError('Pooled eval seeds contain duplicate dataset rows')
    for cell, (merged, _) in cells.items():
        successes = merged['episode_successes'].astype(bool)
        distances = merged['final_task_distance']
        if not np.all(np.isfinite(distances)):
            raise ValueError(f'{cell}: non-finite final distance')
        if not np.array_equal(successes, distances <= 0.04 + 1e-8):
            raise ValueError(f'{cell}: binary success disagrees with final distance')


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    draws = values[
        rng.integers(0, len(values), size=(20_000, len(values)))
    ].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        'mean': float(values.mean()),
        'ci95': [float(lo), float(hi)],
        'states': int(len(values)),
    }


def exact_mcnemar(wins: int, losses: int) -> float:
    discordant = int(wins + losses)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return float(min(1.0, 2.0 * tail))


def paired_summary(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    *,
    binary: bool,
) -> dict:
    left = np.asarray(left)
    right = np.asarray(right)
    result = bootstrap(right.astype(float) - left.astype(float), rng)
    if binary:
        result.update(
            {
                'right_only': int(np.sum(right & ~left)),
                'left_only': int(np.sum(left & ~right)),
                'both': int(np.sum(left & right)),
                'neither': int(np.sum(~left & ~right)),
                'mcnemar_exact_p': exact_mcnemar(
                    int(np.sum(right & ~left)), int(np.sum(left & ~right))
                ),
            }
        )
    return result


def subset_report(
    arrays: dict[str, dict[str, np.ndarray]],
    mask: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    cells = {}
    for cell, values in arrays.items():
        cells[cell] = {
            'success': bootstrap(values['success'][mask].astype(float), rng),
            'final_distance': bootstrap(values['distance'][mask], rng),
        }

    comparisons = {}
    for condition in ('zero', 'prior'):
        comparisons[f'{condition}_k5_minus_k1_success'] = paired_summary(
            arrays[f'{condition}_k1']['success'][mask],
            arrays[f'{condition}_k5']['success'][mask],
            rng,
            binary=True,
        )
        comparisons[f'{condition}_k5_minus_k1_distance'] = paired_summary(
            arrays[f'{condition}_k1']['distance'][mask],
            arrays[f'{condition}_k5']['distance'][mask],
            rng,
            binary=False,
        )
    for model in ('k1', 'k5'):
        comparisons[f'{model}_prior_minus_zero_success'] = paired_summary(
            arrays[f'zero_{model}']['success'][mask],
            arrays[f'prior_{model}']['success'][mask],
            rng,
            binary=True,
        )
        comparisons[f'{model}_prior_minus_zero_distance'] = paired_summary(
            arrays[f'zero_{model}']['distance'][mask],
            arrays[f'prior_{model}']['distance'][mask],
            rng,
            binary=False,
        )

    success_did = (
        arrays['prior_k5']['success'].astype(float)
        - arrays['zero_k5']['success'].astype(float)
        - arrays['prior_k1']['success'].astype(float)
        + arrays['zero_k1']['success'].astype(float)
    )
    distance_did = (
        arrays['prior_k5']['distance']
        - arrays['zero_k5']['distance']
        - arrays['prior_k1']['distance']
        + arrays['zero_k1']['distance']
    )
    comparisons['success_difference_in_differences'] = bootstrap(
        success_did[mask], rng
    )
    comparisons['distance_difference_in_differences'] = bootstrap(
        distance_did[mask], rng
    )
    return {
        'states': int(np.sum(mask)),
        'cells': cells,
        'comparisons': comparisons,
    }


def main() -> None:
    args = parse_args()
    path_groups = {
        cell: getattr(args, cell) for cell in CELLS
    }
    cells = {cell: load_cell(paths) for cell, paths in path_groups.items()}
    assert_protocol(cells)
    rng = np.random.default_rng(args.seed)

    arrays = {
        cell: {
            'success': merged['episode_successes'].astype(bool),
            'distance': merged['final_task_distance'].astype(float),
        }
        for cell, (merged, _) in cells.items()
    }
    reference = cells['zero_k1'][0]
    initial_easy = reference['initial_task_distance'] <= 0.04
    all_mask = np.ones(len(initial_easy), dtype=bool)
    report = {
        'version': 1,
        'protocol': {
            'seeds': sorted(np.unique(reference['seed_per_state']).astype(int).tolist()),
            'states': int(len(all_mask)),
            'unique_rows': int(len(np.unique(reference['dataset_rows']))),
            'initial_easy': int(np.sum(initial_easy)),
            'metadata': cells['zero_k1'][1][0]['metadata_dict'],
        },
        'all': subset_report(arrays, all_mask, rng),
        'nontrivial': subset_report(arrays, ~initial_easy, rng),
        'per_seed': {},
        'inputs': {
            cell: [
                {'path': str(path.resolve()), 'sha256': sha256(path)}
                for path in paths
            ]
            for cell, paths in path_groups.items()
        },
    }
    for seed in report['protocol']['seeds']:
        mask = reference['seed_per_state'] == seed
        report['per_seed'][str(seed)] = {
            cell: {
                'success_rate': float(np.mean(values['success'][mask])),
                'successes': int(np.sum(values['success'][mask])),
                'final_distance': float(np.mean(values['distance'][mask])),
            }
            for cell, values in arrays.items()
        }

    print(
        f'Protocol: N={len(all_mask)}, seeds={report["protocol"]["seeds"]}, '
        f'initial_easy={np.sum(initial_easy)}'
    )
    print('cell       success  final_distance')
    for cell in CELLS:
        success = report['all']['cells'][cell]['success']['mean']
        distance = report['all']['cells'][cell]['final_distance']['mean']
        print(f'{cell:<10} {success:.3f}    {distance:.4f}')
    print('\nPaired success deltas (right-left, state bootstrap 95% CI)')
    for name, values in report['all']['comparisons'].items():
        if 'success' not in name:
            continue
        print(
            f'{name}: {values["mean"]:.4f} '
            f'[{values["ci95"][0]:.4f}, {values["ci95"][1]:.4f}]'
        )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(f'\nreport -> {args.out}')


if __name__ == '__main__':
    main()
