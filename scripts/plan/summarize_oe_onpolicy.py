"""Summarize matched multi-arm planner-query aggregation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    'update_cosine',
    'relative_update_error',
    'elite_overlap',
    'selected_elite_true_cost',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('arms', nargs='+', type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--topk', type=int, default=30)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open(encoding='utf-8') as handle:
        return json.load(handle)


def population_metrics(
    predicted: np.ndarray,
    true_cost: np.ndarray,
    candidates: np.ndarray,
    prev_mean: np.ndarray,
    *,
    topk: int,
) -> dict[str, float]:
    learned_indices = np.argsort(
        predicted,
        kind='stable',
    )[:topk]
    oracle_indices = np.argsort(
        true_cost,
        kind='stable',
    )[:topk]
    learned_mean = candidates[learned_indices].astype(np.float64).mean(axis=0)
    oracle_mean = candidates[oracle_indices].astype(np.float64).mean(axis=0)
    previous = prev_mean.astype(np.float64)
    learned_update = (learned_mean - previous).reshape(-1)
    oracle_update = (oracle_mean - previous).reshape(-1)
    learned_norm = float(np.linalg.norm(learned_update))
    oracle_norm = float(np.linalg.norm(oracle_update))
    cosine = float(
        np.dot(learned_update, oracle_update)
        / max(learned_norm * oracle_norm, 1e-12)
    )
    relative = float(
        np.linalg.norm(learned_update - oracle_update)
        / max(oracle_norm, 1e-12)
    )
    overlap = len(
        set(learned_indices.tolist()) & set(oracle_indices.tolist())
    ) / topk
    return {
        'update_cosine': cosine,
        'relative_update_error': relative,
        'elite_overlap': float(overlap),
        'selected_elite_true_cost': float(
            np.mean(true_cost[learned_indices])
        ),
    }


def archive_metrics(path: Path, *, topk: int) -> tuple[dict, list[int]]:
    with np.load(path, allow_pickle=False) as archive:
        candidates = np.asarray(archive['candidates'])[:, 0]
        predicted = np.asarray(archive['pred'])[:, 0, :, 0]
        true_cost = np.asarray(archive['true'])[:, 0]
        previous = np.asarray(archive['prev_mean'])[:, 0]
        rows = np.asarray(archive['rows'], dtype=np.int64).tolist()
    metrics = []
    for state_i in range(candidates.shape[0]):
        for round_i in range(candidates.shape[1]):
            metrics.append(
                population_metrics(
                    predicted[state_i, round_i],
                    true_cost[state_i, round_i],
                    candidates[state_i, round_i],
                    previous[state_i, round_i],
                    topk=topk,
                )
            )
    return (
        {
            key: float(np.mean([row[key] for row in metrics]))
            for key in METRICS
        },
        rows,
    )


def main() -> None:
    args = parse_args()
    if len(args.arms) < 2:
        raise ValueError('at least two arm directories are required')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    rows_by_round: dict[int, list[tuple[str, list[int]]]] = {}
    for arm_dir in args.arms:
        protocol = read_json(arm_dir / 'protocol.json')
        arm = str(protocol['arm'])
        rounds = int(protocol['rounds'])
        round_rows = []
        for round_i in range(1, rounds + 1):
            round_dir = arm_dir / f'round_{round_i:03d}'
            fresh_metrics, rows = archive_metrics(
                round_dir / 'fresh_rescored.npz',
                topk=args.topk,
            )
            rows_by_round.setdefault(round_i, []).append((arm, rows))
            metrics_history = read_json(
                round_dir / 'train' / 'metrics.json'
            )['history']
            by_epoch = {int(row['epoch']): row for row in metrics_history}
            if set(by_epoch) != {0, 1}:
                raise ValueError(
                    f'{arm} round {round_i}: expected epochs 0 and 1, '
                    f'got {sorted(by_epoch)}'
                )
            train_audit = read_json(round_dir / 'train' / 'audit.json')
            with np.load(
                round_dir / 'fresh.npz',
                allow_pickle=False,
            ) as fresh:
                exclusion_audit = json.loads(
                    str(np.asarray(fresh['exclusion_audit']).item())
                )
            if exclusion_audit['sampled_exclusion_overlap'] != 0:
                raise ValueError(
                    f'{arm} round {round_i}: exclusion overlap is nonzero'
                )
            round_rows.append(
                {
                    'round': round_i,
                    'fresh_pre_update': fresh_metrics,
                    'fixed_validation_pre_update': by_epoch[0]['val'],
                    'fixed_validation_post_update': by_epoch[1]['val'],
                    'replay_mse_pre_update': by_epoch[0][
                        'replay_validation_mse'
                    ],
                    'replay_mse_post_update': by_epoch[1][
                        'replay_validation_mse'
                    ],
                    'training_states': len(train_audit['train_states']),
                    'available_population_records': train_audit[
                        'available_population_records'
                    ],
                    'optimizer_updates': train_audit['updates_per_epoch'],
                    'base_policy': train_audit['base_policy'],
                    'next_policy': (
                        round_dir / 'next_policy.txt'
                    ).read_text(encoding='utf-8').strip(),
                    'sampled_rows': rows,
                    'exclusion_audit': exclusion_audit,
                }
            )

        pressure, pressure_rows = archive_metrics(
            arm_dir / 'pressure_final.npz',
            topk=args.topk,
        )
        summaries.append(
            {
                'arm': arm,
                'protocol': protocol,
                'rounds': round_rows,
                'pressure_final': pressure,
                'pressure_rows': pressure_rows,
                'final_policy': (
                    arm_dir / 'final_policy.txt'
                ).read_text(encoding='utf-8').strip(),
            }
        )

    row_pairing = {}
    for round_i, entries in sorted(rows_by_round.items()):
        reference_arm, reference = entries[0]
        matched = {
            arm: rows == reference for arm, rows in entries
        }
        if not all(matched.values()):
            raise ValueError(
                f'round {round_i} dataset rows are not paired: {matched}'
            )
        row_pairing[str(round_i)] = {
            'reference_arm': reference_arm,
            'rows': reference,
            'matched': matched,
        }

    result = {
        'version': 1,
        'topk': args.topk,
        'row_pairing': row_pairing,
        'arms': summaries,
    }
    with (args.out_dir / 'summary.json').open(
        'w',
        encoding='utf-8',
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write('\n')

    lines = [
        '# On-policy planner-query aggregation',
        '',
        f'- Arms: {len(summaries)}',
        f'- Rounds: {len(row_pairing)}',
        f'- Fresh states per matched round: '
        f'{len(next(iter(row_pairing.values()))["rows"])}',
        '- Dataset rows are exactly paired across arms: yes',
        '- Fixed validation and H8 pressure populations are never trained on.',
        '',
        '## Fresh-query pre-update trajectory',
        '',
        'These metrics measure each current checkpoint before it trains on the '
        'newly collected round. Higher cosine/overlap and lower relative error '
        'or selected true cost are better.',
        '',
        '| arm | round | cosine | relative error | overlap | selected true |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for summary in summaries:
        for row in summary['rounds']:
            metric = row['fresh_pre_update']
            lines.append(
                f'| {summary["arm"]} | {row["round"]} '
                f'| {metric["update_cosine"]:.3f} '
                f'| {metric["relative_update_error"]:.3f} '
                f'| {metric["elite_overlap"]:.3f} '
                f'| {metric["selected_elite_true_cost"]:.2f} |'
            )

    lines.extend(
        [
            '',
            '## Fixed-validation trajectory',
            '',
            '| arm | round | pre cosine | post cosine | pre relative '
            '| post relative | pre overlap | post overlap | replay MSE post |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
        ]
    )
    for summary in summaries:
        for row in summary['rounds']:
            before = row['fixed_validation_pre_update']
            after = row['fixed_validation_post_update']
            lines.append(
                f'| {summary["arm"]} | {row["round"]} '
                f'| {before["update_cosine"]:.3f} '
                f'| {after["update_cosine"]:.3f} '
                f'| {before["relative_update_error"]:.3f} '
                f'| {after["relative_update_error"]:.3f} '
                f'| {before["elite_overlap"]:.3f} '
                f'| {after["elite_overlap"]:.3f} '
                f'| {row["replay_mse_post_update"]:.5f} |'
            )

    lines.extend(
        [
            '',
            '## Final H8/off60 pressure test',
            '',
            '| arm | cosine | relative error | overlap | selected true |',
            '|---|---:|---:|---:|---:|',
        ]
    )
    for summary in summaries:
        metric = summary['pressure_final']
        lines.append(
            f'| {summary["arm"]} '
            f'| {metric["update_cosine"]:.3f} '
            f'| {metric["relative_update_error"]:.3f} '
            f'| {metric["elite_overlap"]:.3f} '
            f'| {metric["selected_elite_true_cost"]:.2f} |'
        )

    (args.out_dir / 'report.md').write_text(
        '\n'.join(lines) + '\n',
        encoding='utf-8',
    )
    print(f'summary -> {args.out_dir}')


if __name__ == '__main__':
    main()
