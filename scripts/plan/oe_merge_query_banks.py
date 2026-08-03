"""Merge disjoint CEM planner-query archives along the state axis.

The merger is intentionally strict: all planner/model metadata must match,
every source must be finite, and dataset rows must be globally disjoint.
Horizon cells are therefore merged separately and can later be used as
training and pressure-test banks without silently padding action sequences.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


STATE_KEYS = (
    'rows',
    'episodes',
    'starts',
    'initial_state',
    'goal_state',
    'candidates',
    'candidate_indices',
    'pred',
    'true',
    'true_pos_l2',
    'true_angle',
    'success',
    'terminal_state',
    'topk_indices',
    'mean',
    'var',
    'prev_mean',
    'prev_var',
    'returned_pred',
    'returned_true',
    'returned_pos_l2',
    'returned_angle',
    'returned_success',
    'returned_terminal_state',
)

INVARIANT_KEYS = (
    'version',
    'generators',
    'scorers',
    'steps',
    'horizon',
    'goal_offset',
    'action_block',
)

FINITE_KEYS = (
    'initial_state',
    'goal_state',
    'candidates',
    'pred',
    'true',
    'terminal_state',
    'prev_mean',
    'prev_var',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument(
        '--generator-name',
        help=(
            'Normalize one-generator source labels before merging. Original '
            'labels remain recorded in merge_audit.'
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_savez(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if len(args.sources) < 2:
        raise ValueError('at least two source archives are required')
    if len(args.sources) != len(set(args.sources)):
        raise ValueError('duplicate source paths are not allowed')

    loaded: list[dict[str, np.ndarray]] = []
    source_audit = []
    for path in args.sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        missing = sorted(
            set((*STATE_KEYS, *INVARIANT_KEYS, *FINITE_KEYS))
            - set(arrays)
        )
        if missing:
            raise ValueError(f'{path} is missing fields {missing}')
        num_states = len(arrays['rows'])
        for key in STATE_KEYS:
            if arrays[key].shape[0] != num_states:
                raise ValueError(
                    f'{path}: {key} has state axis '
                    f'{arrays[key].shape[0]} != {num_states}'
                )
        nonfinite = [
            key for key in FINITE_KEYS if not np.isfinite(arrays[key]).all()
        ]
        if nonfinite:
            raise ValueError(f'{path}: non-finite fields {nonfinite}')
        rows = arrays['rows'].astype(np.int64)
        if len(rows) != len(np.unique(rows)):
            raise ValueError(f'{path}: duplicate dataset rows')
        loaded.append(arrays)
        source_audit.append(
            {
                'path': str(path.resolve()),
                'sha256': sha256(path),
                'num_states': num_states,
                'generators': arrays['generators'].astype(str).tolist(),
                'scorers': arrays['scorers'].astype(str).tolist(),
            }
        )

    reference = loaded[0]
    for source_i, arrays in enumerate(loaded[1:], start=1):
        for key in INVARIANT_KEYS:
            if key == 'generators' and args.generator_name:
                continue
            if not np.array_equal(arrays[key], reference[key]):
                raise ValueError(
                    f'source {source_i} differs for invariant {key}'
                )
    if args.generator_name:
        for source_i, arrays in enumerate(loaded):
            if len(arrays['generators']) != 1:
                raise ValueError(
                    f'source {source_i} has {len(arrays["generators"])} '
                    'generators; --generator-name requires exactly one'
                )

    all_rows = np.concatenate(
        [arrays['rows'].astype(np.int64) for arrays in loaded]
    )
    unique_rows, counts = np.unique(all_rows, return_counts=True)
    duplicated = unique_rows[counts > 1]
    if len(duplicated):
        raise ValueError(
            f'cross-source dataset-row overlap: {duplicated.tolist()}'
        )

    output = {
        key: np.concatenate([arrays[key] for arrays in loaded], axis=0)
        for key in STATE_KEYS
    }
    output.update(
        {key: np.asarray(reference[key]) for key in INVARIANT_KEYS}
    )
    if args.generator_name:
        output['generators'] = np.asarray([args.generator_name])
    output['max_roundtrip_error'] = np.asarray(
        max(
            float(arrays.get('max_roundtrip_error', np.asarray(0.0)))
            for arrays in loaded
        ),
        dtype=np.float64,
    )
    output['elapsed_seconds'] = np.asarray(
        sum(
            float(arrays.get('elapsed_seconds', np.asarray(0.0)))
            for arrays in loaded
        ),
        dtype=np.float64,
    )
    audit = {
        'version': 1,
        'sources': source_audit,
        'num_states': int(len(all_rows)),
        'unique_rows': int(len(unique_rows)),
        'max_roundtrip_error': float(output['max_roundtrip_error']),
        'horizon': int(np.asarray(reference['horizon']).item()),
        'goal_offset': int(np.asarray(reference['goal_offset']).item()),
        'generator_name': args.generator_name,
    }
    output['merge_audit'] = np.asarray(
        json.dumps(audit, sort_keys=True)
    )
    atomic_savez(args.output, output)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f'merged query bank -> {args.output}')


if __name__ == '__main__':
    main()
