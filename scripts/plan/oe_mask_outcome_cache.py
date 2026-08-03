"""Add a deployable candidate-query mask to an imagined-outcome cache.

The full K3 scorer determines the subset before the extra model is queried.
Masked caches let the fixed-trace operator probe measure how much of the
cross-model vector signal survives when K10 is evaluated on only 5--20% of
the population.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('outcome', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--queries', type=int, required=True)
    parser.add_argument(
        '--strategy',
        choices=('elite', 'rank_stratified'),
        default='elite',
    )
    return parser.parse_args()


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    with np.load(args.source, allow_pickle=False) as archive:
        rows = np.asarray(archive['rows'], dtype=np.int64)
        # Primary K3 generator and primary K3 scorer.
        cost = np.asarray(archive['pred'])[:, 0, :, 0].astype(
            np.float32
        )
    with np.load(args.outcome, allow_pickle=False) as archive:
        outcome_rows = np.asarray(archive['rows'], dtype=np.int64)
        prior_audit = (
            json.loads(str(np.asarray(archive['audit']).item()))
            if 'audit' in archive.files
            else {}
        )
    if not np.array_equal(rows, outcome_rows):
        raise ValueError('source/outcome row mismatch')
    if cost.ndim != 3:
        raise ValueError(f'expected (state,round,N) cost, got {cost.shape}')
    candidates = cost.shape[-1]
    if not 1 <= args.queries < candidates:
        raise ValueError(
            f'queries must be inside [1,{candidates - 1}]'
        )

    order = np.argsort(cost, axis=-1, kind='stable')
    if args.strategy == 'elite':
        selected = order[..., : args.queries]
    else:
        positions = np.linspace(
            0,
            candidates - 1,
            args.queries,
            dtype=np.int64,
        )
        selected = np.take(order, positions, axis=-1)
    mask = np.zeros(cost.shape, dtype=bool)
    state_index = np.arange(cost.shape[0])[:, None, None]
    round_index = np.arange(cost.shape[1])[None, :, None]
    mask[state_index, round_index, selected] = True

    audit = {
        'version': 1,
        'kind': 'masked_crossmodel_outcome',
        'source': str(args.source.resolve()),
        'outcome': str(args.outcome.resolve()),
        'queries': args.queries,
        'population': candidates,
        'query_fraction': args.queries / candidates,
        'strategy': args.strategy,
        'selection_uses': 'primary K3 scalar rank only',
        'parent_audit': prior_audit,
    }
    atomic_savez(
        args.out,
        rows=rows,
        parent_outcome=np.asarray(str(args.outcome.resolve())),
        feature_mask=mask,
        audit=np.asarray(json.dumps(audit, sort_keys=True)),
    )
    print(
        f'masked outcome cache ({args.strategy}, '
        f'{args.queries}/{candidates}) -> {args.out}'
    )


if __name__ == '__main__':
    main()
