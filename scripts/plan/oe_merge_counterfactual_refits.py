"""Merge state-sharded counterfactual-refit archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


STATIC_FIELDS = {
    'version',
    'generators',
    'scorers',
    'steps',
    'global_selectors',
    'component_selectors',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('sources', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
    loaded = []
    source_audits = []
    for path in args.sources:
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            loaded.append(
                {key: np.asarray(archive[key]) for key in archive.files}
            )
        source_audits.append(
            {
                'path': str(path.resolve()),
                'sha256': sha256(path),
                'num_states': int(len(loaded[-1]['rows'])),
                'audit': json.loads(str(loaded[-1]['audit'])),
            }
        )
    reference_fields = set(loaded[0])
    for source in loaded[1:]:
        if set(source) != reference_fields:
            raise ValueError('refit shard fields differ')
    for field in STATIC_FIELDS:
        reference = loaded[0][field]
        for source in loaded[1:]:
            if not np.array_equal(reference, source[field]):
                raise ValueError(f'static field differs: {field}')

    order = np.argsort(np.concatenate([item['rows'] for item in loaded]))
    merged = {}
    for field in reference_fields - {'audit'}:
        if field in STATIC_FIELDS:
            merged[field] = loaded[0][field]
        else:
            concatenated = np.concatenate(
                [item[field] for item in loaded],
                axis=0,
            )
            merged[field] = concatenated[order]
    rows = merged['rows']
    if len(np.unique(rows)) != len(rows):
        raise ValueError('merged rows are not unique')
    audit = {
        'version': 1,
        'num_states': int(len(rows)),
        'unique_rows': int(len(np.unique(rows))),
        'sources': source_audits,
    }
    merged['audit'] = np.asarray(json.dumps(audit, sort_keys=True))
    atomic_savez(args.output, **merged)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print(f'merged counterfactual refits -> {args.output}')


if __name__ == '__main__':
    main()
