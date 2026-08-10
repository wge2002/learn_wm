#!/usr/bin/env python3
"""Verify that controlled-metric K1/K5 runs differ only in their objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--run-tag', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def run_name(arm: str, seed: int, run_tag: str) -> str:
    return f'cm_{arm}_s{seed:04d}_{run_tag}'


def load_log(path: Path) -> dict[str, object]:
    text = path.read_text()
    seeds = re.findall(r'\[protocol\] global_seed=(\d+)', text)
    initializations = re.findall(
        r'\[protocol\] loaded_initialization=.*?'
        r'state_sha256=([0-9a-f]{64}) file_sha256=([0-9a-f]{64})',
        text,
    )
    dataset_lines = re.findall(
        r'^\[protocol\] dataset_num_steps=.*$', text, flags=re.MULTILINE
    )
    if len(dataset_lines) != 1:
        raise ValueError(
            f'{path}: expected one dataset protocol line, got '
            f'{len(dataset_lines)}'
        )
    dataset = dict(re.findall(r'(\w+)=([^\s]+)', dataset_lines[0]))
    traces = [
        {
            'batch': int(batch),
            'keys': keys,
            'sha256': digest,
        }
        for batch, keys, digest in re.findall(
            r'^\[pairing\] epoch=0 batch=(\d+) '
            r'keys=([^\s]+) sha256=([0-9a-f]{64})$',
            text,
            flags=re.MULTILINE,
        )
    ]
    if len(seeds) != 1 or len(initializations) != 1:
        raise ValueError(
            f'{path}: expected one global seed and one initialization hash'
        )
    if not traces:
        raise ValueError(f'{path}: no pairing batch fingerprints found')
    if '[grad-guard] skipped non-finite gradient' in text:
        raise ValueError(f'{path}: contains a skipped non-finite update')
    return {
        'global_seed': int(seeds[0]),
        'initialization_state_sha256': initializations[0][0],
        'initialization_file_sha256': initializations[0][1],
        'dataset': dataset,
        'batch_traces': traces,
        'log_sha256': hashlib.sha256(text.encode()).hexdigest(),
    }


def load_config(path: Path) -> dict[str, object]:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f'{path}: config is not a mapping')
    return config


def common_config(config: dict[str, object]) -> dict[str, object]:
    """Remove naming and the one preregistered objective intervention."""

    result = copy.deepcopy(config)
    for key in (
        'output_model_name',
        'subdir',
        'init_weights_path',
        'export_init_weights_path',
        'init_only',
    ):
        result.pop(key, None)
    wm = result.get('wm')
    if isinstance(wm, dict):
        wm.pop('matched_one_step', None)
        wm.pop('unroll', None)
    return result


def config_sha256(config: dict[str, object]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(',', ':')
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('--seeds must be unique')

    pairs = []
    for seed in args.seeds:
        arms = {}
        configs = {}
        for arm in ('k1', 'k5'):
            name = run_name(arm, seed, args.run_tag)
            log_path = args.log_dir / f'train_{name}.log'
            run_dir = args.checkpoint_root / name
            config_path = run_dir / 'config.yaml'
            checkpoint = run_dir / f'weights_epoch_{args.epochs}.pt'
            for required in (log_path, config_path, checkpoint):
                if not required.is_file():
                    raise FileNotFoundError(required)
            arms[arm] = {
                **load_log(log_path),
                'name': name,
                'log': str(log_path.resolve()),
                'config': str(config_path.resolve()),
                'checkpoint': str(checkpoint.resolve()),
            }
            configs[arm] = load_config(config_path)

        errors = []
        if (
            arms['k1']['global_seed'] != seed
            or arms['k5']['global_seed'] != seed
        ):
            errors.append(
                'global seed does not match the declared training seed'
            )
        for key in (
            'initialization_state_sha256',
            'initialization_file_sha256',
            'dataset',
            'batch_traces',
        ):
            if arms['k1'][key] != arms['k5'][key]:
                errors.append(f'K1/K5 mismatch in {key}')
        if int(arms['k1']['dataset'].get('dataset_num_steps', -1)) != 8:
            errors.append('paired dataset_num_steps is not 8')

        common_k1 = common_config(configs['k1'])
        common_k5 = common_config(configs['k5'])
        if common_k1 != common_k5:
            errors.append(
                'resolved configs differ outside the objective intervention'
            )
        if configs['k1'].get('wm', {}).get('matched_one_step') is not True:
            errors.append('K1 does not enable matched_one_step')
        if configs['k5'].get('wm', {}).get('unroll') != 5:
            errors.append('K5 does not use open-loop unroll=5')
        for arm in ('k1', 'k5'):
            config = configs[arm]
            if config.get('seed') != seed:
                errors.append(f'{arm} config seed mismatch')
            if config.get('nonfinite_grad_policy') != 'error':
                errors.append(f'{arm} is not using strict non-finite handling')
            if config.get('trainer', {}).get('max_epochs') != args.epochs:
                errors.append(f'{arm} max_epochs mismatch')
            forbidden_trainer_keys = {
                'fast_dev_run',
                'limit_train_batches',
                'max_steps',
                'overfit_batches',
            }
            present = forbidden_trainer_keys.intersection(
                config.get('trainer', {})
            )
            if present:
                errors.append(
                    f'{arm} contains formal-run trainer limits: '
                    f'{sorted(present)}'
                )
            predictor = config.get('model', {}).get('predictor', {})
            if predictor.get('dropout') != 0.0:
                errors.append(f'{arm} predictor dropout is not disabled')

        pair = {
            'pair_id': f'seed_{seed:04d}',
            'training_seed': seed,
            'status': 'PASS' if not errors else 'FAIL',
            'errors': errors,
            'common_config_sha256': config_sha256(common_k1),
            'initialization_state_sha256': arms['k1'][
                'initialization_state_sha256'
            ],
            'initialization_file_sha256': arms['k1'][
                'initialization_file_sha256'
            ],
            'dataset_protocol': arms['k1']['dataset'],
            'batch_traces': arms['k1']['batch_traces'],
            'arms': arms,
        }
        pairs.append(pair)

    output = {
        'status': (
            'PASS'
            if all(pair['status'] == 'PASS' for pair in pairs)
            else 'FAIL'
        ),
        'run_tag': args.run_tag,
        'epochs': args.epochs,
        'pairs': pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(f'[pairing-proof] {output["status"]}: wrote {args.output}')
    if output['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
