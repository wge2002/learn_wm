from __future__ import annotations

import json
import sys
from copy import deepcopy

from scripts.plan import summarize_controlled_metric_pairs as summary
from scripts.plan import verify_controlled_metric_pairing as verifier


def trajectory(sample_index: int, *, k5: bool) -> dict[str, object]:
    return {
        'sample_index': sample_index,
        'pencil': {
            'logdet_I_plus_pencil': 2.0 + (0.4 if k5 else 0.0),
            'trace_pencil': 3.0 + (0.3 if k5 else 0.0),
        },
        'horizon': {
            'log_shear_rms': 1.0 - (0.2 if k5 else 0.0),
            'log_scale_mean': -0.5,
            'max_gain': 1.2,
        },
        'action_to_residual_trace_ratio': 1.0 + (0.2 if k5 else 0.0),
        'action_energy_trace': 2.0,
        'residual_energy_trace': 2.0,
    }


def sufficiency() -> dict[str, object]:
    return {
        probe: {'r2_uniform': 0.8}
        for probe in summary.SUFFICIENCY_PROBES
    } | {'block_motion_probe': {'roc_auc': 0.8}}


def g1() -> dict[str, object]:
    linear = {'r2_uniform': 0.95}
    return {
        'forward': {'linear': linear, 'orthogonal': {'r2': 0.60}},
        'reverse': {'linear': linear, 'orthogonal': {'r2': 0.60}},
        'cycle': {
            'source_to_target_to_source': {'r2_uniform': 0.95},
            'target_to_source_to_target': {'r2_uniform': 0.95},
        },
    }


def audit(seed: int, k1_checkpoint: str, k5_checkpoint: str) -> dict:
    common_model = {'sufficiency': sufficiency()}
    rows = list(range(4))
    return {
        'config': {
            'pair_id': f'seed_{seed:04d}',
            'training_seed': seed,
            'checkpoint_epoch': 30,
            'num_samples': 16,
            'jacobian_samples': 4,
            'history': 3,
            'horizon': 5,
            'frameskip': 5,
            'ridge_alpha': 1e-3,
            'block_motion_threshold': 1e-3,
            'seed': 20260810,
            'dataset': '/data/pusht.h5',
        },
        'reference': 'K1',
        'metadata': {
            'episodes': list(range(16)),
            'starts': list(range(16)),
            'train_indices': list(range(12)),
            'test_indices': list(range(12, 16)),
            'jacobian_indices': rows,
            'action_mean': [0.0, 0.0],
            'action_std': [1.0, 1.0],
        },
        'models': {
            'K1': {
                **common_model,
                'checkpoint': k1_checkpoint,
                'controlled_metric': {
                    'per_trajectory': [
                        trajectory(index, k5=False) for index in rows
                    ]
                },
            },
            'K5': {
                **common_model,
                'checkpoint': k5_checkpoint,
                'controlled_metric': {
                    'per_trajectory': [
                        trajectory(index, k5=True) for index in rows
                    ]
                },
            },
        },
        'g1_bidirectional_maps': {'K1_vs_K5': g1()},
    }


def test_multi_seed_summary_reaches_only_provisional_pass(
    tmp_path, monkeypatch
):
    inputs = []
    proof_pairs = []
    for seed in (7, 13, 42):
        k1 = str(tmp_path / f'k1_s{seed}.pt')
        k5 = str(tmp_path / f'k5_s{seed}.pt')
        path = tmp_path / f'audit_s{seed}.json'
        path.write_text(json.dumps(audit(seed, k1, k5)))
        inputs.append(path)
        proof_pairs.append(
            {
                'pair_id': f'seed_{seed:04d}',
                'training_seed': seed,
                'arms': {
                    'k1': {'checkpoint': k1},
                    'k5': {'checkpoint': k5},
                },
            }
        )

    proof = tmp_path / 'proof.json'
    proof.write_text(
        json.dumps({'status': 'PASS', 'epochs': 30, 'pairs': proof_pairs})
    )
    output_json = tmp_path / 'summary.json'
    output_md = tmp_path / 'summary.md'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'summarize_controlled_metric_pairs.py',
            '--input',
            *[str(path) for path in inputs],
            '--protocol-proof',
            str(proof),
            '--bootstraps',
            '1000',
            '--output-json',
            str(output_json),
            '--output-md',
            str(output_md),
        ],
    )

    summary.main()

    result = json.loads(output_json.read_text())
    assert result['protocol_proof_status'] == 'PASS'
    assert result['decision']['status'] == 'PROVISIONAL_PASS'
    assert all(
        gate['pass'] for gate in result['decision']['gates'].values()
    )
    assert 'not the intrinsic CP_H certificate' in output_md.read_text()


def test_pairing_verifier_accepts_only_the_declared_objective_delta(
    tmp_path, monkeypatch
):
    log_dir = tmp_path / 'logs'
    checkpoint_root = tmp_path / 'checkpoints'
    log_dir.mkdir()
    state_hash = 'a' * 64
    file_hash = 'b' * 64
    trace_hash = 'c' * 64
    log = '\n'.join(
        [
            '[protocol] global_seed=7 workers_seeded=true',
            (
                '[protocol] dataset_num_steps=8 dataset_samples=100 '
                'train_samples=90 val_samples=10 train_batches=9 '
                'split_sha256=split loader_state_sha256=loader'
            ),
            (
                '[protocol] loaded_initialization=/init.pt '
                f'state_sha256={state_hash} file_sha256={file_hash}'
            ),
            (
                '[pairing] epoch=0 batch=0 keys=action,state,proprio '
                f'sha256={trace_hash}'
            ),
        ]
    )
    common = {
        'seed': 7,
        'nonfinite_grad_policy': 'skip',
        'nonfinite_max_skip_frac': 0.0001,
        'nonfinite_max_total_skips': 3,
        'init_weights_path': '/init.pt',
        'output_model_name': None,
        'subdir': None,
        'trainer': {'max_epochs': 30, 'deterministic': True},
        'data': {'dataset': {'num_steps': 8}},
        'model': {'predictor': {'dropout': 0.0}},
        'wm': {'history_size': 3, 'num_preds': 1},
    }
    for arm in ('k1', 'k5'):
        name = f'cm_{arm}_s0007_unit'
        run_dir = checkpoint_root / name
        run_dir.mkdir(parents=True)
        (run_dir / 'weights_epoch_30.pt').write_bytes(b'checkpoint')
        config = deepcopy(common)
        config['output_model_name'] = name
        config['subdir'] = name
        config['wm']['unroll_tf' if arm == 'k1' else 'unroll'] = 5
        (run_dir / 'config.yaml').write_text(json.dumps(config))
        (log_dir / f'train_{name}.log').write_text(log)

    output = tmp_path / 'proof.json'
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'verify_controlled_metric_pairing.py',
            '--log-dir',
            str(log_dir),
            '--checkpoint-root',
            str(checkpoint_root),
            '--run-tag',
            'unit',
            '--seeds',
            '7',
            '--epochs',
            '30',
            '--output',
            str(output),
        ],
    )

    verifier.main()

    proof = json.loads(output.read_text())
    assert proof['status'] == 'PASS'
    assert proof['pairs'][0]['initialization_state_sha256'] == state_hash
