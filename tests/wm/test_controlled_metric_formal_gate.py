"""Static guards on the v3 formal paired protocol's preconditions.

Two things must hold before the six formal trainings may start, and neither is
checkable from a training log after the fact:

1. The encoder-FP32 precision island is a *common numerical protocol*, so it
   must be enabled identically in both arms. Enabling it in one arm only would
   silently add a numerics difference to the K1-TF/K5 intervention.
2. Training must be gated on the two-seed stability *evidence file*, not on a
   boolean an operator can export. The old ``LEWM_FIRST_INF_ROOTCAUSE_RESOLVED``
   bypass must stay gone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / 'scripts/train/config'
K1_CONFIG = CONFIG_DIR / 'lewm_paired_k1.yaml'
K5_CONFIG = CONFIG_DIR / 'lewm_paired_k5.yaml'
LAUNCHER = REPO / 'scripts/plan/run_controlled_metric_paired.sh'
DLC_WRAPPER = REPO / 'scripts/plan/run_controlled_metric_paired_dlc.sh'
LAST_FAILURE_STEP = 137496
REMOVED_BYPASS = 'LEWM_FIRST_INF_ROOTCAUSE_RESOLVED'


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def load(path: Path) -> dict:
    return yaml.safe_load(read(path))


@pytest.mark.parametrize('script', [LAUNCHER, DLC_WRAPPER])
def test_launchers_are_valid_bash(script: Path):
    subprocess.run(['bash', '-n', str(script)], check=True)


def test_both_formal_arms_enable_the_encoder_fp32_island():
    # Explicit in both files, not inherited, so a reader of either config sees
    # the numerical protocol without resolving the defaults chain.
    assert 'encoder_fp32: true' in read(K1_CONFIG)
    assert 'encoder_fp32: true' in read(K5_CONFIG)
    assert load(K1_CONFIG)['encoder_fp32'] is True
    assert load(K5_CONFIG)['encoder_fp32'] is True


def test_the_island_is_symmetric_and_labelled_as_shared_protocol():
    k1 = read(K1_CONFIG)
    k5 = read(K5_CONFIG)
    for text in (k1, k5):
        assert 'NOT the K1-TF/K5 intervention' in text
    # The only intended difference between the arms is the unroll setting.
    assert load(K1_CONFIG)['wm'] == {'unroll_tf': 5}
    assert load(K5_CONFIG)['wm'] == {'unroll': 5}
    # Everything else in the shared numerical protocol must match exactly.
    for key in (
        'encoder_fp32',
        'nonfinite_grad_policy',
        'nonfinite_max_skip_frac',
        'nonfinite_max_total_skips',
        'pairing_trace_batches',
        'model',
        'data',
        'trainer',
    ):
        assert load(K1_CONFIG)[key] == load(K5_CONFIG)[key], key


def test_the_boolean_bypass_is_gone_everywhere():
    for script in (LAUNCHER, DLC_WRAPPER):
        assert REMOVED_BYPASS not in read(script)


def test_training_requires_the_gate_file_before_any_init_or_train_work():
    script = read(LAUNCHER)
    gate_call = script.index('if has_phase train; then\n  require_stability_gate')
    # The gate is checked before the init phase runs, not merely before train.
    assert gate_call < script.index('if has_phase init; then')
    assert gate_call < script.index('if has_phase train; then\n  for seed in')
    assert 'LEWM_STABILITY_GATE is not set' in script
    assert 'no such gate file' in script


def test_training_pins_the_complete_formal_wave_and_clean_commit():
    script = read(LAUNCHER)
    assert 'NGPU=${NGPU:-6}' in script
    assert 'PHASES=init,train NGPU=6' in script
    for variable, expected in (
        ('EPOCHS', '30'),
        ('SEEDS', '7 13 42'),
        ('NGPU', '6'),
    ):
        assert f'if [ "${variable}" != "{expected}" ]; then' in script
        assert f'formal training blocked: {variable}=' in script
    assert 'diff --quiet' in script
    assert 'diff --cached --quiet' in script
    assert 'tracked repository changes detected' in script


def test_gate_validates_every_required_field_and_the_commit():
    script = read(LAUNCHER)
    for key, value in (
        ('result', 'PASS'),
        ('gate', 'encoder_fp32_two_seed_stability'),
        ('seeds', '13 42'),
        ('encoder_fp32', 'true'),
        ('max_epochs', '30'),
        ('nonfinite_grad_policy', 'error'),
    ):
        assert f'gate_expect {key} ' in script, key
        assert value in script, value
    # Horizon strictly past the later historical first-Inf step, and the gate
    # must have been produced by the commit that is about to train.
    assert f'-le {LAST_FAILURE_STEP} ]' in script
    assert 'rev-parse HEAD' in script
    assert 'validated commit' in script
    # Exactly-one-line matching, so a duplicated or absent key cannot pass.
    assert "expected exactly one '$key=' line" in script


def test_audit_and_summarize_phases_do_not_need_the_gate():
    script = read(LAUNCHER)
    # The requirement is guarded by the train phase alone.
    assert script.count('require_stability_gate') == 2
    assert 'if has_phase train; then\n  require_stability_gate\nfi' in script
    for phase in ('audit', 'summarize'):
        block = script.index(f'if has_phase {phase}; then')
        assert 'require_stability_gate' not in script[block:]


def test_dlc_wrapper_requires_and_forwards_the_gate_path():
    script = read(DLC_WRAPPER)
    assert 'LEWM_STABILITY_GATE is not set' in script
    assert 'export LEWM_STABILITY_GATE' in script
    # Still exactly six GPUs for the six independent single-card trainings.
    assert 'test "$NGPU" -eq 6' in script
    assert 'test "$GPU_IDS" = 0,1,2,3,4,5' in script
