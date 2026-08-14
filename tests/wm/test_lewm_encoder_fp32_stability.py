"""Static guards on the encoder-FP32 two-seed stability validation.

These are deliberately fast and dependency-free: they read the launcher and the
config rather than training anything. The mistake they exist to prevent is the
one already made once -- running the validation on a 13-epoch horizon. The
cosine schedule is epoch-based (``total_steps = max_epochs * steps_per_epoch``
in ``scripts/train/lewm.py``), so a different ``max_epochs`` changes the
learning rate at every step and validates a recipe nobody ran.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

REPO = Path(__file__).resolve().parents[2]
LAUNCHER = REPO / 'scripts/plan/run_lewm_nonfinite_rootcause_dlc.sh'
CONFIG_DIR = REPO / 'scripts/train/config'
STABILITY_CONFIG = CONFIG_DIR / 'lewm_encoder_fp32_stability.yaml'
REPRO_CONFIG = CONFIG_DIR / 'lewm_nonfinite_v2_k1_repro.yaml'

# The two historical first-Inf global steps. The validation is meaningless
# unless it runs past the later one.
HISTORICAL_FAILURE_STEPS = (115683, 137496)
LAST_FAILURE_STEP = max(HISTORICAL_FAILURE_STEPS)


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def load(path: Path) -> dict:
    return yaml.safe_load(read(path))


def test_launcher_is_valid_bash():
    subprocess.run(['bash', '-n', str(LAUNCHER)], check=True)


def test_stability_config_pins_the_thirty_epoch_cosine_horizon():
    config = load(STABILITY_CONFIG)
    # The regression this whole file exists for.
    assert config['trainer']['max_epochs'] == 30


def test_stability_config_inherits_the_historical_v2_k1_recipe():
    config = load(STABILITY_CONFIG)
    # Inheritance, not restatement: the recipe, init handling and data must be
    # the reproduction's, so the only difference is encoder_fp32.
    assert 'lewm_nonfinite_v2_k1_repro' in config['defaults']
    # And the parent must not itself have drifted off the paired K1 recipe.
    assert 'lewm_paired_k1' in load(REPRO_CONFIG)['defaults']


def test_stability_config_enables_encoder_fp32_with_zero_tolerance():
    config = load(STABILITY_CONFIG)
    assert config['encoder_fp32'] is True
    # Zero tolerated raw-gradient events: policy=error, no skip budget at all.
    assert config['nonfinite_grad_policy'] == 'error'
    assert config['nonfinite_max_skip_frac'] == 0.0
    assert config['nonfinite_max_total_skips'] == 0


def test_encoder_fp32_defaults_off_so_no_existing_recipe_changes():
    assert load(CONFIG_DIR / 'lewm.yaml')['encoder_fp32'] is False
    # Explicitly false in the reproduction arm, not merely absent: its parent
    # lewm_paired_k1 now sets encoder_fp32: true as the formal arms' common
    # numerical protocol, and Hydra propagates that through the defaults list.
    # Absence would therefore resolve to true and the stability config would no
    # longer be a single intervention over the reproduction.
    assert load(REPRO_CONFIG)['encoder_fp32'] is False


def test_launcher_horizon_floor_clears_both_historical_failures():
    script = read(LAUNCHER)
    # Default horizon, and the assertion that guards a caller-supplied one.
    assert f'STABILITY_STOP_AFTER_STEP:-138{0:03d}' in script
    assert f'test "$STABILITY_STOP_AFTER_STEP" -gt {LAST_FAILURE_STEP}' in script
    assert f'-le {LAST_FAILURE_STEP} ]' in script
    assert 'test "$EPOCHS" -eq 30' in script
    for step in HISTORICAL_FAILURE_STEPS:
        assert str(step) in script


def test_launcher_requests_exactly_two_gpus_one_seed_each():
    script = read(LAUNCHER)
    # Two distinct GPU indices, rejected otherwise, and one single-device
    # training process per seed. Never a whole 8-GPU node.
    assert 'ROOTCAUSE_GPU_IDS:-"0 1"' in script
    assert 'must contain two distinct GPU indices' in script
    assert 'SPECS="13:$GPU_SEED13 42:$GPU_SEED42"' in script
    assert 'trainer.devices=1' in script
    assert 'test "${#names[@]}" -eq 2' in script


def test_stability_mode_gates_on_evidence_not_just_exit_code():
    script = read(LAUNCHER)
    # rc=0 alone is not evidence of work (cf. the silent-success DLC job), so
    # every one of these conditions must appear.
    assert 'expected clean exit' in script
    assert 'non-finite bundle(s)' in script
    assert 'recorded an offending gradient' in script
    assert 'never printed the stability-stop marker' in script
    assert 'PASS_${name}.txt' in script


def test_aggregate_gate_requires_both_seeds_to_pass():
    script = read(LAUNCHER)
    gate_index = script.index('gate="$OUT/STABILITY_GATE_PASS.txt"')
    # Every per-seed PASS file is asserted to exist, and the run count checked,
    # before the aggregate gate file is opened for writing.
    assert script.index('test -f "$OUT/PASS_${name}.txt"') < gate_index
    assert script.index('test "${#names[@]}" -eq 2') < gate_index
    assert script.index('test "$failed" -eq 0') < gate_index


def test_stability_mode_does_not_auto_launch_or_weaken_formal_pairing():
    script = read(LAUNCHER)
    # The note must describe what is actually true: this gate is the evidence
    # the formal launcher consumes, it just does not launch anything itself.
    assert 'does not auto-launch the formal wave' in script
    assert 'LEWM_STABILITY_GATE' in script
    assert 'does not authorize a formal run' not in script
    # No formal config is referenced, and nothing submits to DLC from here.
    for forbidden in ('lewm_paired_k1', 'lewm_paired_k5', 'dlc-run', 'dlc submit'):
        assert forbidden not in script


def test_repro_mode_still_expects_failure_after_the_refactor():
    script = read(LAUNCHER)
    # The two modes must keep opposite success criteria on the same recipe.
    assert 'completed without reproducing first Inf' in script
    assert 'CONFIG_NAME=lewm_nonfinite_v2_k1_repro' in script
    assert 'CONFIG_NAME=lewm_encoder_fp32_stability' in script
    assert 'MODE=${MODE:-repro}' in script
