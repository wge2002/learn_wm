from __future__ import annotations

import math
import re
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('stable_pretraining')

import inspect  # noqa: E402

import stable_pretraining as spt  # noqa: E402

from scripts.train.lewm import (  # noqa: E402
    DivergenceTraceCallback,
    NonFiniteGradGuardCallback,
    RawGradientModule,
    capture_nonfinite_replay_state,
    export_initial_weights,
    file_sha256,
    matched_one_step_prediction,
    state_dict_sha256,
)
from stable_worldmodel.wm.lewm import LeWM  # noqa: E402


class FakeViT(torch.nn.Module):
    """Minimal stand-in for ``vit_hf``: records the dtype it computed in."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)
        self.hidden_dtype = None

    def forward(self, pixels, interpolate_pos_encoding=False):
        hidden = self.proj(pixels).unsqueeze(1)
        self.hidden_dtype = hidden.dtype
        return SimpleNamespace(last_hidden_state=hidden)


def build_lewm(encoder_fp32: bool) -> LeWM:
    return LeWM(
        encoder=FakeViT(),
        predictor=torch.nn.Identity(),
        action_encoder=torch.nn.Identity(),
        projector=torch.nn.Linear(4, 4),
        encoder_fp32=encoder_fp32,
    )


class FakePredictor:
    def __init__(self):
        self.shapes = None

    def predict(self, embeddings, actions):
        self.shapes = (tuple(embeddings.shape), tuple(actions.shape))
        return embeddings + 10.0


def test_matched_one_step_uses_only_common_clip_prefix():
    model = FakePredictor()
    embeddings = torch.arange(2 * 8 * 4).reshape(2, 8, 4).float()
    actions = torch.zeros(2, 8, 6)

    prediction, target = matched_one_step_prediction(
        model, embeddings, actions, history_size=3
    )

    assert model.shapes == ((2, 3, 4), (2, 3, 6))
    assert torch.equal(prediction, embeddings[:, :3] + 10.0)
    assert torch.equal(target, embeddings[:, 1:4])


def test_encoder_fp32_island_is_opt_in_and_leaves_projector_in_bf16():
    pixels = torch.randn(2, 3, 4)

    model = build_lewm(encoder_fp32=False)
    assert model.encoder_fp32 is False
    with torch.autocast('cpu', dtype=torch.bfloat16):
        default = model.encode({'pixels': pixels})
    # Default path: the encoder runs in the ambient autocast dtype. This is the
    # historical numerics that produced the two non-finite backwards.
    assert model.encoder.hidden_dtype is torch.bfloat16
    assert default['emb'].dtype is torch.bfloat16

    island = build_lewm(encoder_fp32=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        opted_in = island.encode({'pixels': pixels})
    # Opted in: the encoder is an FP32 island, while the projector -- which the
    # replay shows to be well conditioned -- still runs under autocast.
    assert island.encoder.hidden_dtype is torch.float32
    assert opted_in['emb'].dtype is torch.bfloat16


def test_upstream_calls_after_manual_backward_before_clipping():
    # The guard's correctness rests on this upstream ordering. If a future
    # stable_pretraining release moves the hook after clip_gradients, the guard
    # silently degrades to the post-clip reading it was written to replace.
    source = inspect.getsource(spt.Module.training_step)
    backward = source.index('self.manual_backward(')
    hook = source.index('self.after_manual_backward()')
    clip = source.index('self.clip_gradients(')
    assert backward < hook < clip


def test_raw_gradient_module_dispatches_guard_before_clipping():
    calls = []

    class Recorder:
        def on_raw_gradients(self, trainer, pl_module, optimizer):
            calls.append(('raw', optimizer))

    optimizer = torch.optim.AdamW(torch.nn.Linear(2, 1).parameters())
    recorder = Recorder()
    module = SimpleNamespace(
        trainer=SimpleNamespace(callbacks=[recorder]),
        optimizers=lambda: optimizer,
    )

    RawGradientModule.after_manual_backward(module)

    assert calls == [('raw', optimizer)]
    # The guard must not ALSO override the post-clip Lightning hook, or one bad
    # step would be reported twice with contradictory numbers. Check the class
    # dict, not hasattr: the Callback base defines an inert default.
    for guard in (NonFiniteGradGuardCallback, DivergenceTraceCallback):
        assert 'on_before_optimizer_step' not in vars(guard)
        assert 'on_raw_gradients' in vars(guard)


def test_guard_sees_unclipped_gradients_in_the_real_step_order(capsys):
    # Reproduce training_step's sequence around a single inf. The v2 evidence
    # reported finite_max_abs=0 for every tensor because clipping had already
    # multiplied the healthy gradients by 1/inf == 0. Running in the correct
    # window must instead report the true magnitudes.
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 3.0)
    model.weight.grad.view(-1)[0] = float('inf')

    callback = NonFiniteGradGuardCallback(policy='skip')
    trainer = SimpleNamespace(current_epoch=1, global_step=7)
    callback.on_raw_gradients(trainer, model, optimizer)

    printed = capsys.readouterr().out
    assert 'finite_max_abs=3' in printed
    assert 'raw_grad_norm=inf' in printed
    # 15 surviving weight elements + 4 bias elements, all 3.0, so the healthy
    # remainder is sqrt(19 * 9) == sqrt(171) ~= 13.077. Masking whole tensors
    # instead of elements drops the entire weight matrix and prints sqrt(36)
    # == 6, the bias norm alone -- a plausible-looking number that answers a
    # different question. Parse the printed value rather than substring-match
    # it, so 13.0 could never satisfy a `'13' in printed` check.
    finite_printed = re.search(r'finite_grad_norm=(\S+)', printed)
    assert finite_printed is not None, printed
    assert float(finite_printed.group(1)) == pytest.approx(
        math.sqrt(171.0), rel=1e-6
    )
    # And the containment still holds: nothing reaches the optimizer.
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonfinite_guard_is_an_exact_adamw_parameter_skip():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.2)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    parameters_before = [
        parameter.detach().clone() for parameter in model.parameters()
    ]
    state_before = {
        parameter: {
            key: (
                value.detach().clone()
                if torch.is_tensor(value)
                else deepcopy(value)
            )
            for key, value in optimizer.state[parameter].items()
        }
        for parameter in model.parameters()
    }
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.ones_like(parameter)
        if index == 0:
            parameter.grad.view(-1)[0] = float('inf')

    callback = NonFiniteGradGuardCallback(policy='skip')
    trainer = SimpleNamespace(current_epoch=0, global_step=1)
    callback.on_raw_gradients(trainer, model, optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())

    optimizer.step()
    for parameter, expected in zip(
        model.parameters(), parameters_before, strict=True
    ):
        assert torch.equal(parameter, expected)
        for key, expected_state in state_before[parameter].items():
            actual_state = optimizer.state[parameter][key]
            if torch.is_tensor(expected_state):
                assert torch.equal(actual_state, expected_state)
            else:
                assert actual_state == expected_state


def test_formal_nonfinite_policy_fails_the_run():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('inf'))
    callback = NonFiniteGradGuardCallback()
    trainer = SimpleNamespace(current_epoch=2, global_step=17)

    with pytest.raises(FloatingPointError, match='strict runs'):
        callback.on_raw_gradients(trainer, model, optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_nonfinite_skip_budget_aborts_after_preregistered_limit():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    callback = NonFiniteGradGuardCallback(
        max_total_skips=1, policy='skip'
    )
    trainer = SimpleNamespace(current_epoch=2, global_step=17)

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('inf'))
    callback.on_raw_gradients(trainer, model, optimizer)

    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('inf'))
    trainer.global_step += 1
    with pytest.raises(RuntimeError, match='preregistered limit'):
        callback.on_raw_gradients(trainer, model, optimizer)


def test_rootcause_evidence_keeps_pre_forward_rng_and_bn_buffers(
    tmp_path, monkeypatch
):
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
    optimizer = torch.optim.AdamW(model.parameters())
    callback = NonFiniteGradGuardCallback()
    trainer = SimpleNamespace(current_epoch=3, global_step=23)
    monkeypatch.setenv('SWM_CAPTURE_NONFINITE_REPLAY', '1')
    monkeypatch.setenv('SWM_NONFINITE_EVIDENCE_DIR', str(tmp_path))

    capture_nonfinite_replay_state(model)
    expected_running_mean = model[0].running_mean.detach().clone()
    model[0].running_mean.add_(4.0)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('inf'))

    with pytest.raises(FloatingPointError):
        callback.on_raw_gradients(trainer, model, optimizer)

    bundle = torch.load(
        tmp_path / 'nonfinite_e3_s23.pt',
        map_location='cpu',
        weights_only=False,
    )
    assert set(bundle['pre_forward_rng']) == {'cpu', 'cuda'}
    assert torch.equal(
        bundle['pre_forward_buffers']['0.running_mean'],
        expected_running_mean,
    )


def test_diagnostic_stop_preserves_scheduler_horizon(monkeypatch):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    callback = NonFiniteGradGuardCallback()
    trainer = SimpleNamespace(current_epoch=12, global_step=138000)
    monkeypatch.setenv('SWM_DIAGNOSTIC_STOP_AFTER_STEP', '138000')
    callback.on_train_start(trainer, model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    with pytest.raises(RuntimeError, match='historical failure window'):
        callback.on_raw_gradients(trainer, model, optimizer)


def test_stability_stop_exits_cleanly_and_leaves_max_epochs_alone(monkeypatch):
    # The validation counterpart of the diagnostic stop: crossing the horizon
    # with finite gradients is a PASS, so it must request should_stop rather
    # than raise, and must not shorten the epoch-based cosine schedule.
    monkeypatch.setenv('SWM_STABILITY_STOP_AFTER_STEP', '138000')
    callback = NonFiniteGradGuardCallback(policy='error')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = SimpleNamespace(
        current_epoch=29, global_step=138000, should_stop=False, max_epochs=30
    )
    callback.on_train_start(trainer, model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    callback.on_raw_gradients(trainer, model, optimizer)

    assert trainer.should_stop is True
    assert trainer.max_epochs == 30
    assert callback.skipped == 0
    # Gradients survive: this is a clean stop, not a containment event.
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_stability_stop_does_not_fire_before_the_horizon(monkeypatch):
    monkeypatch.setenv('SWM_STABILITY_STOP_AFTER_STEP', '138000')
    callback = NonFiniteGradGuardCallback(policy='error')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = SimpleNamespace(
        current_epoch=29, global_step=137496, should_stop=False
    )
    callback.on_train_start(trainer, model)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    callback.on_raw_gradients(trainer, model, optimizer)

    # Exactly at the later historical failure step, so the run must continue.
    assert trainer.should_stop is False


def test_stability_stop_still_fails_on_a_nonfinite_gradient(monkeypatch):
    monkeypatch.setenv('SWM_STABILITY_STOP_AFTER_STEP', '138000')
    callback = NonFiniteGradGuardCallback(policy='error')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    trainer = SimpleNamespace(
        current_epoch=5, global_step=60000, should_stop=False
    )
    callback.on_train_start(trainer, model)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float('inf'))

    # Zero tolerated events: the stop mechanism must not soften the guard.
    with pytest.raises(FloatingPointError):
        callback.on_raw_gradients(trainer, model, optimizer)
    assert trainer.should_stop is False


def test_stop_mechanisms_are_mutually_exclusive(monkeypatch):
    # They demand opposite exit codes at the same horizon, so allowing both
    # would make the run's success criterion ambiguous.
    monkeypatch.setenv('SWM_STABILITY_STOP_AFTER_STEP', '138000')
    monkeypatch.setenv('SWM_DIAGNOSTIC_STOP_AFTER_STEP', '138000')
    callback = NonFiniteGradGuardCallback(policy='error')
    with pytest.raises(ValueError, match='mutually exclusive'):
        callback.on_train_start(SimpleNamespace(), torch.nn.Linear(2, 1))


def test_stability_stop_refuses_a_skip_policy(monkeypatch):
    monkeypatch.setenv('SWM_STABILITY_STOP_AFTER_STEP', '138000')
    callback = NonFiniteGradGuardCallback(policy='skip')
    with pytest.raises(ValueError, match='policy=error'):
        callback.on_train_start(SimpleNamespace(), torch.nn.Linear(2, 1))


def test_initialization_export_is_immutable_and_bitwise_checked(tmp_path):
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 3)
    path = tmp_path / 'initialization.pt'

    digest, reused = export_initial_weights(model, path)
    assert not reused
    assert digest == state_dict_sha256(model.state_dict())
    assert len(file_sha256(path)) == 64

    clone = torch.nn.Linear(4, 3)
    clone.load_state_dict(model.state_dict())
    reused_digest, reused = export_initial_weights(clone, path)
    assert reused
    assert reused_digest == digest

    with torch.no_grad():
        clone.weight[0, 0].add_(1.0)
    with pytest.raises(FileExistsError, match='does not match'):
        export_initial_weights(clone, path)


def test_state_dict_hash_supports_scalar_buffers():
    state = {
        'weight': torch.arange(4, dtype=torch.float32),
        'num_batches_tracked': torch.tensor(0, dtype=torch.int64),
    }

    assert len(state_dict_sha256(state)) == 64
