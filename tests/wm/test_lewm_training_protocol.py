from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip('stable_pretraining')

from scripts.train.lewm import (  # noqa: E402
    NonFiniteGradGuardCallback,
    export_initial_weights,
    file_sha256,
    matched_one_step_prediction,
    state_dict_sha256,
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

    callback = NonFiniteGradGuardCallback()
    trainer = SimpleNamespace(current_epoch=0, global_step=1)
    callback.on_before_optimizer_step(trainer, model, optimizer)
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
    callback = NonFiniteGradGuardCallback(policy='error')
    trainer = SimpleNamespace(current_epoch=2, global_step=17)

    with pytest.raises(FloatingPointError, match='formal paired runs'):
        callback.on_before_optimizer_step(trainer, model, optimizer)
    assert all(parameter.grad is None for parameter in model.parameters())


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
