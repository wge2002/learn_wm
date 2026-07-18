"""Tests for sparse held-out validation of CEM iterates."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMSolver, CrossValidatedCEMSolver


class TargetCostModel(torch.nn.Module):
    def __init__(self, target: float) -> None:
        super().__init__()
        self.register_buffer('target', torch.tensor(float(target)))

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ) -> torch.Tensor:
        return (action_candidates - self.target).pow(2).sum(dim=(-1, -2))


def make_solver(selection_space: str = 'means', refit_topk: int = 5):
    proposer = TargetCostModel(0.0)
    base = CEMSolver(
        model=proposer,
        n_steps=3,
        num_samples=20,
        batch_size=1,
        topk=5,
        seed=7,
    )
    verifier = TargetCostModel(0.75)
    solver = CrossValidatedCEMSolver(
        base,
        verifier,
        steps=[0, 2],
        selection_space=selection_space,
        refit_topk=refit_topk,
    )
    action_space = gym_spaces.Box(
        low=-1,
        high=1,
        shape=(2, 2),
        dtype=np.float32,
    )
    config = PlanConfig(horizon=3, receding_horizon=2)
    solver.configure(action_space=action_space, n_envs=2, config=config)
    return solver


def test_cross_validated_cem_selects_lowest_verifier_cost():
    solver = make_solver('means')
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    output = solver(info)
    audit = output['cross_validation']

    assert output['actions'].shape == (2, 3, 2)
    assert audit['verifier_costs'].shape == (2, 2)
    assert audit['steps'].tolist() == [[0, 2], [0, 2]]
    assert torch.equal(
        audit['selected_index'],
        audit['verifier_costs'].argmin(dim=1),
    )

    means = torch.cat(
        [
            torch.stack(
                [entry['mean'] for entry in trace],
                dim=1,
            )
            for trace in solver.recorder.history
        ]
    )
    batch = torch.arange(2)
    expected = means[batch, audit['selected_index']]
    torch.testing.assert_close(output['actions'], expected)


def test_cross_validated_cem_can_select_generator_elites():
    solver = make_solver('elites')
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    output = solver(info)
    audit = output['cross_validation']

    assert audit['verifier_costs'].shape == (2, 10)
    assert audit['steps'].tolist() == [
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 2, 2, 2, 2, 2],
    ]
    assert output['actions'].shape == (2, 3, 2)


def test_cross_validated_cem_can_select_sampled_populations():
    solver = make_solver('populations')
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    output = solver(info)
    audit = output['cross_validation']

    assert audit['verifier_costs'].shape == (2, 40)
    assert audit['steps'].tolist() == [
        [0] * 20 + [2] * 20,
        [0] * 20 + [2] * 20,
    ]
    assert output['actions'].shape == (2, 3, 2)


def test_cross_validated_cem_can_refit_verifier_elites():
    solver = make_solver('refit', refit_topk=5)
    info = {'pixels': torch.randn(2, 1, 3, 8, 8)}
    output = solver(info)
    audit = output['cross_validation']

    assert audit['verifier_costs'].shape == (2, 40)
    assert audit['selected_indices'].shape == (2, 5)
    candidates, _ = solver._archive_candidates()
    batch = torch.arange(2)[:, None].expand(2, 5)
    expected = candidates[batch, audit['selected_indices']].mean(dim=1)
    torch.testing.assert_close(output['actions'], expected)


def test_cross_validated_cem_validates_configuration():
    base = CEMSolver(
        model=TargetCostModel(0.0),
        n_steps=3,
        num_samples=10,
        topk=2,
    )
    verifier = TargetCostModel(1.0)
    with pytest.raises(ValueError, match='selection_space'):
        CrossValidatedCEMSolver(
            base,
            verifier,
            steps=[0],
            selection_space='population',  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match='outside'):
        CrossValidatedCEMSolver(base, verifier, steps=[3])
    with pytest.raises(ValueError, match='refit_topk'):
        CrossValidatedCEMSolver(
            base,
            verifier,
            steps=[0],
            selection_space='refit',
            refit_topk=0,
        )
