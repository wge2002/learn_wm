"""Tests for post-search multi-proposer CEM portfolios."""

import numpy as np
import pytest
import torch
from gymnasium import spaces as gym_spaces

from stable_worldmodel.policy import PlanConfig
from stable_worldmodel.solver import CEMPortfolioSolver, CEMSolver


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


def make_portfolio() -> CEMPortfolioSolver:
    models = [TargetCostModel(target) for target in (-0.5, 0.0, 0.5)]
    solvers = [
        CEMSolver(
            model=model,
            n_steps=3,
            num_samples=20,
            batch_size=1,
            topk=5,
            seed=7,
        )
        for model in models
    ]
    portfolio = CEMPortfolioSolver(
        solvers=solvers,
        models=models,
        names=['left', 'center', 'right'],
        steps=[0, 2],
    )
    portfolio.configure(
        action_space=gym_spaces.Box(
            low=-1,
            high=1,
            shape=(2, 2),
            dtype=np.float32,
        ),
        n_envs=2,
        config=PlanConfig(horizon=3, receding_horizon=2),
    )
    return portfolio


def test_cem_portfolio_selects_rank_consensus_candidate():
    portfolio = make_portfolio()
    output = portfolio({'pixels': torch.randn(2, 1, 3, 8, 8)})
    audit = output['portfolio']

    assert output['actions'].shape == (2, 3, 2)
    assert audit['scorer_costs'].shape == (2, 3, 6)
    assert audit['candidates'].shape == (2, 6, 3, 2)
    assert torch.equal(
        audit['selected_index'],
        audit['consensus'].argmin(dim=1),
    )
    batch = torch.arange(2)
    torch.testing.assert_close(
        output['actions'],
        audit['candidates'][batch, audit['selected_index']],
    )
    assert portfolio.selection_summary()['num_plans'] == 2


def test_cem_portfolio_validates_configuration():
    model = TargetCostModel(0.0)
    solver = CEMSolver(model=model, n_steps=3)
    with pytest.raises(ValueError, match='at least two'):
        CEMPortfolioSolver(
            solvers=[solver],
            models=[model],
            names=['only'],
            steps=[0],
        )
    with pytest.raises(ValueError, match='equal size'):
        CEMPortfolioSolver(
            solvers=[solver, solver],
            models=[model],
            names=['one', 'two'],
            steps=[0],
        )
