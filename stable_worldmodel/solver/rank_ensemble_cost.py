"""Scale-invariant rank aggregation for world-model planning costs."""

from __future__ import annotations

import torch
from torch import nn


class RankEnsembleCost(nn.Module):
    """Expose several world models as one rank-consensus cost model.

    Latent goal distances from independently trained world models are not
    calibrated in scale. Fractional ranks preserve each model's candidate
    ordering, after which their mean can drive an ordinary optimizer on a
    shared population.
    """

    def __init__(self, models: list[nn.Module], names: list[str]) -> None:
        super().__init__()
        if len(models) < 2:
            raise ValueError('rank ensemble needs at least two models')
        if len(models) != len(names):
            raise ValueError('models and names must have equal size')
        if len(set(names)) != len(names):
            raise ValueError('rank ensemble names must be unique')
        self.models = nn.ModuleList(models)
        self.names = tuple(names)

    @staticmethod
    def fractional_ranks(values: torch.Tensor) -> torch.Tensor:
        """Return stable ranks in ``[0, 1]`` along the candidate axis."""
        n_candidates = values.shape[-1]
        if n_candidates < 1:
            raise ValueError('candidate axis must be non-empty')
        if n_candidates == 1:
            return torch.zeros_like(values)
        order = torch.argsort(values, dim=-1, stable=True)
        ranks = torch.argsort(order, dim=-1, stable=True)
        return ranks.to(values.dtype) / (n_candidates - 1)

    def get_cost(
        self,
        info_dict: dict,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        costs = torch.stack(
            [model.get_cost(info_dict, actions) for model in self.models],
            dim=1,
        )
        expected = (len(actions), len(self.models), actions.shape[1])
        if costs.shape != expected:
            raise ValueError(
                f'ensemble costs have shape {costs.shape}, expected {expected}'
            )
        return self.fractional_ranks(costs).mean(dim=1)
