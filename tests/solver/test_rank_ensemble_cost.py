import pytest
import torch
from torch import nn

from stable_worldmodel.solver import RankEnsembleCost


class DummyCost(nn.Module):
    def __init__(self, costs):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.costs = torch.as_tensor(costs, dtype=torch.float32)

    def get_cost(self, info_dict, actions):
        del info_dict
        return self.costs[: len(actions), : actions.shape[1]]


def test_fractional_ranks_are_scale_invariant():
    values = torch.tensor([[20.0, -10.0, 5.0]])
    scaled = 4.0 * values + 13.0
    expected = torch.tensor([[1.0, 0.0, 0.5]])
    assert torch.equal(
        RankEnsembleCost.fractional_ranks(values),
        expected,
    )
    assert torch.equal(
        RankEnsembleCost.fractional_ranks(scaled),
        expected,
    )


def test_rank_ensemble_averages_model_orderings():
    models = [
        DummyCost([[0.0, 2.0, 1.0]]),
        DummyCost([[2.0, 1.0, 0.0]]),
    ]
    ensemble = RankEnsembleCost(models, ['left', 'right'])
    actions = torch.zeros(1, 3, 2, 1)
    actual = ensemble.get_cost({}, actions)
    expected = torch.tensor([[0.5, 0.75, 0.25]])
    assert torch.equal(actual, expected)


def test_rank_ensemble_validates_names_and_output_shape():
    model = DummyCost([[0.0, 1.0]])
    with pytest.raises(ValueError, match='at least two'):
        RankEnsembleCost([model], ['only'])
    with pytest.raises(ValueError, match='unique'):
        RankEnsembleCost([model, model], ['same', 'same'])

    bad = DummyCost([[0.0]])
    ensemble = RankEnsembleCost([model, bad], ['good', 'bad'])
    with pytest.raises((RuntimeError, ValueError)):
        ensemble.get_cost({}, torch.zeros(1, 2, 1, 1))
