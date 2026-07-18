"""Per-candidate recorders for rank-oracle and CEM-selection audits.

Records the FINAL CEM iteration's full candidate set and predicted costs per
env, so an oracle harness can re-execute stratified candidates in a cloned
simulator and measure predicted-vs-true rank inversion. Solver-agnostic as
long as the solver passes (step, candidates, costs) to callbacks.
"""

import torch

from .common import Callback


class CandidateRecorder(Callback):
    """Keep the last iteration's candidates and predicted costs.

    history entries (one per solved batch): dict with
      'candidates': (B, N, H, D) float16 cpu tensor (normalized action space)
      'costs':      (B, N) float32 cpu tensor (predicted)
    """

    name = 'candidates'
    output_key = 'candidates'

    def __init__(self, n_steps: int) -> None:
        super().__init__(reduction='none')
        self.n_steps = int(n_steps)

    def __call__(self, *, step, candidates, costs, **kw) -> None:
        if step == self.n_steps - 1:
            self._current.append(
                {
                    'candidates': candidates.detach().to('cpu', torch.float16),
                    'costs': costs.detach().float().cpu(),
                }
            )


class CEMPopulationRecorder(Callback):
    """Record full populations at selected CEM iterations.

    This recorder is intentionally opt-in because full populations are much
    larger than scalar callback histories. Candidate tensors and distribution
    parameters are quantized to float16 on CPU; costs stay float32 so the
    same serialized candidates can be re-scored by every audit model.

    Each history entry contains:

    ``step``
        Zero-based CEM iteration.
    ``candidates``
        ``(B, N, H, D)`` normalized action candidates.
    ``costs``
        ``(B, N)`` predicted costs from the generator model.
    ``topk_inds``
        ``(B, K)`` elite indices into ``candidates``.
    ``mean`` / ``var``
        Updated CEM distribution parameters after the elite refit.
    ``prev_mean`` / ``prev_var``
        Distribution parameters from which this population was sampled.
    """

    name = 'population_trace'
    output_key = 'population_trace'

    def __init__(self, steps: list[int] | tuple[int, ...] | set[int]) -> None:
        super().__init__(reduction='none')
        normalized = sorted({int(step) for step in steps})
        if not normalized or normalized[0] < 0:
            raise ValueError('steps must contain non-negative CEM iterations')
        self.steps = frozenset(normalized)

    @staticmethod
    def _half_cpu(value: torch.Tensor) -> torch.Tensor:
        return value.detach().to('cpu', torch.float16)

    def __call__(
        self,
        *,
        step,
        candidates,
        costs,
        topk_inds,
        mean,
        var,
        prev_mean,
        prev_var,
        **kw,
    ) -> None:
        if int(step) not in self.steps:
            return
        self._current.append(
            {
                'step': int(step),
                'candidates': self._half_cpu(candidates),
                'costs': costs.detach().float().cpu(),
                'topk_inds': topk_inds.detach().to('cpu', torch.int32),
                'mean': self._half_cpu(mean),
                'var': self._half_cpu(var),
                'prev_mean': self._half_cpu(prev_mean),
                'prev_var': self._half_cpu(prev_var),
            }
        )


class CEMArchiveRecorder(Callback):
    """Keep selected CEM iterates for post-search validation.

    Unlike :class:`CEMPopulationRecorder`, this callback does not retain the
    full trace metadata. By default its archive contains only updated means
    and elites and is small enough to use at every MPC replan. The sampled
    population can be enabled explicitly for a generate-then-validate
    intervention in which a second model scores a final proposal bank once,
    without shaping the CEM path that produced it.
    """

    name = 'cem_archive'
    output_key = 'cem_archive'

    def __init__(
        self,
        steps: list[int] | tuple[int, ...] | set[int],
        *,
        record_population: bool = False,
    ) -> None:
        super().__init__(reduction='none')
        normalized = sorted({int(step) for step in steps})
        if not normalized or normalized[0] < 0:
            raise ValueError('steps must contain non-negative CEM iterations')
        self.steps = tuple(normalized)
        self._selected = frozenset(normalized)
        self.record_population = bool(record_population)

    def __call__(
        self,
        *,
        step,
        mean,
        topk_candidates,
        topk_vals,
        candidates,
        **kw,
    ) -> None:
        if int(step) not in self._selected:
            return
        entry = {
            'step': int(step),
            'mean': mean.detach().float().cpu(),
            'topk_candidates': topk_candidates.detach().float().cpu(),
            'topk_vals': topk_vals.detach().float().cpu(),
        }
        if self.record_population:
            entry['candidates'] = candidates.detach().float().cpu()
        self._current.append(entry)
