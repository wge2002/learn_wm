"""Per-candidate recorder for rank-oracle audits (Gate A/B, Week-1).

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
            self._current.append({
                'candidates': candidates.detach().to('cpu', torch.float16),
                'costs': costs.detach().float().cpu(),
            })
