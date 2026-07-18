"""Post-hoc validation of a CEM optimization path with a held-out model."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np
import torch

from .callbacks import CEMArchiveRecorder


SelectionSpace = Literal['means', 'elites', 'populations', 'refit']


def slice_info(info_dict: dict, start: int, end: int) -> dict:
    sliced = {}
    for key, value in info_dict.items():
        try:
            sliced[key] = value[start:end]
        except (TypeError, IndexError):
            sliced[key] = value
    return sliced


def score_action_candidates(
    model,
    info_dict: dict,
    candidates: torch.Tensor,
    *,
    env_batch_size: int = 1,
) -> torch.Tensor:
    """Score CPU action banks with a world model in environment batches."""
    try:
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
    except (AttributeError, StopIteration):
        device = candidates.device
        dtype = candidates.dtype

    score_batches = []
    for start in range(0, len(candidates), env_batch_size):
        end = min(start + env_batch_size, len(candidates))
        candidate_batch = candidates[start:end].to(
            device=device,
            dtype=dtype,
        )
        n_candidates = candidate_batch.shape[1]
        info_batch = slice_info(info_dict, start, end)
        expanded = {}
        for key, value in info_batch.items():
            if torch.is_tensor(value):
                target_dtype = dtype if value.is_floating_point() else None
                value = value.to(device=device, dtype=target_dtype)
                expanded[key] = value.unsqueeze(1).expand(
                    len(candidate_batch),
                    n_candidates,
                    *value.shape[1:],
                )
            elif isinstance(value, np.ndarray):
                expanded[key] = np.repeat(
                    value[:, None, ...],
                    n_candidates,
                    axis=1,
                )
            else:
                expanded[key] = value

        scores = model.get_cost(expanded, candidate_batch)
        expected = (len(candidate_batch), n_candidates)
        if not isinstance(scores, torch.Tensor) or scores.shape != expected:
            raise ValueError(
                f'verifier cost has shape '
                f'{getattr(scores, "shape", None)}, expected {expected}'
            )
        score_batches.append(scores.detach().float().cpu())
    return torch.cat(score_batches)


class CrossValidatedCEMSolver:
    """Validate sparse CEM iterates without letting the verifier shape search.

    The wrapped CEM solver generates its usual adaptive proposal path. Updated
    means or generator-selected elites from a small set of iterations are then
    scored once by a second world model. This role separation is deliberate:
    the verifier judges actions selected by the proposer but is not itself
    queried on every population and optimized against for all CEM iterations.

    Args:
        solver: Configured-compatible CEM solver with a ``callbacks`` list.
        verifier: Held-out model implementing ``get_cost``.
        steps: Zero-based CEM iterations to archive.
        selection_space: Validate updated means, proposer-selected elites, or
            complete sampled populations from the archived rounds.
        verifier_batch_size: Number of environments scored together.
        refit_topk: For ``selection_space='refit'``, return the mean of this
            many verifier-selected candidates instead of one sampled action.
    """

    def __init__(
        self,
        solver,
        verifier,
        steps: list[int] | tuple[int, ...] | set[int],
        selection_space: SelectionSpace = 'means',
        verifier_batch_size: int = 1,
        refit_topk: int = 30,
    ) -> None:
        if selection_space not in (
            'means',
            'elites',
            'populations',
            'refit',
        ):
            raise ValueError(
                "selection_space must be 'means', 'elites', 'populations', "
                "or 'refit'"
            )
        if verifier_batch_size < 1:
            raise ValueError('verifier_batch_size must be positive')
        if refit_topk < 1:
            raise ValueError('refit_topk must be positive')
        if not hasattr(solver, 'callbacks'):
            raise TypeError('wrapped solver must expose a callbacks list')

        self.solver = solver
        self.verifier = verifier
        self.selection_space = selection_space
        self.verifier_batch_size = int(verifier_batch_size)
        self.refit_topk = int(refit_topk)
        self.selection_history: list[dict[str, torch.Tensor]] = []

        archives = [
            callback
            for callback in solver.callbacks
            if isinstance(callback, CEMArchiveRecorder)
        ]
        if len(archives) > 1:
            raise ValueError('wrapped solver has multiple CEM archives')
        requested_steps = tuple(sorted({int(step) for step in steps}))
        if archives:
            self.recorder = archives[0]
            if self.recorder.steps != requested_steps:
                raise ValueError(
                    'existing CEM archive uses different steps: '
                    f'{self.recorder.steps} != {requested_steps}'
                )
            if (
                selection_space in ('populations', 'refit')
                and not self.recorder.record_population
            ):
                raise ValueError(
                    'existing CEM archive does not record populations'
                )
        else:
            self.recorder = CEMArchiveRecorder(
                requested_steps,
                record_population=selection_space in ('populations', 'refit'),
            )
            solver.callbacks.append(self.recorder)

        n_steps = getattr(solver, 'n_steps', None)
        if n_steps is not None and self.recorder.steps[-1] >= int(n_steps):
            raise ValueError(
                f'archive step {self.recorder.steps[-1]} is outside '
                f'the wrapped solver with {n_steps} iterations'
            )

    @property
    def n_envs(self) -> int:
        return self.solver.n_envs

    @property
    def action_dim(self) -> int:
        return self.solver.action_dim

    @property
    def horizon(self) -> int:
        return self.solver.horizon

    def configure(self, **kwargs: Any) -> None:
        self.solver.configure(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    def selection_summary(self) -> dict:
        """Return aggregate validation choices across closed-loop replans."""
        if not self.selection_history:
            return {
                'num_plans': 0,
                'step_counts': {},
                'mean_selected_step': float('nan'),
            }
        selected_steps = torch.cat(
            [entry['selected_step'] for entry in self.selection_history]
        )
        unique, counts = torch.unique(selected_steps, return_counts=True)
        return {
            'num_plans': int(len(selected_steps)),
            'step_counts': {
                int(step): int(count)
                for step, count in zip(unique, counts, strict=True)
            },
            'mean_selected_step': float(selected_steps.float().mean()),
        }

    def _archive_candidates(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_candidates = []
        batch_steps = []
        expected_steps = list(self.recorder.steps)

        for batch_i, trace in enumerate(self.recorder.history):
            actual_steps = [int(entry['step']) for entry in trace]
            if actual_steps != expected_steps:
                raise RuntimeError(
                    f'archive batch {batch_i} has steps {actual_steps}, '
                    f'expected {expected_steps}'
                )

            if self.selection_space == 'means':
                candidates = torch.stack(
                    [entry['mean'] for entry in trace],
                    dim=1,
                )
                steps = torch.as_tensor(expected_steps, dtype=torch.int64)
            elif self.selection_space == 'elites':
                elite_counts = {
                    int(entry['topk_candidates'].shape[1]) for entry in trace
                }
                if len(elite_counts) != 1:
                    raise RuntimeError(
                        f'archive batch {batch_i} changes elite count'
                    )
                elite_count = elite_counts.pop()
                candidates = torch.cat(
                    [entry['topk_candidates'] for entry in trace],
                    dim=1,
                )
                steps = torch.as_tensor(
                    expected_steps,
                    dtype=torch.int64,
                ).repeat_interleave(elite_count)
            else:
                population_counts = {
                    int(entry['candidates'].shape[1]) for entry in trace
                }
                if len(population_counts) != 1:
                    raise RuntimeError(
                        f'archive batch {batch_i} changes population size'
                    )
                population_count = population_counts.pop()
                candidates = torch.cat(
                    [entry['candidates'] for entry in trace],
                    dim=1,
                )
                steps = torch.as_tensor(
                    expected_steps,
                    dtype=torch.int64,
                ).repeat_interleave(population_count)

            batch_candidates.append(candidates)
            batch_steps.append(
                steps.unsqueeze(0).expand(candidates.shape[0], -1)
            )

        if not batch_candidates:
            raise RuntimeError('CEM archive is empty after solve')
        return torch.cat(batch_candidates), torch.cat(batch_steps)

    def _score(
        self,
        info_dict: dict,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        return score_action_candidates(
            self.verifier,
            info_dict,
            candidates,
            env_batch_size=self.verifier_batch_size,
        )

    @torch.inference_mode()
    def solve(
        self,
        info_dict: dict,
        init_action: torch.Tensor | None = None,
    ) -> dict:
        started = time.time()
        outputs = self.solver(info_dict, init_action=init_action)
        candidates, candidate_steps = self._archive_candidates()
        verifier_costs = self._score(info_dict, candidates)
        selected = verifier_costs.argmin(dim=1)
        batch = torch.arange(len(selected))
        proposer_actions = outputs['actions']
        selected_indices = selected[:, None]
        if self.selection_space == 'refit':
            refit_count = min(self.refit_topk, candidates.shape[1])
            selected_indices = torch.topk(
                verifier_costs,
                k=refit_count,
                dim=1,
                largest=False,
            ).indices
            expanded_batch = batch[:, None].expand_as(selected_indices)
            outputs['actions'] = candidates[
                expanded_batch,
                selected_indices,
            ].mean(dim=1)
        else:
            outputs['actions'] = candidates[batch, selected]
        selected_step = candidate_steps[batch, selected]
        self.selection_history.append(
            {
                'selected_index': selected.clone(),
                'selected_indices': selected_indices.clone(),
                'selected_step': selected_step.clone(),
                'verifier_costs': verifier_costs.clone(),
            }
        )
        outputs['cross_validation'] = {
            'selection_space': self.selection_space,
            'steps': candidate_steps,
            'verifier_costs': verifier_costs,
            'selected_index': selected,
            'selected_indices': selected_indices,
            'selected_step': selected_step,
            'proposer_actions': proposer_actions,
            'elapsed_seconds': time.time() - started,
        }
        return outputs
