"""Evaluate the locked base-anchor adoption rule for sparse BP-OE.

The recursive branch-value selector chooses between two corrective branches.
This probe preserves the ordinary K3 plan as an explicit no-op anchor and
adopts the learned corrective branch iff its final K10 model cost is lower
than the anchor's K10 cost.  There is no fitted threshold or evaluation-set
hyperparameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oe_recursive_branch_value_selector import (
    load_shards,
    method_index,
    paired_summary,
    selector_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', type=Path, required=True)
    parser.add_argument(
        '--selector-predictions',
        type=Path,
        required=True,
    )
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--branch-method',
        default='bp_sparse_matched',
    )
    parser.add_argument(
        '--anchor-method',
        default='k3_1x300',
    )
    parser.add_argument(
        '--comparison-method',
        default='k3_1x300',
    )
    parser.add_argument('--bootstrap', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=20260720)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_shards(args.eval_dir)
    with np.load(
        args.selector_predictions,
        allow_pickle=False,
    ) as archive:
        prediction_rows = np.asarray(archive['eval_rows'])
        branch_index = np.asarray(
            archive['eval_chosen_index'],
            dtype=np.int16,
        )
    if not np.array_equal(data['rows'], prediction_rows):
        raise ValueError('selector predictions and Gate rows differ')

    anchor_i = method_index(data, args.anchor_method)
    comparison_i = method_index(data, args.comparison_method)
    branch_i = method_index(data, args.branch_method)
    primary_i = selector_index(data, 'primary')
    anchor = data['selected_true'][
        :, anchor_i, primary_i
    ]
    anchor_success = data['selected_success'][
        :, anchor_i, primary_i
    ]
    comparison = data['selected_true'][
        :, comparison_i, primary_i
    ]
    comparison_success = data['selected_success'][
        :, comparison_i, primary_i
    ]
    pair_cost = data['final_branch_true'][:, branch_i]
    pair_success = data['final_branch_success'][:, branch_i]
    learned_cost = pair_cost[
        np.arange(len(pair_cost)),
        branch_index,
    ]
    learned_success = pair_success[
        np.arange(len(pair_success)),
        branch_index,
    ]
    model_cost = data['final_model_cost']
    anchor_k10 = model_cost[:, anchor_i, 0, 1]
    learned_k10 = model_cost[
        np.arange(len(pair_cost)),
        branch_i,
        branch_index,
        1,
    ]

    adopt = learned_k10 < anchor_k10
    selected_cost = np.where(adopt, learned_cost, anchor)
    selected_success = np.where(
        adopt,
        learned_success,
        anchor_success,
    )
    oracle_cost = np.minimum(
        anchor,
        np.min(pair_cost, axis=1),
    )
    oracle_success = (
        anchor_success
        | np.any(pair_success, axis=1)
    )
    rng = np.random.default_rng(args.seed)
    report = {
        'version': 1,
        'rule': (
            'adopt learned BP branch iff its final K10 cost is lower '
            'than the ordinary K3 anchor final K10 cost'
        ),
        'threshold_fitted': False,
        'rows': int(len(data['rows'])),
        'adoption_rate': float(np.mean(adopt)),
        'anchor_method': args.anchor_method,
        'comparison_method': args.comparison_method,
        'anchor': {
            'mean_cost': float(np.mean(anchor)),
            'actual_success': float(np.mean(anchor_success)),
        },
        'comparison': {
            'mean_cost': float(np.mean(comparison)),
            'actual_success': float(np.mean(comparison_success)),
        },
        'learned_branch_without_anchor': {
            **paired_summary(
                learned_cost,
                comparison,
                bootstrap=args.bootstrap,
                rng=rng,
            ),
            'actual_success': float(np.mean(learned_success)),
        },
        'anchor_adoption_gate': {
            **paired_summary(
                selected_cost,
                comparison,
                bootstrap=args.bootstrap,
                rng=rng,
            ),
            'actual_success': float(np.mean(selected_success)),
        },
        'anchor_plus_two_branch_oracle': {
            **paired_summary(
                oracle_cost,
                comparison,
                bootstrap=args.bootstrap,
                rng=rng,
            ),
            'actual_success': float(np.mean(oracle_success)),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'anchor_report.json').write_text(
        json.dumps(report, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    np.savez_compressed(
        args.out / 'anchor_predictions.npz',
        rows=data['rows'],
        anchor=anchor,
        comparison=comparison,
        learned_branch=learned_cost,
        anchor_k10=anchor_k10,
        learned_branch_k10=learned_k10,
        adopt=adopt,
        selected=selected_cost,
        oracle=oracle_cost,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
