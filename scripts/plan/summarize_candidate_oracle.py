"""Summarize a paired candidate-oracle matrix with reproducible CIs.

The input directory is expected to contain cells named ``h{H}_off{OFFSET}``.
Each cell must contain one ``k*_reference.npz`` result and any number of
``k*_paired.npz`` results produced by ``candidate_oracle.py``.

This script first verifies that every model in a cell was evaluated on the
same states, goals, action candidates, and simulator outcomes.  It then writes
per-model means and paired bootstrap comparisons.  Bootstrap resampling is
over planning states, so the confidence intervals preserve the paired design.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import re

import numpy as np


CELL_RE = re.compile(r"^h(?P<horizon>\d+)_off(?P<offset>\d+)$")
RESULT_RE = re.compile(r"^k(?P<k>\d+)_(?:reference|paired)\.npz$")

# direction=1 means larger is better; direction=-1 means smaller is better.
METRICS = {
    "spearman": 1,
    "kendall": 1,
    "inversion": -1,
    "topk_precision": 1,
    "regret": -1,
    "normalized_regret": -1,
    "selected_true_percentile": -1,
    "success_hit": 1,
}
PAIRED_KEYS = (
    "rows",
    "episodes",
    "starts",
    "candidates",
    "true",
    "true_pos_l2",
    "true_angle",
    "success",
    "terminal_state",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--reference-k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Defaults to MATRIX/summary.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def arrays_equal(left: np.ndarray, right: np.ndarray) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if np.issubdtype(left.dtype, np.inexact):
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def bootstrap_ci(
    values: np.ndarray,
    indices: np.ndarray,
) -> tuple[float, float]:
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    sampled = values[indices]
    finite = np.isfinite(sampled)
    counts = finite.sum(axis=1)
    means = np.divide(
        np.where(finite, sampled, 0.0).sum(axis=1),
        counts,
        out=np.full(len(sampled), np.nan),
        where=counts > 0,
    )
    means = means[counts > 0]
    if not len(means):
        return float("nan"), float("nan")
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def metric_values(result: dict[str, np.ndarray], metric: str) -> np.ndarray:
    if metric in result:
        return np.asarray(result[metric], dtype=np.float64)

    predicted = np.asarray(result["pred"], dtype=np.float64)
    true = np.asarray(result["true"], dtype=np.float64)
    selected = np.argmin(predicted, axis=1)
    row = np.arange(len(selected))

    if metric == "normalized_regret":
        chosen = true[row, selected]
        best = np.min(true, axis=1)
        span = np.max(true, axis=1) - best
        return np.divide(
            chosen - best,
            span,
            out=np.zeros_like(chosen),
            where=span > 0,
        )
    if metric == "selected_true_percentile":
        n_candidates = true.shape[1]
        if n_candidates == 1:
            return np.zeros(len(selected), dtype=np.float64)
        ranks = np.argsort(
            np.argsort(true, axis=1, kind="stable"),
            axis=1,
            kind="stable",
        )
        return ranks[row, selected] / (n_candidates - 1)
    if metric == "success_hit":
        success = np.asarray(result["success"], dtype=bool)
        values = success[row, selected].astype(np.float64)
        values[~success.any(axis=1)] = np.nan
        return values
    raise KeyError(metric)


def finite_mean_std(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan"), float("nan")
    std = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    return float(np.mean(finite)), std


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    matrix = args.matrix.resolve()
    out_dir = (args.out_dir or matrix / "summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cells = []
    for path in matrix.iterdir():
        match = CELL_RE.match(path.name)
        if path.is_dir() and match:
            cells.append(
                (
                    int(match.group("horizon")),
                    int(match.group("offset")),
                    path,
                )
            )
    cells.sort()
    if not cells:
        raise ValueError(f"No h{{H}}_off{{OFFSET}} cells found in {matrix}")

    summary_rows: list[dict] = []
    pairwise_rows: list[dict] = []
    audit_cells: list[dict] = []

    for horizon, offset, cell_path in cells:
        results: dict[int, dict[str, np.ndarray]] = {}
        sources: dict[int, str] = {}
        for result_path in sorted(cell_path.glob("k*.npz")):
            match = RESULT_RE.match(result_path.name)
            if not match:
                continue
            k_train = int(match.group("k"))
            if k_train in results:
                raise ValueError(
                    f"Duplicate K={k_train} results in {cell_path}"
                )
            results[k_train] = load_result(result_path)
            sources[k_train] = result_path.name

        if args.reference_k not in results:
            raise ValueError(
                f"Missing reference K={args.reference_k} in {cell_path}"
            )
        declared_references = [
            k_train
            for k_train, source in sources.items()
            if source.endswith("_reference.npz")
        ]
        if declared_references != [args.reference_k]:
            raise ValueError(
                f"{cell_path} declares reference models "
                f"{declared_references}, expected [{args.reference_k}]"
            )
        if len(results) < 2:
            raise ValueError(f"Need at least two models in {cell_path}")

        reference = results[args.reference_k]
        n_states = int(reference["rows"].shape[0])
        boot_indices = rng.integers(
            0,
            n_states,
            size=(args.bootstrap, n_states),
        )

        for k_train, result in results.items():
            for key in PAIRED_KEYS:
                if key not in reference or key not in result:
                    raise ValueError(f"Missing paired key {key} in {cell_path}")
                if not arrays_equal(reference[key], result[key]):
                    raise ValueError(
                        f"Paired audit failed for {cell_path.name}, "
                        f"K={k_train}, key={key}"
                    )
            for metric in METRICS:
                values = metric_values(result, metric)
                if values.shape != (n_states,):
                    raise ValueError(
                        f"{cell_path.name} K={k_train} {metric} has "
                        f"shape {values.shape}, expected {(n_states,)}"
                    )
                mean, std = finite_mean_std(values)
                summary_rows.append(
                    {
                        "horizon": horizon,
                        "goal_offset": offset,
                        "k_train": k_train,
                        "metric": metric,
                        "mean": mean,
                        "std": std,
                        "n_finite": int(np.isfinite(values).sum()),
                        "source": sources[k_train],
                    }
                )

        ks = sorted(results)
        for model_a_index, model_a in enumerate(ks):
            for model_b in ks[model_a_index + 1 :]:
                for metric, direction in METRICS.items():
                    values_a = metric_values(results[model_a], metric)
                    values_b = metric_values(results[model_b], metric)
                    raw_delta = values_b - values_a
                    advantage = direction * raw_delta
                    raw_mean, _ = finite_mean_std(raw_delta)
                    advantage_mean, _ = finite_mean_std(advantage)
                    ci_low, ci_high = bootstrap_ci(
                        advantage,
                        boot_indices,
                    )
                    pairwise_rows.append(
                        {
                            "horizon": horizon,
                            "goal_offset": offset,
                            "metric": metric,
                            "model_a": model_a,
                            "model_b": model_b,
                            "raw_delta_b_minus_a": raw_mean,
                            "advantage_b_minus_a": advantage_mean,
                            "advantage_ci_low": ci_low,
                            "advantage_ci_high": ci_high,
                            "significant": bool(
                                ci_low > 0.0 or ci_high < 0.0
                            ),
                        }
                    )

        audit_cells.append(
            {
                "horizon": horizon,
                "goal_offset": offset,
                "n_states": n_states,
                "models": ks,
                "paired_keys_exact": True,
            }
        )

    reference_rows = []
    for row in pairwise_rows:
        if args.reference_k not in (row["model_a"], row["model_b"]):
            continue
        normalized = dict(row)
        if row["model_a"] == args.reference_k:
            normalized["compared_k"] = row["model_b"]
            normalized["advantage_vs_reference"] = row[
                "advantage_b_minus_a"
            ]
            normalized["advantage_ci_low_vs_reference"] = row[
                "advantage_ci_low"
            ]
            normalized["advantage_ci_high_vs_reference"] = row[
                "advantage_ci_high"
            ]
        else:
            normalized["compared_k"] = row["model_a"]
            normalized["advantage_vs_reference"] = -row[
                "advantage_b_minus_a"
            ]
            normalized["advantage_ci_low_vs_reference"] = -row[
                "advantage_ci_high"
            ]
            normalized["advantage_ci_high_vs_reference"] = -row[
                "advantage_ci_low"
            ]
        reference_rows.append(normalized)

    winner_credits: dict[str, Counter] = {
        metric: Counter() for metric in METRICS
    }
    winner_inclusions: dict[str, Counter] = {
        metric: Counter() for metric in METRICS
    }
    for horizon, offset, _ in cells:
        for metric, direction in METRICS.items():
            candidates = [
                row
                for row in summary_rows
                if row["horizon"] == horizon
                and row["goal_offset"] == offset
                and row["metric"] == metric
                and np.isfinite(row["mean"])
            ]
            if not candidates:
                continue
            scores = [direction * row["mean"] for row in candidates]
            best = max(scores)
            winners = [
                row
                for row, score in zip(candidates, scores, strict=True)
                if np.isclose(score, best, rtol=1e-12, atol=1e-12)
            ]
            credit = 1.0 / len(winners)
            for winner in winners:
                k_train = winner["k_train"]
                winner_credits[metric][k_train] += credit
                winner_inclusions[metric][k_train] += 1

    write_csv(out_dir / "means.csv", summary_rows)
    write_csv(out_dir / "pairwise.csv", pairwise_rows)
    write_csv(out_dir / f"paired_vs_k{args.reference_k}.csv", reference_rows)
    audit = {
        "matrix": str(matrix),
        "reference_k": args.reference_k,
        "bootstrap_samples": args.bootstrap,
        "bootstrap_seed": args.seed,
        "cells": audit_cells,
        "winner_credits": {
            metric: dict(sorted(counts.items()))
            for metric, counts in winner_credits.items()
        },
        "winner_inclusions": {
            metric: dict(sorted(counts.items()))
            for metric, counts in winner_inclusions.items()
        },
    }
    (out_dir / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"audited {len(cells)} cells; outputs: {out_dir}")
    for metric, counts in winner_credits.items():
        rendered = ", ".join(
            f"K{k}={count:g}" for k, count in sorted(counts.items())
        )
        print(f"{metric} winner credits: {rendered}")


if __name__ == "__main__":
    main()
