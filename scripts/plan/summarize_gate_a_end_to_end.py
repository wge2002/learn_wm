"""Summarize the Gate-A end-to-end planning matrix.

The input directory should contain the text results written by ``eval_wm.py``
for the five training horizons, five planning horizons, three goal offsets,
and the ``fixcand`` / ``fixcalls`` protocols.  The evaluator uses a shared
seed and shared ordered start set, so episode success vectors can be compared
with paired resampling.

The script validates the complete 150-run design and writes:

* ``runs.csv``: every run, including the 50-bit episode success vector;
* ``horizon_summary.csv``: success averaged over the three goal offsets;
* ``cell_best_vs_runner.csv``: descriptive paired CIs within each cell;
* ``horizon_best_vs_runner.csv``: offset-stratified paired CIs by horizon;
* ``summary.json``: compact audit facts used by the temporal research log.

Best-vs-runner comparisons are descriptive because the two models are chosen
after observing the same data.  They are useful as a gate diagnostic, not as
unadjusted confirmatory tests.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import re
from statistics import mean


MODEL_TO_K = {
    "iter2_baseline": 1,
    "pd_d192_k2": 2,
    "pd_d192_k3": 3,
    "iter2_multistep": 5,
    "pd_d192_k10": 10,
}
KS = (1, 2, 3, 5, 10)
HORIZONS = (1, 3, 5, 8, 10)
OFFSETS = (25, 40, 60)
PROTOCOLS = ("fixcand", "fixcalls")

RESULT_RE = re.compile(
    r"^gateA_"
    r"(?P<model>iter2_baseline|pd_d192_k2|pd_d192_k3|"
    r"iter2_multistep|pd_d192_k10)"
    r"_h(?P<horizon>1|3|5|8|10)"
    r"_off(?P<offset>25|40|60)"
    r"_(?P<protocol>fixcand|fixcalls)\.txt$"
)
RATE_RE = re.compile(r"success_rate':\s*(?P<value>[0-9.]+)")
VECTOR_RE = re.compile(
    r"episode_successes': array\(\[(?P<values>.*?)\]\), 'seeds'",
    re.DOTALL,
)
SAMPLES_RE = re.compile(r"^\s*num_samples:\s*(?P<value>\d+)", re.MULTILINE)
TIME_RE = re.compile(r"evaluation_time:\s*(?P<value>[0-9.]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Defaults to RESULTS/summary.",
    )
    parser.add_argument(
        "--source-label",
        help="Stable provenance label stored instead of the local RESULTS path.",
    )
    parser.add_argument("--bootstrap", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_717)
    return parser.parse_args()


def required_match(pattern: re.Pattern[str], text: str, source: Path) -> re.Match:
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"{source}: missing pattern {pattern.pattern!r}")
    return match


def parse_result(path: Path) -> dict:
    name_match = RESULT_RE.match(path.name)
    if name_match is None:
        raise ValueError(f"Unexpected Gate-A filename: {path.name}")

    text = path.read_text(encoding="utf-8")
    rate = float(required_match(RATE_RE, text, path).group("value"))
    vector_text = required_match(VECTOR_RE, text, path).group("values")
    successes = tuple(value == "True" for value in re.findall(r"True|False", vector_text))
    if len(successes) != 50:
        raise ValueError(f"{path}: expected 50 episode outcomes, got {len(successes)}")

    measured_rate = 100.0 * sum(successes) / len(successes)
    if abs(rate - measured_rate) > 1e-8:
        raise ValueError(
            f"{path}: success_rate={rate} disagrees with vector rate={measured_rate}"
        )

    return {
        "protocol": name_match.group("protocol"),
        "horizon": int(name_match.group("horizon")),
        "goal_offset": int(name_match.group("offset")),
        "k_train": MODEL_TO_K[name_match.group("model")],
        "num_samples": int(required_match(SAMPLES_RE, text, path).group("value")),
        "success_rate": measured_rate,
        "n_success": sum(successes),
        "evaluation_time_seconds": float(
            required_match(TIME_RE, text, path).group("value")
        ),
        "successes": successes,
        "source": path.name,
    }


def quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap_ci(
    per_cluster_deltas: list[float],
    *,
    bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    n_clusters = len(per_cluster_deltas)
    samples = [
        mean(per_cluster_deltas[rng.randrange(n_clusters)] for _ in range(n_clusters))
        for _ in range(bootstrap)
    ]
    samples.sort()
    return quantile(samples, 0.025), quantile(samples, 0.975)


def stratified_paired_bootstrap_ci(
    strata: list[list[float]],
    *,
    bootstrap: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not strata or any(not stratum for stratum in strata):
        raise ValueError("Every bootstrap stratum must be non-empty")
    samples = []
    for _ in range(bootstrap):
        stratum_means = []
        for stratum in strata:
            size = len(stratum)
            stratum_means.append(
                mean(stratum[rng.randrange(size)] for _ in range(size))
            )
        samples.append(mean(stratum_means))
    samples.sort()
    return quantile(samples, 0.025), quantile(samples, 0.975)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def expected_samples(protocol: str, horizon: int) -> int:
    if protocol == "fixcand":
        return 300
    return 1500 // horizon


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1:
        raise ValueError("--bootstrap must be positive")

    result_dir = args.results.resolve()
    out_dir = (args.out_dir or result_dir / "summary").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        parse_result(path)
        for path in sorted(result_dir.glob("gateA_*.txt"))
        if RESULT_RE.match(path.name)
    ]
    expected_count = len(PROTOCOLS) * len(HORIZONS) * len(OFFSETS) * len(KS)
    if len(runs) != expected_count:
        raise ValueError(f"Expected {expected_count} matrix results, got {len(runs)}")

    index = {}
    for run in runs:
        key = (
            run["protocol"],
            run["horizon"],
            run["goal_offset"],
            run["k_train"],
        )
        if key in index:
            raise ValueError(f"Duplicate matrix cell: {key}")
        if run["num_samples"] != expected_samples(run["protocol"], run["horizon"]):
            raise ValueError(
                f"{run['source']}: unexpected num_samples={run['num_samples']}"
            )
        index[key] = run

    expected_keys = {
        (protocol, horizon, offset, k_train)
        for protocol in PROTOCOLS
        for horizon in HORIZONS
        for offset in OFFSETS
        for k_train in KS
    }
    missing = expected_keys - set(index)
    if missing:
        raise ValueError(f"Missing matrix cells: {sorted(missing)}")

    run_rows = []
    for run in runs:
        row = {key: value for key, value in run.items() if key != "successes"}
        row["success_bits"] = "".join("1" if value else "0" for value in run["successes"])
        run_rows.append(row)
    run_rows.sort(
        key=lambda row: (
            PROTOCOLS.index(row["protocol"]),
            row["horizon"],
            row["goal_offset"],
            row["k_train"],
        )
    )
    write_csv(out_dir / "runs.csv", run_rows)

    horizon_rows = []
    for protocol in PROTOCOLS:
        for horizon in HORIZONS:
            for k_train in KS:
                selected = [
                    index[protocol, horizon, offset, k_train]
                    for offset in OFFSETS
                ]
                n_success = sum(run["n_success"] for run in selected)
                n_trials = sum(len(run["successes"]) for run in selected)
                horizon_rows.append(
                    {
                        "protocol": protocol,
                        "horizon": horizon,
                        "k_train": k_train,
                        "success_rate": 100.0 * n_success / n_trials,
                        "n_success": n_success,
                        "n_trials": n_trials,
                    }
                )
    write_csv(out_dir / "horizon_summary.csv", horizon_rows)

    rng = random.Random(args.seed)
    cell_comparison_rows = []
    for protocol in PROTOCOLS:
        for horizon in HORIZONS:
            for offset in OFFSETS:
                ordered = sorted(
                    KS,
                    key=lambda k_train: (
                        index[protocol, horizon, offset, k_train]["success_rate"],
                        k_train,
                    ),
                    reverse=True,
                )
                winner, runner_up = ordered[:2]
                winner_run = index[protocol, horizon, offset, winner]
                runner_run = index[protocol, horizon, offset, runner_up]
                deltas = [
                    100.0 * (int(left) - int(right))
                    for left, right in zip(
                        winner_run["successes"],
                        runner_run["successes"],
                    )
                ]
                ci_low, ci_high = paired_bootstrap_ci(
                    deltas,
                    bootstrap=args.bootstrap,
                    rng=rng,
                )
                cell_comparison_rows.append(
                    {
                        "protocol": protocol,
                        "horizon": horizon,
                        "goal_offset": offset,
                        "winner_k": winner,
                        "winner_success_rate": winner_run["success_rate"],
                        "runner_up_k": runner_up,
                        "runner_up_success_rate": runner_run["success_rate"],
                        "advantage_pp": mean(deltas),
                        "ci_low_pp": ci_low,
                        "ci_high_pp": ci_high,
                        "descriptive_ci_excludes_zero": ci_low > 0.0,
                    }
                )
    write_csv(out_dir / "cell_best_vs_runner.csv", cell_comparison_rows)

    horizon_comparison_rows = []
    for protocol in PROTOCOLS:
        for horizon in HORIZONS:
            rates = {
                k_train: (
                    100.0
                    * sum(
                        index[
                            protocol,
                            horizon,
                            offset,
                            k_train,
                        ]["n_success"]
                        for offset in OFFSETS
                    )
                    / (len(OFFSETS) * 50)
                )
                for k_train in KS
            }
            ordered = sorted(
                KS,
                key=lambda k_train: (rates[k_train], k_train),
                reverse=True,
            )
            winner, runner_up = ordered[:2]
            # Models are paired within each offset. Different offsets use
            # different physical start rows, so resample the three offsets as
            # separate strata rather than pairing equal evaluator indices.
            offset_strata = [
                [
                    100.0
                    * (
                        int(
                            index[
                                protocol,
                                horizon,
                                offset,
                                winner,
                            ]["successes"][episode]
                        )
                        - int(
                            index[
                                protocol,
                                horizon,
                                offset,
                                runner_up,
                            ]["successes"][episode]
                        )
                    )
                    for episode in range(50)
                ]
                for offset in OFFSETS
            ]
            advantage = mean(mean(stratum) for stratum in offset_strata)
            ci_low, ci_high = stratified_paired_bootstrap_ci(
                offset_strata,
                bootstrap=args.bootstrap,
                rng=rng,
            )
            horizon_comparison_rows.append(
                {
                    "protocol": protocol,
                    "horizon": horizon,
                    "winner_k": winner,
                    "winner_success_rate": rates[winner],
                    "runner_up_k": runner_up,
                    "runner_up_success_rate": rates[runner_up],
                    "advantage_pp": advantage,
                    "ci_low_pp": ci_low,
                    "ci_high_pp": ci_high,
                    "descriptive_ci_excludes_zero": ci_low > 0.0,
                }
            )
    write_csv(out_dir / "horizon_best_vs_runner.csv", horizon_comparison_rows)

    winner_credits = {}
    for protocol in PROTOCOLS:
        credits = {str(k_train): 0.0 for k_train in KS}
        for horizon in HORIZONS:
            for offset in OFFSETS:
                rates = {
                    k_train: index[
                        protocol,
                        horizon,
                        offset,
                        k_train,
                    ]["success_rate"]
                    for k_train in KS
                }
                best = max(rates.values())
                winners = [
                    k_train for k_train, value in rates.items() if value == best
                ]
                for winner in winners:
                    credits[str(winner)] += 1.0 / len(winners)
        winner_credits[protocol] = credits

    global_success = {
        protocol: {
            str(k_train): mean(
                index[protocol, horizon, offset, k_train]["success_rate"]
                for horizon in HORIZONS
                for offset in OFFSETS
            )
            for k_train in KS
        }
        for protocol in PROTOCOLS
    }

    matched_horizon = {}
    for protocol in PROTOCOLS:
        deltas = []
        for horizon in (1, 3, 5, 10):
            for offset in OFFSETS:
                matched = index[protocol, horizon, offset, horizon]["success_rate"]
                best_other = max(
                    index[protocol, horizon, offset, k_train]["success_rate"]
                    for k_train in KS
                    if k_train != horizon
                )
                deltas.append(matched - best_other)
        matched_horizon[protocol] = {
            "mean_advantage_pp": mean(deltas),
            "positive_cells": sum(delta > 0.0 for delta in deltas),
            "tied_cells": sum(delta == 0.0 for delta in deltas),
            "negative_cells": sum(delta < 0.0 for delta in deltas),
            "n_cells": len(deltas),
        }

    h5_repeat_mismatches = []
    for offset in OFFSETS:
        for k_train in KS:
            fixed_candidates = index["fixcand", 5, offset, k_train]
            fixed_calls = index["fixcalls", 5, offset, k_train]
            differing_episodes = [
                episode
                for episode, (left, right) in enumerate(
                    zip(
                        fixed_candidates["successes"],
                        fixed_calls["successes"],
                    )
                )
                if left != right
            ]
            if differing_episodes:
                h5_repeat_mismatches.append(
                    {
                        "goal_offset": offset,
                        "k_train": k_train,
                        "fixcand_success_rate": fixed_candidates["success_rate"],
                        "fixcalls_success_rate": fixed_calls["success_rate"],
                        "differing_episode_indices": differing_episodes,
                    }
                )

    summary = {
        "source_directory": args.source_label or str(result_dir),
        "n_runs": len(runs),
        "n_failures": 0,
        "episodes_per_run": 50,
        "bootstrap": args.bootstrap,
        "bootstrap_seed": args.seed,
        "global_success_rate": global_success,
        "winner_credits_fractional": winner_credits,
        "matched_k_equals_h": matched_horizon,
        "cell_best_vs_runner_ci_excludes_zero": {
            protocol: sum(
                row["descriptive_ci_excludes_zero"]
                for row in cell_comparison_rows
                if row["protocol"] == protocol
            )
            for protocol in PROTOCOLS
        },
        "horizon_best_vs_runner_ci_excludes_zero": {
            protocol: sum(
                row["descriptive_ci_excludes_zero"]
                for row in horizon_comparison_rows
                if row["protocol"] == protocol
            )
            for protocol in PROTOCOLS
        },
        "h5_identical_configuration_repeat": {
            "n_cells": len(OFFSETS) * len(KS),
            "n_exact_vectors": len(OFFSETS) * len(KS) - len(h5_repeat_mismatches),
            "mismatches": h5_repeat_mismatches,
        },
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Validated {len(runs)} Gate-A runs")
    print(f"Wrote summaries to {out_dir}")
    for protocol in PROTOCOLS:
        rates = global_success[protocol]
        best = max(KS, key=lambda k_train: rates[str(k_train)])
        print(
            f"{protocol}: global best K={best}, "
            f"success={rates[str(best)]:.2f}%"
        )
    print(
        "H=5 repeated-config exact vectors: "
        f"{summary['h5_identical_configuration_repeat']['n_exact_vectors']}/"
        f"{summary['h5_identical_configuration_repeat']['n_cells']}"
    )


if __name__ == "__main__":
    main()
