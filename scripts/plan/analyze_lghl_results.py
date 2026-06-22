#!/usr/bin/env python3
"""Combine, summarize, and plot LGHL sweep results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SHIFT_ORDER = ["id", "visual", "geometry"]
SHIFT_COLORS = {
    "id": "#2563eb",
    "visual": "#dc2626",
    "geometry": "#16a34a",
}


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("success_rate", "") == "":
                    continue
                row = dict(row)
                for key in [
                    "horizon",
                    "receding_horizon",
                    "action_block",
                    "verify_gap_env_steps",
                    "seed",
                    "returncode",
                ]:
                    row[key] = int(row[key])
                for key in ["success_rate", "elapsed_sec"]:
                    row[key] = float(row[key])
                rows.append(row)
    rows.sort(
        key=lambda r: (
            SHIFT_ORDER.index(r["shift"])
            if r["shift"] in SHIFT_ORDER
            else len(SHIFT_ORDER),
            r["horizon"],
            r["receding_horizon"],
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], fraction: float) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["env"], row["shift"], row["horizon"])].append(row)

    out = []
    for (env, shift, horizon), group in sorted(
        groups.items(),
        key=lambda item: (
            item[0][0],
            SHIFT_ORDER.index(item[0][1])
            if item[0][1] in SHIFT_ORDER
            else len(SHIFT_ORDER),
            item[0][2],
        ),
    ):
        group = sorted(group, key=lambda r: r["receding_horizon"])
        baseline = group[0]["success_rate"]
        threshold = baseline * fraction
        half_life = ""
        half_life_env_steps = ""
        for row in group:
            if row["success_rate"] <= threshold:
                half_life = row["receding_horizon"]
                half_life_env_steps = row["verify_gap_env_steps"]
                break

        successes = [row["success_rate"] for row in group]
        max_row = max(group, key=lambda r: r["success_rate"])
        min_row = min(group, key=lambda r: r["success_rate"])
        out.append(
            {
                "env": env,
                "shift": shift,
                "horizon": horizon,
                "baseline_receding_horizon": group[0]["receding_horizon"],
                "baseline_success_rate": round(baseline, 6),
                "drop_fraction": fraction,
                "threshold_success_rate": round(threshold, 6),
                "half_life_receding_horizon": half_life,
                "half_life_env_steps": half_life_env_steps,
                "mean_success_rate": round(sum(successes) / len(successes), 6),
                "min_success_rate": round(min_row["success_rate"], 6),
                "min_success_receding_horizon": min_row["receding_horizon"],
                "max_success_rate": round(max_row["success_rate"], 6),
                "max_success_receding_horizon": max_row["receding_horizon"],
                "success_at_max_receding_horizon": round(
                    group[-1]["success_rate"], 6
                ),
                "max_tested_receding_horizon": group[-1][
                    "receding_horizon"
                ],
                "num_points": len(group),
            }
        )
    return out


def compare_to_id(rows: list[dict]) -> list[dict]:
    by_cell = {
        (row["env"], row["horizon"], row["receding_horizon"], row["shift"]): row
        for row in rows
    }
    out = []
    for row in rows:
        if row["shift"] == "id":
            continue
        base = by_cell.get(
            (row["env"], row["horizon"], row["receding_horizon"], "id")
        )
        if base is None:
            continue
        out.append(
            {
                "env": row["env"],
                "shift": row["shift"],
                "horizon": row["horizon"],
                "receding_horizon": row["receding_horizon"],
                "verify_gap_env_steps": row["verify_gap_env_steps"],
                "id_success_rate": round(base["success_rate"], 6),
                "shift_success_rate": round(row["success_rate"], 6),
                "delta_vs_id": round(row["success_rate"] - base["success_rate"], 6),
                "ratio_vs_id": round(
                    row["success_rate"] / base["success_rate"], 6
                )
                if base["success_rate"]
                else "",
            }
        )
    return out


def rows_for(rows: list[dict], *, shift: str | None = None, horizon: int | None = None):
    out = rows
    if shift is not None:
        out = [row for row in out if row["shift"] == shift]
    if horizon is not None:
        out = [row for row in out if row["horizon"] == horizon]
    return sorted(out, key=lambda r: r["receding_horizon"])


def plot_success_curves(rows: list[dict], path: Path) -> None:
    horizons = sorted({row["horizon"] for row in rows})
    ncols = 3
    nrows = (len(horizons) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 7), sharey=True)
    axes = list(axes.flat)

    for ax, horizon in zip(axes, horizons):
        for shift in SHIFT_ORDER:
            group = rows_for(rows, shift=shift, horizon=horizon)
            if not group:
                continue
            ax.plot(
                [row["verify_gap_env_steps"] for row in group],
                [row["success_rate"] for row in group],
                marker="o",
                linewidth=2,
                label=shift,
                color=SHIFT_COLORS.get(shift),
            )
        ax.set_title(f"H={horizon}")
        ax.set_xlabel("verify gap (env steps)")
        ax.grid(True, alpha=0.25)
    for ax in axes[len(horizons) :]:
        ax.axis("off")
    axes[0].set_ylabel("success rate (%)")
    axes[0].legend(loc="lower right")
    fig.suptitle("LGHL success curves by planning horizon")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_shift_means(summary_rows: list[dict], path: Path) -> None:
    horizons = sorted({int(row["horizon"]) for row in summary_rows})
    x = list(range(len(horizons)))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, shift in enumerate(SHIFT_ORDER):
        values = []
        for horizon in horizons:
            row = next(
                (
                    row
                    for row in summary_rows
                    if row["shift"] == shift and int(row["horizon"]) == horizon
                ),
                None,
            )
            values.append(float(row["mean_success_rate"]) if row else 0.0)
        xs = [v + (offset - 1) * width for v in x]
        ax.bar(
            xs,
            values,
            width=width,
            label=shift,
            color=SHIFT_COLORS.get(shift),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_xlabel("planning horizon H")
    ax.set_ylabel("mean success over tested R (%)")
    ax.set_title("Average robustness over re-ground intervals")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_heatmaps(rows: list[dict], path: Path) -> None:
    horizons = sorted({row["horizon"] for row in rows})
    receding = sorted({row["receding_horizon"] for row in rows})
    fig, axes = plt.subplots(1, len(SHIFT_ORDER), figsize=(15, 4), sharey=True)
    for ax, shift in zip(axes, SHIFT_ORDER):
        lookup = {
            (row["horizon"], row["receding_horizon"]): row["success_rate"]
            for row in rows
            if row["shift"] == shift
        }
        matrix = []
        for horizon in horizons:
            matrix.append(
                [
                    lookup.get((horizon, r), float("nan"))
                    if r <= horizon
                    else float("nan")
                    for r in receding
                ]
            )
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="viridis")
        ax.set_title(shift)
        ax.set_xticks(range(len(receding)))
        ax.set_xticklabels(receding)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels(horizons)
        ax.set_xlabel("R")
        for i, horizon in enumerate(horizons):
            for j, r in enumerate(receding):
                value = lookup.get((horizon, r))
                if value is not None:
                    ax.text(j, i, f"{value:.0f}", ha="center", va="center", color="w")
    axes[0].set_ylabel("H")
    fig.subplots_adjust(right=0.9, wspace=0.18)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
    fig.colorbar(image, cax=cbar_ax, label="success rate (%)")
    fig.suptitle("Success-rate heatmaps")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--out-dir", default="outputs/lghl_analysis")
    parser.add_argument("--drop-fraction", type=float, default=0.5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows([Path(p) for p in args.input])
    result_fields = [
        "env",
        "shift",
        "variations",
        "horizon",
        "receding_horizon",
        "action_block",
        "verify_gap_env_steps",
        "seed",
        "success_rate",
        "returncode",
        "elapsed_sec",
    ]
    write_csv(out_dir / "lghl_combined_results.csv", rows, result_fields)

    summary = summarize(rows, args.drop_fraction)
    summary_fields = list(summary[0].keys())
    write_csv(out_dir / "lghl_combined_summary.csv", summary, summary_fields)

    comparison = compare_to_id(rows)
    comparison_fields = list(comparison[0].keys())
    write_csv(
        out_dir / "lghl_shift_vs_id.csv", comparison, comparison_fields
    )

    plot_success_curves(rows, out_dir / "lghl_success_curves.png")
    plot_shift_means(summary, out_dir / "lghl_mean_success_by_shift.png")
    plot_heatmaps(rows, out_dir / "lghl_success_heatmaps.png")

    print(f"Wrote analysis to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
