#!/usr/bin/env python3
"""Run and summarize a latent grounding half-life sweep for stable-worldmodel.

This script assumes it is run against a stable-worldmodel checkout. It sweeps
planning horizon H and receding horizon R, where R is the number of planned
steps executed before the policy is forced to re-ground on a fresh observation.

The stable-worldmodel patch in this workspace lets dataset-driven eval pass
``eval.reset_options`` into ``World.reset`` so FoV shifts persist during rollout.
Without that patch, the in-distribution sweep still works, but FoV shift labels
will not affect the environment.
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SUCCESS_RE = re.compile(r"['\"]success_rate['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_shift(raw: str) -> tuple[str, list[str]]:
    """Parse a shift spec like ``visual:background.color,agent.color``."""
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            "Shift must be LABEL:variation.a,variation.b or LABEL:"
        )
    label, values = raw.split(":", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Shift label cannot be empty.")
    variations = [v.strip() for v in values.split(",") if v.strip()]
    return label, variations


def build_reset_override(variations: list[str]) -> str:
    if not variations:
        return "eval.reset_options=null"
    body = ",".join(variations)
    return f"eval.reset_options={{variation:[{body}]}}"


def build_command(args, horizon: int, receding: int, shift_label: str, variations):
    script = Path(args.repo) / "scripts" / "plan" / "eval_wm.py"
    out_name = (
        f"lghl_{args.env}_{shift_label}_h{horizon}_r{receding}_s{args.seed}.txt"
    )
    cmd = [
        sys.executable,
        str(script),
        "--config-name",
        args.config_name,
        f"policy={args.policy}",
        f"seed={args.seed}",
        f"eval.num_eval={args.num_eval}",
        f"eval.dataset_name={args.dataset_name}",
        f"eval.goal_offset_steps={args.goal_offset}",
        f"eval.eval_budget={args.eval_budget}",
        f"plan_config.horizon={horizon}",
        f"plan_config.receding_horizon={receding}",
        f"plan_config.action_block={args.action_block}",
        f"output.filename={out_name}",
        build_reset_override(variations),
    ]
    cmd.extend(args.extra_override)
    return cmd


def parse_success_rate(stdout: str, stderr: str) -> float | None:
    text = stdout + "\n" + stderr
    matches = SUCCESS_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def tail_text(path: Path, n_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    return text[-n_chars:]


def format_command(cmd: list[str], gpu: str | None = None) -> str:
    prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
    return prefix + " ".join(cmd)


def run_one(
    cmd: list[str],
    cwd: Path,
    execute: bool,
    *,
    env: dict[str, str] | None = None,
    gpu: str | None = None,
    log_path: Path | None = None,
    timeout_sec: float = 0,
) -> tuple[int, float | None, float]:
    if not execute:
        print(format_command(cmd, gpu))
        return 0, None, 0.0

    start = time.time()
    if log_path is None:
        log_path = Path("outputs/lghl_logs/run.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timeout = timeout_sec if timeout_sec > 0 else None

    with log_path.open("w") as log:
        log.write(f"$ {format_command(cmd, gpu)}\n\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = 124
            log.write(
                f"\n[TIMEOUT] Command exceeded --timeout-sec={timeout_sec}\n"
            )

    elapsed = time.time() - start
    text = log_path.read_text(errors="replace")
    success_rate = parse_success_rate(text, "")

    if returncode != 0 or success_rate is None:
        print("=" * 80)
        print("Command failed or success_rate was not found:")
        print(format_command(cmd, gpu))
        print(f"--- log: {log_path} ---")
        print(tail_text(log_path))

    return returncode, success_rate, elapsed


def parse_gpu_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], fraction: float) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row["success_rate"] == "":
            continue
        groups[(row["env"], row["shift"], row["horizon"])].append(row)

    summary = []
    for (env, shift, horizon), group in sorted(groups.items()):
        group = sorted(group, key=lambda r: int(r["receding_horizon"]))
        baseline = float(group[0]["success_rate"])
        threshold = baseline * fraction
        half_life = ""
        half_life_env_steps = ""
        for row in group:
            if float(row["success_rate"]) <= threshold:
                half_life = row["receding_horizon"]
                half_life_env_steps = row["verify_gap_env_steps"]
                break
        summary.append(
            {
                "env": env,
                "shift": shift,
                "horizon": horizon,
                "baseline_receding_horizon": group[0]["receding_horizon"],
                "baseline_success_rate": baseline,
                "drop_fraction": fraction,
                "threshold_success_rate": threshold,
                "half_life_receding_horizon": half_life,
                "half_life_env_steps": half_life_env_steps,
                "max_tested_receding_horizon": group[-1]["receding_horizon"],
            }
        )
    return summary


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env",
        "shift",
        "horizon",
        "baseline_receding_horizon",
        "baseline_success_rate",
        "drop_fraction",
        "threshold_success_rate",
        "half_life_receding_horizon",
        "half_life_env_steps",
        "max_tested_receding_horizon",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=".",
        help="Path to a stable-worldmodel checkout.",
    )
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--env", default="pusht")
    parser.add_argument("--policy", required=True, help="Checkpoint name or path.")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--horizons", default="1,2,3,5,8,10")
    parser.add_argument("--receding", default="1,2,3,5,8,10")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--shift",
        action="append",
        type=parse_shift,
        default=None,
        help="FoV shift as LABEL:var.a,var.b. Use LABEL: for no shift.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Extra Hydra override, can be repeated.",
    )
    parser.add_argument(
        "--drop-fraction",
        type=float,
        default=0.5,
        help="Half-life threshold as a fraction of the smallest-R success rate.",
    )
    parser.add_argument("--out", default="outputs/lghl_results.csv")
    parser.add_argument("--summary-out", default="outputs/lghl_summary.csv")
    parser.add_argument(
        "--log-dir",
        default="outputs/lghl_logs",
        help="Directory for per-run stdout/stderr logs.",
    )
    parser.add_argument(
        "--gpus",
        default="",
        help=(
            "Comma-separated physical GPU ids for parallel execution, e.g. "
            "0,6. Empty means run sequentially without setting "
            "CUDA_VISIBLE_DEVICES."
        ),
    )
    parser.add_argument(
        "--jobs-per-gpu",
        type=int,
        default=1,
        help="Number of concurrent eval subprocesses to schedule per GPU.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=0,
        help="Per-run timeout in seconds. 0 disables timeouts.",
    )
    parser.add_argument(
        "--threads-per-run",
        type=int,
        default=2,
        help=(
            "CPU thread cap for each eval subprocess. Sets OMP/MKL/OpenBLAS/"
            "NumExpr/VecLib variables and SWM_TORCH_THREADS."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands. Default is to print the planned commands.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    horizons = parse_int_list(args.horizons)
    receding_values = parse_int_list(args.receding)
    shifts = args.shift or [("id", [])]
    gpus = parse_gpu_list(args.gpus)
    rows = []
    tasks = []

    for shift_label, variations in shifts:
        for horizon in horizons:
            for receding in receding_values:
                if receding > horizon:
                    continue
                cmd = build_command(args, horizon, receding, shift_label, variations)
                row = {
                    "env": args.env,
                    "shift": shift_label,
                    "variations": ",".join(variations),
                    "horizon": horizon,
                    "receding_horizon": receding,
                    "action_block": args.action_block,
                    "verify_gap_env_steps": receding * args.action_block,
                    "seed": args.seed,
                    "success_rate": "",
                    "returncode": "",
                    "elapsed_sec": "",
                }
                log_name = (
                    f"lghl_{args.env}_{shift_label}_h{horizon}_"
                    f"r{receding}_s{args.seed}.log"
                )
                tasks.append((len(tasks), row, cmd, Path(args.log_dir) / log_name))

    def run_task(task, gpu: str | None = None):
        idx, row, cmd, log_path = task
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        if args.threads_per_run > 0:
            threads = str(args.threads_per_run)
            env.setdefault("OMP_NUM_THREADS", threads)
            env.setdefault("MKL_NUM_THREADS", threads)
            env.setdefault("OPENBLAS_NUM_THREADS", threads)
            env.setdefault("NUMEXPR_NUM_THREADS", threads)
            env.setdefault("VECLIB_MAXIMUM_THREADS", threads)
            env.setdefault("SWM_TORCH_THREADS", threads)
        env.setdefault("PYTHONUNBUFFERED", "1")

        if args.execute:
            print(
                f"[start] task={idx + 1}/{len(tasks)} gpu={gpu or '-'} "
                f"log={log_path}",
                flush=True,
            )
        returncode, success_rate, elapsed = run_one(
            cmd,
            repo,
            args.execute,
            env=env,
            gpu=gpu,
            log_path=log_path,
            timeout_sec=args.timeout_sec,
        )
        row = dict(row)
        row["success_rate"] = "" if success_rate is None else success_rate
        row["returncode"] = returncode
        row["elapsed_sec"] = round(elapsed, 3)
        if args.execute:
            print(
                f"[done] task={idx + 1}/{len(tasks)} gpu={gpu or '-'} "
                f"returncode={returncode} success_rate={row['success_rate']} "
                f"elapsed_sec={row['elapsed_sec']}",
                flush=True,
            )
        return idx, row

    if args.execute and gpus:
        if args.jobs_per_gpu < 1:
            raise ValueError("--jobs-per-gpu must be >= 1")
        gpu_slots: queue.Queue[str] = queue.Queue()
        for gpu in gpus:
            for _ in range(args.jobs_per_gpu):
                gpu_slots.put(gpu)

        def run_task_from_pool(task):
            gpu = gpu_slots.get()
            try:
                return run_task(task, gpu)
            finally:
                gpu_slots.put(gpu)

        max_workers = len(gpus) * args.jobs_per_gpu
        completed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(run_task_from_pool, task) for task in tasks]
            for future in as_completed(futures):
                completed.append(future.result())
        rows = [row for _, row in sorted(completed, key=lambda item: item[0])]
    else:
        for task in tasks:
            gpu = None
            if gpus:
                gpu = gpus[task[0] % len(gpus)]
            _, row = run_task(task, gpu)
            rows.append(row)

    out = Path(args.out)
    write_rows(out, rows)
    summary = summarize(rows, args.drop_fraction)
    write_summary(Path(args.summary_out), summary)
    print(f"Wrote rows to {out.resolve()}")
    print(f"Wrote summary to {Path(args.summary_out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
