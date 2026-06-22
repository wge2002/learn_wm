#!/usr/bin/env python3
"""Run a compact LGHL warm_start ablation for PushT."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SUCCESS_RE = re.compile(r"['\"]success_rate['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)")


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_bool_list(raw: str) -> list[bool]:
    out = []
    for x in raw.split(","):
        x = x.strip().lower()
        if not x:
            continue
        if x in {"1", "true", "yes"}:
            out.append(True)
        elif x in {"0", "false", "no"}:
            out.append(False)
        else:
            raise argparse.ArgumentTypeError(f"invalid bool {x!r}")
    return out


def parse_shift(raw: str) -> tuple[str, list[str]]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError("shift must be LABEL:variation,...")
    label, values = raw.split(":", 1)
    return label, [v.strip() for v in values.split(",") if v.strip()]


def build_reset_override(variations: list[str]) -> str:
    if not variations:
        return "eval.reset_options=null"
    return f"eval.reset_options={{variation:[{','.join(variations)}]}}"


def parse_success_rate(text: str) -> float | None:
    matches = SUCCESS_RE.findall(text)
    return float(matches[-1]) if matches else None


def build_command(
    args,
    *,
    horizon: int,
    receding: int,
    warm_start: bool,
    shift_label: str,
    variations: list[str],
) -> list[str]:
    script = Path(args.repo) / "scripts" / "plan" / "eval_wm.py"
    warm_label = "warm" if warm_start else "cold"
    out_name = (
        f"lghl_warm_ablation_{args.env}_{shift_label}_{warm_label}"
        f"_h{horizon}_r{receding}_s{args.seed}.txt"
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
        f"eval.video={str(args.video).lower()}",
        f"plan_config.horizon={horizon}",
        f"plan_config.receding_horizon={receding}",
        f"plan_config.action_block={args.action_block}",
        f"+plan_config.warm_start={str(warm_start).lower()}",
        f"output.filename={out_name}",
        build_reset_override(variations),
    ]
    cmd.extend(args.extra_override)
    return cmd


def run_one(
    *,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    gpu: str | None,
    log_path: Path,
    timeout_sec: float,
) -> tuple[int, float | None, float]:
    start = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        gpu_prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu else ""
        log.write("$ " + gpu_prefix + " ".join(cmd) + "\n\n")
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec if timeout_sec > 0 else None,
            )
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = 124
            log.write(f"\n[TIMEOUT] exceeded {timeout_sec}s\n")
    text = log_path.read_text(errors="replace")
    return code, parse_success_rate(text), time.time() - start


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "env",
        "shift",
        "variations",
        "warm_start",
        "horizon",
        "receding_horizon",
        "action_block",
        "verify_gap_env_steps",
        "seed",
        "num_eval",
        "success_rate",
        "returncode",
        "elapsed_sec",
        "log_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--config-name", default="pusht")
    parser.add_argument("--policy", default="quentinll/lewm-pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train.h5")
    parser.add_argument("--env", default="pusht")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--receding", default="1,3,5")
    parser.add_argument("--warm-start", default="true,false")
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--threads-per-run", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=0)
    parser.add_argument("--output-dir", default="outputs/lghl_phase4_warm_start")
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--shift",
        action="append",
        type=parse_shift,
        default=None,
    )
    parser.add_argument("extra_override", nargs="*")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir)
    receding_values = parse_int_list(args.receding)
    warm_values = parse_bool_list(args.warm_start)
    shifts = args.shift or [
        ("id", []),
        ("visual", ["background.color", "agent.color", "block.color", "goal.color"]),
        ("geometry", ["block.scale", "agent.scale", "block.shape", "goal.scale"]),
    ]
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    gpu_pool: list[str | None] = gpus or [None]

    jobs = []
    for warm in warm_values:
        for shift_label, variations in shifts:
            for receding in receding_values:
                if receding > args.horizon:
                    continue
                jobs.append((warm, shift_label, variations, receding))

    rows: list[dict] = []
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = []
        for job_idx, (warm, shift_label, variations, receding) in enumerate(jobs):
            gpu = gpu_pool[job_idx % len(gpu_pool)]
            env = os.environ.copy()
            if gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu
            if args.threads_per_run > 0:
                env["SWM_TORCH_THREADS"] = str(args.threads_per_run)
                env.setdefault(
                    "OMP_NUM_THREADS", str(args.threads_per_run)
                )
            warm_label = "warm" if warm else "cold"
            log_path = (
                output_dir
                / "logs"
                / f"{shift_label}_{warm_label}_h{args.horizon}_r{receding}.log"
            )
            cmd = build_command(
                args,
                horizon=args.horizon,
                receding=receding,
                warm_start=warm,
                shift_label=shift_label,
                variations=variations,
            )
            fut = pool.submit(
                run_one,
                cmd=cmd,
                cwd=repo,
                env=env,
                gpu=gpu,
                log_path=log_path,
                timeout_sec=args.timeout_sec,
            )
            fut.meta = (gpu, warm, shift_label, variations, receding, log_path)
            futures.append(fut)

        for fut in as_completed(futures):
            gpu, warm, shift_label, variations, receding, log_path = fut.meta
            code, success, elapsed = fut.result()
            rows.append(
                {
                    "env": args.env,
                    "shift": shift_label,
                    "variations": ",".join(variations),
                    "warm_start": str(warm).lower(),
                    "horizon": args.horizon,
                    "receding_horizon": receding,
                    "action_block": args.action_block,
                    "verify_gap_env_steps": receding * args.action_block,
                    "seed": args.seed,
                    "num_eval": args.num_eval,
                    "success_rate": "" if success is None else success,
                    "returncode": code,
                    "elapsed_sec": elapsed,
                    "log_path": str(log_path),
                }
            )
            print(
                f"[warm] shift={shift_label} warm={warm} R={receding} "
                f"success={success} code={code} elapsed={elapsed:.1f}s",
                flush=True,
            )
            write_csv(output_dir / "phase4_warm_start_ablation.csv", rows)

    rows = sorted(
        rows,
        key=lambda r: (
            r["shift"],
            r["warm_start"],
            int(r["receding_horizon"]),
        ),
    )
    write_csv(output_dir / "phase4_warm_start_ablation.csv", rows)
    meta = {
        "elapsed_sec": time.time() - start_all,
        "num_jobs": len(jobs),
        "policy": args.policy,
        "dataset_name": args.dataset_name,
        "num_eval": args.num_eval,
        "horizon": args.horizon,
        "receding": receding_values,
        "warm_start": warm_values,
        "seed": args.seed,
    }
    (output_dir / "phase4_warm_start_metadata.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"[warm] wrote results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
