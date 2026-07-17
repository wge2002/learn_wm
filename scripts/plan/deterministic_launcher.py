"""Run a Python script under a strict reproducibility configuration.

Usage:
    PYTHONHASHSEED=42 SWM_SEED=42 \
      python scripts/plan/deterministic_launcher.py \
        scripts/plan/eval_wm.py [ARGS ...]

Unlike a bare ``runpy.run_path`` call, this launcher mirrors Python's script
startup semantics: ``sys.argv[0]`` points at the target and the target's
directory is placed at the front of ``sys.path``.  The latter is required by
planning scripts that import sibling modules such as ``eval_wm``.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
import runpy
import sys


# Python reads PYTHONHASHSEED only at interpreter startup.  Re-exec once when
# needed so cross-process audits do not inherit a randomized hash seed.
seed_text = os.environ.get("SWM_SEED", "42")
if (
    os.environ.get("PYTHONHASHSEED") != seed_text
    and os.environ.get("_SWM_DETERMINISTIC_REEXEC") != "1"
):
    reexec_env = os.environ.copy()
    reexec_env["PYTHONHASHSEED"] = seed_text
    reexec_env["_SWM_DETERMINISTIC_REEXEC"] = "1"
    os.execvpe(
        sys.executable,
        [sys.executable, *sys.argv],
        reexec_env,
    )

# This variable must be set before CUDA creates a cuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: deterministic_launcher.py TARGET [ARGS ...]"
        )

    seed = int(os.environ.get("SWM_SEED", "42"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    target_args = sys.argv[2:]
    sys.argv = [str(target), *target_args]
    sys.path.insert(0, str(target.parent))

    print(
        "[deterministic-launcher] "
        f"seed={seed} cublas={os.environ['CUBLAS_WORKSPACE_CONFIG']} "
        "tf32=False deterministic_algorithms=True "
        f"target={target}",
        flush=True,
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
