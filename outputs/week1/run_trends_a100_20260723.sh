#!/usr/bin/env bash
# Locked A100 execution for the 2026-trends verdict:
#   1. Direction-A K1+curvature / K1+bisim head-to-head.
#   2. Anchor+dose verification wave, under fresh run names.
#
# The HDF5 file is copied to tmpfs before this driver is started. Reading
# pixels directly from that file avoids the much slower GPFS pixel sidecar.
set -Eeuo pipefail

WORKTREE=${WORKTREE:-/225010117/code/learn_wm_h2h_20260723}
PY=${PY:-/225010117/code/learn_wm/.venv-clean/bin/python}
DS=${DS:-/dev/shm/pusht_expert_train.h5}
STABLEWM_HOME=${STABLEWM_HOME:-/225010117/stablewm}
RUN_ROOT=${RUN_ROOT:-/225010117/logs/trends_a100_20260723}
GPU_IDS=${GPU_IDS:-0,1,2,3}
NGPU=${NGPU:-4}
WORKERS=${WORKERS:-8}
PREFETCH=${PREFETCH:-2}
VERIFY_SUFFIX=${VERIFY_SUFFIX:-_v2_20260723}

export PYTHONPATH="$WORKTREE"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export STABLEWM_HOME
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-4}

mkdir -p "$RUN_ROOT/head2head" "$RUN_ROOT/verify"
cd "$WORKTREE"
test -x "$PY"
test -f "$DS"
bash -n outputs/week1/run_head2head.sh outputs/week1/run_verify_wave.sh

on_exit() {
  local rc=$?
  printf '%s rc=%s\n' "$(date --iso-8601=seconds)" "$rc" \
    > "$RUN_ROOT/driver.exit"
}
trap on_exit EXIT

printf '%s head-to-head start\n' "$(date --iso-8601=seconds)"
env \
  PY="$PY" DS="$DS" OUT="$RUN_ROOT/head2head" \
  NGPU="$NGPU" GPU_IDS="$GPU_IDS" \
  WORKERS="$WORKERS" PREFETCH="$PREFETCH" \
  PHASES=train,eval \
  GPU_IMAGE_PREPROCESS=true USE_PIXEL_SIDECAR=false \
  bash outputs/week1/run_head2head.sh
printf '%s head-to-head done\n' "$(date --iso-8601=seconds)"

printf '%s verify wave start\n' "$(date --iso-8601=seconds)"
env \
  PY="$PY" DS="$DS" OUT="$RUN_ROOT/verify" \
  NGPU="$NGPU" GPU_IDS="$GPU_IDS" \
  WORKERS="$WORKERS" PREFETCH="$PREFETCH" \
  PHASES=train,eval VERIFY_SUFFIX="$VERIFY_SUFFIX" \
  GPU_IMAGE_PREPROCESS=true USE_PIXEL_SIDECAR=false \
  bash outputs/week1/run_verify_wave.sh
printf '%s verify wave done\n' "$(date --iso-8601=seconds)"
