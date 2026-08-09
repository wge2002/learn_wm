#!/usr/bin/env bash
# Seed-variance wave for the bisim (Invariant-JEPA) candidate, run under the
# non-finite gradient guard.
#
# WHY THIS WAVE EXISTS
#
# The h2hfix wave produced exactly one usable checkpoint: bisimfix_adapt_s1
# (bisim / raw / adaptive, seed 1) at val 0.128 / pred 0.006 / sigreg 1.355 --
# the best result on record, beating the pd_d192_k5_s7 anchor (0.135 / 0.009).
# It cannot be claimed, because the IDENTICAL config at seed 3072 died at epoch
# 14. With n=1 there is no way to tell a real effect from a lucky draw.
#
# The deaths turned out not to be an aux-regularizer defect at all. Root cause
# (measured 2026-08-07): `precision: bf16` has no GradScaler, so a single `inf`
# gradient element reaches `gradient_clip_val`, which computes
# `clip_coef = 1/inf = 0` and multiplies every gradient by it -- `inf * 0 = NaN`
# -- and AdamW writes that NaN into the parameter and its moments. One step
# later every parameter is NaN. `fp16` would have skipped the step for free.
# `NonFiniteGradGuardCallback` restores that behaviour, verified by injecting an
# inf into a live run (SWM_INJECT_INF_AT_STEP=5): the run survived and saved.
#
# So this wave re-measures the same recipe with the failure mode removed, and
# spends its GPUs on the number the paper actually needs -- the seed spread of
# the winning config -- rather than on more single-seed configurations.
#
# 6 bisim seeds + 2 curvature probes = 8 GPUs, one model per GPU (NOT DDP),
# 30 epochs, ~23 h at the measured 4.16 it/s (11306 steps/epoch).
#
#   bisimg_s0001  seed 1     CONTROL. Already completed WITHOUT the guard at
#                            0.128/0.006. The guard must not move it: a healthy
#                            run never triggers it, so a different result here
#                            would mean the guard perturbs training and the
#                            whole wave is suspect. Kept under a new name so the
#                            original checkpoint stays intact for comparison.
#   bisimg_s3072  seed 3072  the run that died at epoch 14; the direct test of
#                            whether the guard converts a death into a result.
#   bisimg_s0007  seed 7     \
#   bisimg_s0013  seed 13     > fresh seeds for the mean/variance
#   bisimg_s0042  seed 42    /
#   bisimg_s0101  seed 101   /
#
# Curvature gets 2 GPUs, not 4. Its problem is NOT only the NaN: curv_d192 ran
# all 30 epochs and still landed at val 5.18 / sigreg 55.7, i.e. representation
# collapse, which no gradient guard can fix. These two runs only answer "can it
# now finish at all", and are the two variants that survived longest in h2hfix
# (unit_adapt reached epoch 15, unit reached epoch 10).
#
# Usage: NGPU=8 bash outputs/week1/run_seedvar.sh
set -Eeuo pipefail
PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
PIXELS=${PIXELS:-${DS%.h5}_pixels.npy}
OUT=${OUT:-outputs/week1}
NGPU=${NGPU:-8}
GPU_IDS=${GPU_IDS:-}
WORKERS=${WORKERS:-6}
PREFETCH=${PREFETCH:-2}
PHASES=${PHASES:-train}
GPU_IMAGE_PREPROCESS=${GPU_IMAGE_PREPROCESS:-true}
USE_PIXEL_SIDECAR=${USE_PIXEL_SIDECAR:-false}
CKPT_ROOT=${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}
mkdir -p "$OUT"; test -f "$DS"; test "$NGPU" -ge 1
if [ -n "$GPU_IDS" ]; then
  IFS=',' read -r -a GPUS <<< "$GPU_IDS"
  [ "${#GPUS[@]}" -eq "$NGPU" ] || {
    echo "GPU_IDS must contain exactly NGPU=$NGPU comma-separated ids" >&2
    exit 2
  }
else
  GPUS=()
  for ((g=0; g<NGPU; g++)); do GPUS+=("$g"); done
fi
declare -a PIDS=() JOBS=() ARTIFACTS=()
I=0
wait_batch() { local f=0 i; for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}" \
      && { [ -z "${ARTIFACTS[$i]}" ] || [ -f "${ARTIFACTS[$i]}" ]; }; then
      echo "DONE ${JOBS[$i]} $(date -Is)"
    else
      echo "FAILED ${JOBS[$i]} $(date -Is)"
      f=1
    fi
  done; PIDS=(); JOBS=(); ARTIFACTS=(); return "$f"; }
has() { case ",$PHASES," in *,"$1",*) return 0;; *) return 1;; esac; }
use_pixel_sidecar() {
  case "$USE_PIXEL_SIDECAR" in 1|true|TRUE|yes|YES) return 0;; *) return 1;; esac
}

train() { # NAME SEED CONFIG AUX_SPACE BETA_MODE
  [ -f "$CKPT_ROOT/$1/weights_epoch_30.pt" ] && { echo "SKIP $1"; return; }
  local gpu=${GPUS[$((I % NGPU))]}
  local -a pixel_args=()
  use_pixel_sidecar && pixel_args=("+data.dataset.pixels_path=$PIXELS")
  echo "START train_$1 gpu=$gpu seed=$2 space=$4 beta_mode=$5 $(date -Is)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train/lewm.py --config-name "$3" \
    output_model_name="$1" subdir="$1" seed="$2" \
    wm.aux_space="$4" wm.aux_beta_mode="$5" \
    trainer.max_epochs=30 trainer.devices=1 \
    loader.num_workers="$WORKERS" loader.prefetch_factor="$PREFETCH" \
    data.dataset.name="$DS" gpu_image_preprocess="$GPU_IMAGE_PREPROCESS" \
    "${pixel_args[@]}" \
    > "$OUT/train_$1.log" 2>&1 &
  PIDS+=("$!"); JOBS+=("train_$1")
  ARTIFACTS+=("$CKPT_ROOT/$1/weights_epoch_30.pt"); I=$((I+1))
  if [ "${#PIDS[@]}" -eq "$NGPU" ]; then
    wait_batch
  fi
}

if has train; then
  use_pixel_sidecar && test -f "$PIXELS"
  # bisim raw+adaptive across 6 seeds: the number the paper needs
  train bisimg_s0001 1    lewm_bisim raw adaptive
  train bisimg_s3072 3072 lewm_bisim raw adaptive
  train bisimg_s0007 7    lewm_bisim raw adaptive
  train bisimg_s0013 13   lewm_bisim raw adaptive
  train bisimg_s0042 42   lewm_bisim raw adaptive
  train bisimg_s0101 101  lewm_bisim raw adaptive
  # curvature: does the guard let it finish at all?
  train curvg_unit_adapt 3072 lewm_curv unit adaptive
  train curvg_unit       3072 lewm_curv unit static
  [ "${#PIDS[@]}" -gt 0 ] && wait_batch; echo "SEEDVAR TRAININGS DONE $(date -Is)"
fi
echo "SEEDVAR DONE $(date -Is)"
