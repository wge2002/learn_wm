#!/usr/bin/env bash
# Anchor+dose verification wave (= Horizon-Bundle baseline-2 construction).
# Six trainings, then near/far-goal evaluation. Certificate scripts are
# optional because the historical outputs/pd and outputs/gauge utilities are
# not part of every checkout.
#
# Usage:
#   PHASES=train NGPU=3 bash outputs/week1/run_verify_wave.sh
#   PHASES=eval  NGPU=4 bash outputs/week1/run_verify_wave.sh
#   PHASES=train,eval NGPU=4 bash outputs/week1/run_verify_wave.sh
set -Eeuo pipefail

PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
PIXELS=${PIXELS:-${DS%.h5}_pixels.npy}
OUT=${OUT:-outputs/week1}
NGPU=${NGPU:-4}
GPU_IDS=${GPU_IDS:-}
WORKERS=${WORKERS:-2}
PREFETCH=${PREFETCH:-1}
PHASES=${PHASES:-train,eval}
GPU_IMAGE_PREPROCESS=${GPU_IMAGE_PREPROCESS:-false}
USE_PIXEL_SIDECAR=${USE_PIXEL_SIDECAR:-true}
VERIFY_SUFFIX=${VERIFY_SUFFIX:-}
CKPT_ROOT=${STABLEWM_HOME:-$HOME/.stable_worldmodel}/checkpoints
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}

mkdir -p "$OUT"
test -f "$DS"
test "$NGPU" -ge 1
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

declare -a PIDS=()
declare -a JOBS=()
declare -a ARTIFACTS=()
I=0

has_phase() {
  case ",$PHASES," in
    *,"$1",*) return 0 ;;
    *) return 1 ;;
  esac
}

use_pixel_sidecar() {
  case "$USE_PIXEL_SIDECAR" in
    1|true|TRUE|yes|YES) return 0 ;;
    *) return 1 ;;
  esac
}

wait_batch() {
  local failed=0
  local i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}" \
      && { [ -z "${ARTIFACTS[$i]}" ] || [ -f "${ARTIFACTS[$i]}" ]; }; then
      echo "DONE ${JOBS[$i]} $(date --iso-8601=seconds)"
    else
      echo "FAILED ${JOBS[$i]} $(date --iso-8601=seconds)"
      failed=1
    fi
  done
  PIDS=()
  JOBS=()
  ARTIFACTS=()
  return "$failed"
}

train() { # NAME SEED CONFIG [OVERRIDES...]
  local name=$1
  local seed=$2
  local config=$3
  shift 3
  local gpu=${GPUS[$((I % NGPU))]}
  local -a pixel_args=()
  use_pixel_sidecar && pixel_args=("+data.dataset.pixels_path=$PIXELS")

  if [ -f "$CKPT_ROOT/$name/weights_epoch_30.pt" ]; then
    echo "SKIP $name: epoch-30 checkpoint exists"
    return
  fi

  echo "START train_$name gpu=$gpu $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/train/lewm.py \
    --config-name "$config" \
    output_model_name="$name" subdir="$name" seed="$seed" \
    trainer.max_epochs=30 trainer.devices=1 \
    loader.num_workers="$WORKERS" loader.prefetch_factor="$PREFETCH" \
    data.dataset.name="$DS" gpu_image_preprocess="$GPU_IMAGE_PREPROCESS" \
    "${pixel_args[@]}" "$@" \
    > "$OUT/train_$name.log" 2>&1 &
  PIDS+=("$!")
  JOBS+=("train_$name")
  ARTIFACTS+=("$CKPT_ROOT/$name/weights_epoch_30.pt")
  I=$((I + 1))
  if [ "${#PIDS[@]}" -eq "$NGPU" ]; then
    wait_batch
  fi
}

eval_model() { # NAME OFF SEED
  local name=$1
  local offset=$2
  local seed=$3
  local gpu=${GPUS[$((I % NGPU))]}
  local policy
  local job="plan_${name}_off${offset}_s${seed}"

  if [ -f "$CKPT_ROOT/$name/weights_epoch_30.pt" ]; then
    policy="$name/weights_epoch_30.pt"
  elif [ -d "$CKPT_ROOT/${name}_eval" ]; then
    policy="${name}_eval"
  else
    echo "SKIP $job: checkpoint missing"
    return
  fi

  echo "START $job gpu=$gpu $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/plan/eval_wm.py \
    policy="$policy" seed="$seed" +plan_config.history_len=3 \
    eval.goal_offset_steps="$offset" eval.num_eval=50 eval.video=false \
    eval.dataset_name="$DS" \
    output.filename="verify_${name}_off${offset}_seed${seed}.txt" \
    > "$OUT/${job}.log" 2>&1 &
  PIDS+=("$!")
  JOBS+=("$job")
  ARTIFACTS+=("")
  I=$((I + 1))
  if [ "${#PIDS[@]}" -eq "$NGPU" ]; then
    wait_batch
  fi
}

if has_phase train; then
  use_pixel_sidecar && test -f "$PIXELS"
  train "mix_d192_g10_s1${VERIFY_SUFFIX}" 1 lewm_mix wm.mix_gamma=1.0
  train "mix_d192_g10_s7${VERIFY_SUFFIX}" 7 lewm_mix wm.mix_gamma=1.0
  train "mix_d192_g05${VERIFY_SUFFIX}" 3072 lewm_mix wm.mix_gamma=0.5
  train "mix_d192_g20${VERIFY_SUFFIX}" 3072 lewm_mix wm.mix_gamma=2.0
  train "pd_d192_k5_s7${VERIFY_SUFFIX}" 7 lewm_multistep
  train "pd_d192_k1_s7${VERIFY_SUFFIX}" 7 lewm data.dataset.num_steps=4
  if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
  fi
  echo "TRAININGS DONE $(date --iso-8601=seconds)"
fi

if has_phase eval; then
  NEW_MODELS="mix_d192_g10_s1${VERIFY_SUFFIX} mix_d192_g10_s7${VERIFY_SUFFIX} mix_d192_g05${VERIFY_SUFFIX} mix_d192_g20${VERIFY_SUFFIX} pd_d192_k5_s7${VERIFY_SUFFIX} pd_d192_k1_s7${VERIFY_SUFFIX}"
  EVAL_MODELS=${EVAL_MODELS:-"$NEW_MODELS mix_d192_g10 mix_d192_g03 mix_d192_g01"}
  for NAME in $EVAL_MODELS; do
    for OFF in 25 40; do
      for SEED in 42 123 7; do
        eval_model "$NAME" "$OFF" "$SEED"
      done
    done
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
  fi
  echo "EVALUATIONS DONE $(date --iso-8601=seconds)"
fi

if has_phase cert; then
  required=(
    scripts/plan/regime_stepB_eval_data.py
    outputs/gauge/jac_spectrum.py
    outputs/gauge/separation.py
    outputs/pd/probe_r2.py
  )
  for path in "${required[@]}"; do
    if [ ! -f "$path" ]; then
      echo "CERT PHASE UNAVAILABLE: missing $path"
      exit 2
    fi
  done
  echo "CERT PHASE PREFLIGHT PASSED; run the locked certificate protocol separately."
fi

echo "VERIFY WAVE DONE $(date --iso-8601=seconds)"
