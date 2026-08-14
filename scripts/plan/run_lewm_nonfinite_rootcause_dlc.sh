#!/usr/bin/env bash
# Two-GPU diagnosis of the v2 K1 non-finite gradients. One entry point, two
# modes with opposite success criteria on the same recipe, init and data:
#
#   MODE=repro      (default) expected FAILURE. Succeeds only if both training
#                   processes die through the strict guard and each writes one
#                   replay evidence bundle. Establishes the defect.
#   MODE=stability  expected SUCCESS. Adds encoder_fp32=true as the single
#                   intervention and succeeds only if both processes cross
#                   global_step 138000 -- past both historical failure points,
#                   115683 and 137496 -- with ZERO non-finite events and exit
#                   zero. Validates the fix.
#
# Sharing the entry point is the point: the two modes cannot drift in recipe,
# initialization, dataset or seed-to-GPU mapping. Neither mode touches the
# formal paired protocol and neither launches a formal run.
set -Eeo pipefail

RBS_DLC_WORKDIR=/mnt/home/gewang/code/learn_wm
. /mnt/home/gewang/.config/rbs-dlc/dlc_entry_prelude.sh
set -u

REPO=/mnt/home/gewang/code/learn_wm
PY=/mnt/home/gewang/venv-clean/bin/python
DS=/mnt/home/gewang/data/learn_wm/pusht_expert_train.h5
STABLEWM_ROOT=/mnt/home/gewang/swmhome/learn_wm
CKPT_ROOT=$STABLEWM_ROOT/checkpoints
INIT_ROOT=$CKPT_ROOT/paired_initializations/controlled_metric_paired_20260810
MODE=${MODE:-repro}
case "$MODE" in
  repro)
    CONFIG_NAME=lewm_nonfinite_v2_k1_repro
    DEFAULT_RUN_TAG=nonfinite_rootcause_v2k1_20260813_r1
    NAME_PREFIX=nfdiag_v2k1
    ;;
  stability)
    CONFIG_NAME=lewm_encoder_fp32_stability
    DEFAULT_RUN_TAG=encoder_fp32_stability_v2k1_20260814_r1
    NAME_PREFIX=fp32stab_v2k1
    ;;
  *)
    echo "MODE must be repro or stability, got '$MODE'" >&2
    exit 2
    ;;
esac
RUN_TAG=${RUN_TAG:-$DEFAULT_RUN_TAG}
OUT=${OUT:-$REPO/outputs/$RUN_TAG}
EPOCHS=${EPOCHS:-30}
# Both historical first-Inf steps are 115683 (seed 42) and 137496 (seed 13).
# The horizon must clear the later one, so 137496 is the hard floor asserted
# below and 138000 is the default with margin.
DIAGNOSTIC_STOP_AFTER_STEP=${DIAGNOSTIC_STOP_AFTER_STEP:-138000}
STABILITY_STOP_AFTER_STEP=${STABILITY_STOP_AFTER_STEP:-138000}
PREFLIGHT_MODE=${RBS_ROOTCAUSE_PREFLIGHT:-0}
ROOTCAUSE_GPU_IDS=${ROOTCAUSE_GPU_IDS:-"0 1"}
read -r GPU_SEED13 GPU_SEED42 extra_gpu <<< "$ROOTCAUSE_GPU_IDS"
if [ -z "${GPU_SEED13:-}" ] || [ -z "${GPU_SEED42:-}" ] \
  || [ -n "${extra_gpu:-}" ] || [ "$GPU_SEED13" = "$GPU_SEED42" ]; then
  echo "ROOTCAUSE_GPU_IDS must contain two distinct GPU indices" >&2
  exit 2
fi
# Exactly two seeds on exactly two GPUs, one independent single-device run per
# GPU. This is a 2-GPU request, not a node request: a third card would sit idle.
SPECS="13:$GPU_SEED13 42:$GPU_SEED42"
train_limit_args=()
if [ "$PREFLIGHT_MODE" = 1 ]; then
  EPOCHS=1
  SPECS="13:$GPU_SEED13"
  train_limit_args=(
    +trainer.limit_train_batches=2
    +trainer.limit_val_batches=1
  )
fi
export STABLEWM_HOME="$STABLEWM_ROOT"
export LOCAL_DATASET_DIR=${DS%/*}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWM_TORCH_THREADS=2
export HYDRA_FULL_ERROR=1

cd "$REPO"
test -x "$PY"
test -f "$DS"
if [ "$PREFLIGHT_MODE" != 1 ]; then
  # Keep the original 30-epoch horizon because the epoch-based cosine
  # scheduler depends on max_epochs. A shorter Trainer horizon changes the
  # learning-rate trajectory and cannot reproduce the historical failures.
  test "$EPOCHS" -eq 30
  if [ "$MODE" = repro ]; then
    test "$DIAGNOSTIC_STOP_AFTER_STEP" -gt 137496
  elif [ "$MODE" = stability ]; then
    test "$STABILITY_STOP_AFTER_STEP" -gt 137496
  fi
fi
visible_gpus=$(nvidia-smi -L | wc -l)
test "$visible_gpus" -ge 2
test "$GPU_SEED13" -ge 0
test "$GPU_SEED13" -lt "$visible_gpus"
test "$GPU_SEED42" -ge 0
test "$GPU_SEED42" -lt "$visible_gpus"
echo "root-cause GPU mapping: seed13=gpu$GPU_SEED13 seed42=gpu$GPU_SEED42"

git_safe=(git -c "safe.directory=$REPO")
current_commit=$("${git_safe[@]}" rev-parse HEAD)
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$current_commit" != "$EXPECTED_COMMIT" ]; then
  echo "expected commit $EXPECTED_COMMIT, found $current_commit" >&2
  exit 2
fi
if ! "${git_safe[@]}" diff --quiet || ! "${git_safe[@]}" diff --cached --quiet; then
  echo "tracked repository changes detected; refusing diagnosis" >&2
  exit 2
fi

mkdir -p "$OUT"
printf '%s\n' "$current_commit" > "$OUT/source_commit.txt"
"${git_safe[@]}" status --porcelain > "$OUT/source_status.txt"

declare -a pids=()
declare -a names=()
declare -a seeds=()
for spec in $SPECS; do
  seed=${spec%%:*}
  gpu=${spec##*:}
  seed_tag=$(printf '%04d' "$seed")
  name="${NAME_PREFIX}_s${seed_tag}_${RUN_TAG}"
  init="$INIT_ROOT/init_s${seed_tag}.pt"
  evidence="$OUT/evidence/$name"
  test -f "$init"
  if [ -d "$CKPT_ROOT/$name" ] || [ -e "$OUT/train_${name}.log" ]; then
    echo "refusing to overwrite prior diagnostic $name" >&2
    exit 2
  fi
  mkdir -p "$evidence"
  # Exactly one stop variable is exported, because the guard rejects both at
  # once: they demand opposite exit codes at the horizon.
  if [ "$MODE" = stability ]; then
    stop_env=(SWM_STABILITY_STOP_AFTER_STEP="$STABILITY_STOP_AFTER_STEP")
  else
    stop_env=(SWM_DIAGNOSTIC_STOP_AFTER_STEP="$DIAGNOSTIC_STOP_AFTER_STEP")
  fi
  echo "START $name gpu=$gpu mode=$MODE $(date --iso-8601=seconds)"
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE \
      -u MASTER_ADDR -u MASTER_PORT -u GROUP_RANK -u ROLE_RANK \
      -u TORCHELASTIC_RUN_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED="$seed" \
      SWM_NONFINITE_EVIDENCE_DIR="$evidence" \
      SWM_CAPTURE_NONFINITE_REPLAY=1 \
      "${stop_env[@]}" \
    "$PY" scripts/train/lewm.py \
      --config-name "$CONFIG_NAME" \
      output_model_name="$name" subdir="$name" seed="$seed" \
      init_weights_path="$init" trainer.max_epochs="$EPOCHS" \
      trainer.devices=1 data.dataset.name="$DS" \
      loader.num_workers=6 loader.prefetch_factor=2 \
      gpu_image_preprocess=true \
      "${train_limit_args[@]}" \
      > "$OUT/train_${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  seeds+=("$seed")
done

declare -a rcs=()
set +e
for index in "${!pids[@]}"; do
  wait "${pids[$index]}"
  rcs+=("$?")
done
set -e

failed=0
for index in "${!names[@]}"; do
  name=${names[$index]}
  rc=${rcs[$index]}
  seed_of_name=${seeds[$index]}
  evidence_dir="$OUT/evidence/$name"
  mapfile -t bundles < <(find "$evidence_dir" -maxdepth 1 -type f -name 'nonfinite_e*_s*.pt' | sort)
  if [ "$PREFLIGHT_MODE" = 1 ]; then
    checkpoint="$CKPT_ROOT/$name/weights_epoch_${EPOCHS}.pt"
    if [ "$rc" -ne 0 ] || [ ! -f "$checkpoint" ]; then
      echo "PREFLIGHT FAILED $name rc=$rc checkpoint=$checkpoint" >&2
      failed=1
    else
      echo "PREFLIGHT PASSED $name checkpoint=$checkpoint"
    fi
  elif [ "$MODE" = stability ]; then
    # Four independent conditions, all required. rc=0 alone is not evidence:
    # DLC job dlcbgswsqfzbqkrz reported Succeeded having run nothing.
    stop_line=$(grep -m1 '^\[grad-guard\] stability stop: ' \
      "$OUT/train_${name}.log" || true)
    reached_step=$(printf '%s' "$stop_line" \
      | sed -n 's/.*at global_step \([0-9]\+\) .*/\1/p')
    pass_file="$OUT/PASS_${name}.txt"
    if [ "$rc" -ne 0 ]; then
      echo "STABILITY FAILED $name rc=$rc (expected clean exit)" >&2
      failed=1
    elif [ "${#bundles[@]}" -ne 0 ]; then
      echo "STABILITY FAILED $name emitted ${#bundles[@]} non-finite bundle(s)" >&2
      failed=1
    elif grep -q '^\[grad-guard\] offending ' "$OUT/train_${name}.log"; then
      echo "STABILITY FAILED $name recorded an offending gradient" >&2
      failed=1
    elif [ -z "$reached_step" ]; then
      echo "STABILITY FAILED $name never printed the stability-stop marker" >&2
      failed=1
    elif [ "$reached_step" -le 137496 ]; then
      echo "STABILITY FAILED $name stopped at $reached_step, not past 137496" >&2
      failed=1
    else
      {
        echo "result=PASS"
        echo "mode=stability"
        echo "run=$name"
        echo "seed=$seed_of_name"
        echo "gpu_visible=1"
        echo "encoder_fp32=true"
        echo "config=$CONFIG_NAME"
        echo "max_epochs=$EPOCHS"
        echo "reached_global_step=$reached_step"
        echo "stop_horizon=$STABILITY_STOP_AFTER_STEP"
        echo "historical_failure_steps=115683,137496"
        echo "nonfinite_events=0"
        echo "nonfinite_bundles=0"
        echo "return_code=$rc"
        echo "commit=$current_commit"
        echo "recorded_at=$(date --iso-8601=seconds)"
      } > "$pass_file"
      echo "STABILITY PASSED $name reached_step=$reached_step evidence=$pass_file"
      grep -m1 '^\[grad-guard\] stability stop: ' "$OUT/train_${name}.log"
    fi
  elif [ "$rc" -eq 0 ]; then
    echo "UNEXPECTED $name completed without reproducing first Inf" >&2
    failed=1
  elif [ "${#bundles[@]}" -ne 1 ]; then
    echo "INVALID $name rc=$rc evidence_bundles=${#bundles[@]}" >&2
    failed=1
  elif ! grep -q '^\[grad-guard\] offending ' "$OUT/train_${name}.log"; then
    echo "INVALID $name has bundle but no offending-gradient record" >&2
    failed=1
  else
    echo "REPRODUCED $name rc=$rc bundle=${bundles[0]}"
    grep -E '^\[grad-guard\] (evidence|offending|wrote evidence)' \
      "$OUT/train_${name}.log"
  fi
done

test "$failed" -eq 0
if [ "$PREFLIGHT_MODE" = 1 ]; then
  echo "NONFINITE ROOT-CAUSE ENTRYPOINT PREFLIGHT COMPLETE"
elif [ "$MODE" = stability ]; then
  # The aggregate gate is written only after every per-seed PASS file exists,
  # so its presence means both seeds passed -- never one. Downstream consumers
  # should test for this file, not parse the log.
  for name in "${names[@]}"; do
    test -f "$OUT/PASS_${name}.txt"
  done
  test "${#names[@]}" -eq 2
  gate="$OUT/STABILITY_GATE_PASS.txt"
  {
    echo "result=PASS"
    echo "gate=encoder_fp32_two_seed_stability"
    echo "seeds=${seeds[*]}"
    echo "runs=${names[*]}"
    echo "gpus=2"
    echo "encoder_fp32=true"
    echo "config=$CONFIG_NAME"
    echo "max_epochs=$EPOCHS"
    echo "stop_horizon=$STABILITY_STOP_AFTER_STEP"
    echo "crossed_historical_failure_steps=115683,137496"
    echo "nonfinite_grad_policy=error"
    echo "commit=$current_commit"
    echo "recorded_at=$(date --iso-8601=seconds)"
    echo "note=diagnostic validation only; does not auto-launch the formal wave. This file is the evidence the formal launcher requires via LEWM_STABILITY_GATE, and it re-validates every field and the commit before training."
  } > "$gate"
  echo "ENCODER FP32 TWO-SEED STABILITY VALIDATION COMPLETE gate=$gate"
else
  echo "NONFINITE ROOT-CAUSE REPRODUCTION EVIDENCE COMPLETE"
fi
