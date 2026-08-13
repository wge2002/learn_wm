#!/usr/bin/env bash
# Two-GPU expected-failure reproduction of the v2 K1 first-Inf events.
# The DLC task succeeds only if both training processes fail through the strict
# guard and each writes a replay evidence bundle.
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
RUN_TAG=${RUN_TAG:-nonfinite_rootcause_v2k1_20260813_r1}
OUT=${OUT:-$REPO/outputs/$RUN_TAG}
EPOCHS=${EPOCHS:-13}
PREFLIGHT_MODE=${RBS_ROOTCAUSE_PREFLIGHT:-0}
ROOTCAUSE_GPU_IDS=${ROOTCAUSE_GPU_IDS:-"0 1"}
read -r GPU_SEED13 GPU_SEED42 extra_gpu <<< "$ROOTCAUSE_GPU_IDS"
if [ -z "${GPU_SEED13:-}" ] || [ -z "${GPU_SEED42:-}" ] \
  || [ -n "${extra_gpu:-}" ] || [ "$GPU_SEED13" = "$GPU_SEED42" ]; then
  echo "ROOTCAUSE_GPU_IDS must contain two distinct GPU indices" >&2
  exit 2
fi
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
  test "$EPOCHS" -ge 13
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
for spec in $SPECS; do
  seed=${spec%%:*}
  gpu=${spec##*:}
  seed_tag=$(printf '%04d' "$seed")
  name="nfdiag_v2k1_s${seed_tag}_${RUN_TAG}"
  init="$INIT_ROOT/init_s${seed_tag}.pt"
  evidence="$OUT/evidence/$name"
  test -f "$init"
  if [ -d "$CKPT_ROOT/$name" ] || [ -e "$OUT/train_${name}.log" ]; then
    echo "refusing to overwrite prior diagnostic $name" >&2
    exit 2
  fi
  mkdir -p "$evidence"
  echo "START $name gpu=$gpu $(date --iso-8601=seconds)"
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE \
      -u MASTER_ADDR -u MASTER_PORT -u GROUP_RANK -u ROLE_RANK \
      -u TORCHELASTIC_RUN_ID \
      CUDA_VISIBLE_DEVICES="$gpu" \
      PYTHONHASHSEED="$seed" \
      SWM_NONFINITE_EVIDENCE_DIR="$evidence" \
      SWM_CAPTURE_NONFINITE_REPLAY=1 \
    "$PY" scripts/train/lewm.py \
      --config-name lewm_nonfinite_v2_k1_repro \
      output_model_name="$name" subdir="$name" seed="$seed" \
      init_weights_path="$init" trainer.max_epochs="$EPOCHS" \
      trainer.devices=1 data.dataset.name="$DS" \
      loader.num_workers=6 loader.prefetch_factor=2 \
      gpu_image_preprocess=true \
      "${train_limit_args[@]}" \
      > "$OUT/train_${name}.log" 2>&1 &
  pids+=("$!")
  names+=("$name")
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
else
  echo "NONFINITE ROOT-CAUSE REPRODUCTION EVIDENCE COMPLETE"
fi
