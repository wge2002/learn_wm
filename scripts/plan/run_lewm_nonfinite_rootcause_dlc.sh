#!/usr/bin/env bash
# Two-GPU expected-failure reproduction of the v2 K1 first-Inf events.
# The DLC task succeeds only if both training processes fail through the strict
# guard and each writes a replay evidence bundle.
set -Eeuo pipefail

TARGET_UID=10011
TARGET_GID=10011
TARGET_HOME=/mnt/home/gewang

if [ "$(id -u)" -eq 0 ]; then
  exec /usr/bin/setpriv \
    --reuid="$TARGET_UID" --regid="$TARGET_GID" --clear-groups \
    /usr/bin/env HOME="$TARGET_HOME" bash "$0" "$@"
fi
if [ "$(id -u)" -ne "$TARGET_UID" ] || [ "$(id -g)" -ne "$TARGET_GID" ]; then
  echo "expected runtime identity $TARGET_UID:$TARGET_GID, found $(id -u):$(id -g)" >&2
  exit 2
fi
export HOME="$TARGET_HOME"
umask 002

REPO=/mnt/home/gewang/code/learn_wm
PY=/mnt/home/gewang/venv-clean/bin/python
DS=/mnt/home/gewang/data/learn_wm/pusht_expert_train.h5
STABLEWM_ROOT=/mnt/home/gewang/swmhome/learn_wm
CKPT_ROOT=$STABLEWM_ROOT/checkpoints
INIT_ROOT=$CKPT_ROOT/paired_initializations/controlled_metric_paired_20260810
RUN_TAG=${RUN_TAG:-nonfinite_rootcause_v2k1_20260813_r1}
OUT=${OUT:-$REPO/outputs/$RUN_TAG}
EPOCHS=${EPOCHS:-13}
export STABLEWM_HOME="$STABLEWM_ROOT"
export LOCAL_DATASET_DIR=${DS%/*}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export SWM_TORCH_THREADS=2
export HYDRA_FULL_ERROR=1

cd "$REPO"
test -x "$PY"
test -f "$DS"
test "$EPOCHS" -ge 13
test "$(nvidia-smi -L | wc -l)" -eq 2

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
for spec in 13:0 42:1; do
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
  if [ "$rc" -eq 0 ]; then
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
echo "NONFINITE ROOT-CAUSE REPRODUCTION EVIDENCE COMPLETE"
