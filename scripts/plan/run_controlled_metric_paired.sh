#!/usr/bin/env bash
# Formal matched-training protocol for the controlled-metric K1/K5 test.
#
# Examples:
#   PHASES=init,train NGPU=6 bash scripts/plan/run_controlled_metric_paired.sh
#   PHASES=audit,summarize NGPU=2 bash scripts/plan/run_controlled_metric_paired.sh
#
# Every training seed gets one immutable initialization state_dict. Both arms
# load that exact file, consume eight-frame windows with the same DataLoader
# seed, and execute the same number of optimizer updates. The training logs
# include initialization, split, loader-state, and first-batch fingerprints.
set -Eeuo pipefail

DEFAULT_PY=python
if [ -x "$HOME/venv-clean/bin/python" ]; then
  DEFAULT_PY="$HOME/venv-clean/bin/python"
fi
PY=${PY:-$DEFAULT_PY}

DEFAULT_DS="$HOME/.stable_worldmodel/pusht_expert_train.h5"
if [ -f "$HOME/data/learn_wm/pusht_expert_train.h5" ]; then
  DEFAULT_DS="$HOME/data/learn_wm/pusht_expert_train.h5"
fi
DS=${DS:-$DEFAULT_DS}
PIXELS=${PIXELS:-${DS%.h5}_pixels.npy}
RUN_TAG=${RUN_TAG:-controlled_metric_paired_v3_20260813}
OUT=${OUT:-outputs/$RUN_TAG}
SEEDS=${SEEDS:-"7 13 42"}
EPOCHS=${EPOCHS:-30}
AUDIT_EPOCHS=${AUDIT_EPOCHS:-"5 10 20 30"}
AUDIT_BANK_SEED=${AUDIT_BANK_SEED:-20260810}
AUDIT_SAMPLES=${AUDIT_SAMPLES:-1024}
JACOBIAN_SAMPLES=${JACOBIAN_SAMPLES:-64}
BOOTSTRAPS=${BOOTSTRAPS:-20000}
NGPU=${NGPU:-6}
GPU_IDS=${GPU_IDS:-}
WORKERS=${WORKERS:-2}
PREFETCH=${PREFETCH:-1}
PHASES=${PHASES:-init,train,audit,summarize}
GPU_IMAGE_PREPROCESS=${GPU_IMAGE_PREPROCESS:-true}
USE_PIXEL_SIDECAR=${USE_PIXEL_SIDECAR:-false}
DEFAULT_STABLEWM_HOME="$HOME/.stable_worldmodel"
if [ -d "$HOME/swmhome/learn_wm/checkpoints" ]; then
  DEFAULT_STABLEWM_HOME="$HOME/swmhome/learn_wm"
elif [ -d "$HOME/swmhome/checkpoints" ]; then
  DEFAULT_STABLEWM_HOME="$HOME/swmhome"
elif [ -d "$HOME/stablewm/checkpoints" ]; then
  DEFAULT_STABLEWM_HOME="$HOME/stablewm"
fi
STABLEWM_ROOT=${STABLEWM_HOME:-$DEFAULT_STABLEWM_HOME}
export STABLEWM_HOME="$STABLEWM_ROOT"
export LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR:-${DS%/*}}
CKPT_ROOT=$STABLEWM_ROOT/checkpoints
INIT_ROOT=${INIT_ROOT:-$CKPT_ROOT/paired_initializations/$RUN_TAG}

# Required by deterministic CUDA GEMM on current PyTorch/CUDA builds.
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}

mkdir -p "$OUT" "$INIT_ROOT"
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
  for ((gpu_index=0; gpu_index<NGPU; gpu_index++)); do
    GPUS+=("$gpu_index")
  done
fi

if [ "$USE_PIXEL_SIDECAR" = true ]; then
  test -f "$PIXELS"
fi

has_phase() {
  case ",$PHASES," in
    *,"$1",*) return 0 ;;
    *) return 1 ;;
  esac
}

# --- Evidence gate for any phase that trains -------------------------------
# There is no boolean bypass. Training requires LEWM_STABILITY_GATE to point at
# the STABILITY_GATE_PASS.txt that MODE=stability actually produced, and every
# field below must match the protocol this repository is currently at. Audit and
# summarize-only phases run without the gate, because they only read
# checkpoints that already exist.

gate_field() { # KEY
  local key=$1
  local hits
  hits=$(grep -c "^$key=" "$LEWM_STABILITY_GATE" || true)
  if [ "$hits" -ne 1 ]; then
    echo "stability gate: expected exactly one '$key=' line, found $hits" >&2
    echo "gate file: $LEWM_STABILITY_GATE" >&2
    exit 2
  fi
  sed -n "s/^$key=//p" "$LEWM_STABILITY_GATE"
}

gate_expect() { # KEY EXPECTED
  local actual
  actual=$(gate_field "$1")
  if [ "$actual" != "$2" ]; then
    echo "stability gate: $1='$actual', expected '$2'" >&2
    echo "gate file: $LEWM_STABILITY_GATE" >&2
    exit 2
  fi
}

require_stability_gate() {
  if [ "$EPOCHS" != "30" ]; then
    echo "formal training blocked: EPOCHS=$EPOCHS, expected 30" >&2
    exit 2
  fi
  if [ "$SEEDS" != "7 13 42" ]; then
    echo "formal training blocked: SEEDS='$SEEDS', expected '7 13 42'" >&2
    exit 2
  fi
  if [ "$NGPU" != "6" ]; then
    echo "formal training blocked: NGPU=$NGPU, expected 6" >&2
    exit 2
  fi
  if ! git -c "safe.directory=$PWD" diff --quiet \
    || ! git -c "safe.directory=$PWD" diff --cached --quiet; then
    echo "formal training blocked: tracked repository changes detected" >&2
    echo "commit the exact code validated by the stability gate first" >&2
    exit 2
  fi

  if [ -z "${LEWM_STABILITY_GATE:-}" ]; then
    echo "formal training blocked: LEWM_STABILITY_GATE is not set" >&2
    echo "point it at the STABILITY_GATE_PASS.txt written by" >&2
    echo "  MODE=stability bash scripts/plan/run_lewm_nonfinite_rootcause_dlc.sh" >&2
    echo "see docs/knowledge/controlled_metric_k1_failure_diagnosis_20260813.md" >&2
    exit 2
  fi
  if [ ! -f "$LEWM_STABILITY_GATE" ]; then
    echo "formal training blocked: no such gate file: $LEWM_STABILITY_GATE" >&2
    exit 2
  fi

  gate_expect result PASS
  gate_expect gate encoder_fp32_two_seed_stability
  # Launcher order, not a sorted set: SPECS="13:... 42:..." writes "13 42".
  gate_expect seeds "13 42"
  gate_expect encoder_fp32 true
  gate_expect max_epochs 30
  gate_expect nonfinite_grad_policy error

  local horizon
  horizon=$(gate_field stop_horizon)
  case "$horizon" in
    ''|*[!0-9]*)
      echo "stability gate: stop_horizon='$horizon' is not a number" >&2
      exit 2
      ;;
  esac
  if [ "$horizon" -le 137496 ]; then
    echo "stability gate: stop_horizon=$horizon did not clear 137496" >&2
    exit 2
  fi

  local gate_commit
  local head_commit
  gate_commit=$(gate_field commit)
  head_commit=$(git -c "safe.directory=$PWD" rev-parse HEAD)
  if [ "$gate_commit" != "$head_commit" ]; then
    echo "stability gate: validated commit $gate_commit != HEAD $head_commit" >&2
    echo "re-run MODE=stability on this commit before formal training" >&2
    exit 2
  fi

  echo "STABILITY GATE OK $LEWM_STABILITY_GATE commit=$head_commit horizon=$horizon"
}

if has_phase train; then
  require_stability_gate
fi

seed_tag() {
  printf '%04d' "$1"
}

run_name() {
  local arm=$1
  local seed=$2
  printf 'cm_%s_s%s_%s' "$arm" "$(seed_tag "$seed")" "$RUN_TAG"
}

init_path() {
  printf '%s/init_s%s.pt' "$INIT_ROOT" "$(seed_tag "$1")"
}

declare -a PIDS=()
declare -a JOBS=()
declare -a ARTIFACTS=()
JOB_INDEX=0

wait_batch() {
  local failed=0
  local job_index
  for job_index in "${!PIDS[@]}"; do
    if wait "${PIDS[$job_index]}" \
      && { [ -z "${ARTIFACTS[$job_index]}" ] \
        || [ -f "${ARTIFACTS[$job_index]}" ]; }; then
      echo "DONE ${JOBS[$job_index]} $(date --iso-8601=seconds)"
    else
      echo "FAILED ${JOBS[$job_index]} $(date --iso-8601=seconds)" >&2
      failed=1
    fi
  done
  PIDS=()
  JOBS=()
  ARTIFACTS=()
  return "$failed"
}

enqueue() { # JOB_NAME ARTIFACT COMMAND...
  local job=$1
  local artifact=$2
  shift 2
  local gpu=${GPUS[$((JOB_INDEX % NGPU))]}
  echo "START $job gpu=$gpu $(date --iso-8601=seconds)"
  env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE \
      -u MASTER_ADDR -u MASTER_PORT -u GROUP_RANK -u ROLE_RANK \
      -u TORCHELASTIC_RUN_ID \
      CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$OUT/$job.log" 2>&1 &
  PIDS+=("$!")
  JOBS+=("$job")
  ARTIFACTS+=("$artifact")
  JOB_INDEX=$((JOB_INDEX + 1))
  if [ "${#PIDS[@]}" -eq "$NGPU" ]; then
    wait_batch
  fi
}

pixel_args=()
if [ "$USE_PIXEL_SIDECAR" = true ]; then
  pixel_args=("+data.dataset.pixels_path=$PIXELS")
fi

if has_phase init; then
  for seed in $SEEDS; do
    artifact=$(init_path "$seed")
    env -u RANK -u LOCAL_RANK -u WORLD_SIZE -u LOCAL_WORLD_SIZE \
      -u MASTER_ADDR -u MASTER_PORT -u GROUP_RANK -u ROLE_RANK \
      -u TORCHELASTIC_RUN_ID \
      PYTHONHASHSEED="$seed" "$PY" scripts/train/lewm.py \
      --config-name lewm_paired_k1 \
      seed="$seed" init_only=true \
      export_init_weights_path="$artifact" \
      data.dataset.name="$DS" \
      loader.num_workers="$WORKERS" loader.prefetch_factor="$PREFETCH" \
      gpu_image_preprocess="$GPU_IMAGE_PREPROCESS" \
      "${pixel_args[@]}" \
      > "$OUT/init_s$(seed_tag "$seed").log" 2>&1
    test -f "$artifact"
  done
fi

if has_phase train; then
  for seed in $SEEDS; do
    artifact=$(init_path "$seed")
    test -f "$artifact"
    for arm in k1 k5; do
      name=$(run_name "$arm" "$seed")
      checkpoint="$CKPT_ROOT/$name/weights_epoch_${EPOCHS}.pt"
      if [ -f "$checkpoint" ]; then
        echo "SKIP train_$name: final checkpoint exists"
        continue
      fi
      if [ -d "$CKPT_ROOT/$name" ]; then
        echo "REFUSE train_$name: partial run directory exists" >&2
        echo "restart both arms under a new RUN_TAG; do not splice pairs" >&2
        exit 2
      fi
      config="lewm_paired_$arm"
      enqueue "train_$name" "$checkpoint" \
        env PYTHONHASHSEED="$seed" \
        SWM_NONFINITE_EVIDENCE_DIR="$OUT/nonfinite_evidence/$name" \
        "$PY" scripts/train/lewm.py \
        --config-name "$config" \
        output_model_name="$name" subdir="$name" seed="$seed" \
        init_weights_path="$artifact" trainer.max_epochs="$EPOCHS" \
        trainer.devices=1 data.dataset.name="$DS" \
        loader.num_workers="$WORKERS" loader.prefetch_factor="$PREFETCH" \
        gpu_image_preprocess="$GPU_IMAGE_PREPROCESS" \
        "${pixel_args[@]}"
    done
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
  fi
fi

if has_phase audit; then
  for epoch in $AUDIT_EPOCHS; do
    test "$epoch" -le "$EPOCHS"
    for seed in $SEEDS; do
      seed_id=$(seed_tag "$seed")
      k1="$CKPT_ROOT/$(run_name k1 "$seed")/weights_epoch_${epoch}.pt"
      k5="$CKPT_ROOT/$(run_name k5 "$seed")/weights_epoch_${epoch}.pt"
      result="$OUT/audit_s${seed_id}_e${epoch}.json"
      test -f "$k1"
      test -f "$k5"
      if [ -f "$result" ]; then
        echo "SKIP audit_s${seed_id}_e${epoch}: result exists"
        continue
      fi
      enqueue "audit_s${seed_id}_e${epoch}" "$result" \
        "$PY" scripts/plan/controlled_metric_audit.py \
        --pair-id "seed_$seed_id" --training-seed "$seed" \
        --checkpoint-epoch "$epoch" \
        --policy "K1=$k1" --policy "K5=$k5" --reference K1 \
        --dataset "$DS" --output "$result" \
        --num-samples "$AUDIT_SAMPLES" \
        --jacobian-samples "$JACOBIAN_SAMPLES" \
        --seed "$AUDIT_BANK_SEED" --device cuda
    done
  done
  if [ "${#PIDS[@]}" -gt 0 ]; then
    wait_batch
  fi
fi

if has_phase summarize; then
  shopt -s nullglob
  audits=("$OUT"/audit_s*_e*.json)
  [ "${#audits[@]}" -gt 0 ] || {
    echo "no audit JSON files found in $OUT" >&2
    exit 2
  }
  proof_args=(
    --log-dir "$OUT"
    --checkpoint-root "$CKPT_ROOT"
    --run-tag "$RUN_TAG"
    --epochs "$EPOCHS"
    --output "$OUT/pairing_proof.json"
    --seeds
  )
  for seed in $SEEDS; do
    proof_args+=("$seed")
  done
  "$PY" scripts/plan/verify_controlled_metric_pairing.py \
    "${proof_args[@]}"
  "$PY" scripts/plan/summarize_controlled_metric_pairs.py \
    --input "${audits[@]}" \
    --protocol-proof "$OUT/pairing_proof.json" \
    --decision-epoch "$EPOCHS" --min-pairs 3 \
    --bootstraps "$BOOTSTRAPS" \
    --output-json "$OUT/paired_summary.json" \
    --output-md "$OUT/paired_summary.md"
fi

echo "CONTROLLED-METRIC PAIRED PROTOCOL DONE $(date --iso-8601=seconds)"
