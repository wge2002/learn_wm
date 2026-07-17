#!/usr/bin/env bash
# Gate A: horizon-matching matrix (zero training, existing checkpoints).
# K_train x H_plan x goal_offset, fixed-candidates protocol (s=300).
# Fixed-model-calls protocol: rerun the H_plan rows with
#   solver.num_samples = round(300 * 5 / H)  (matches candidates x horizon).
# Usage: NGPU=4 bash run_gateA_matrix.sh   (set PY/DS for the A100 box)
set -Eeuo pipefail
PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
OUT=${OUT:-outputs/week1}
mkdir -p "$OUT"
NGPU=${NGPU:-4}
MODELS=${MODELS:-"iter2_baseline pd_d192_k2 pd_d192_k3 iter2_multistep pd_d192_k10"}
POLICY_SUFFIX=${POLICY_SUFFIX:-_eval}
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}

declare -a PIDS=()
declare -a JOBS=()
I=0

wait_batch() {
  local failed=0
  local i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "DONE ${JOBS[$i]} $(date --iso-8601=seconds)"
    else
      echo "FAILED ${JOBS[$i]} $(date --iso-8601=seconds)"
      failed=1
    fi
  done
  PIDS=()
  JOBS=()
  return "$failed"
}

run() { # MODEL H OFF SAMPLES TAG
  local model=$1
  local horizon=$2
  local offset=$3
  local samples=$4
  local tag=$5
  local gpu=$((I % NGPU))
  local receding=$horizon
  local job="gateA_${model}_h${horizon}_off${offset}_${tag}"

  # WorldModelPolicy cannot execute more model steps than were planned.
  # Keep the historical five-step MPC commitment for H>=5 and shorten it
  # for H=1/3.
  if [ "$receding" -gt 5 ]; then
    receding=5
  fi

  echo "START $job gpu=$gpu samples=$samples $(date --iso-8601=seconds)"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/plan/eval_wm.py \
    policy="${model}${POLICY_SUFFIX}" seed=42 +plan_config.history_len=3 \
    plan_config.horizon="$horizon" \
    plan_config.receding_horizon="$receding" \
    eval.goal_offset_steps="$offset" solver.num_samples="$samples" \
    eval.num_eval=50 eval.video=false eval.dataset_name=$DS \
    output.filename="${job}.txt" > "$OUT/${job}.log" 2>&1 &
  PIDS+=("$!")
  JOBS+=("$job")
  I=$((I + 1))
  if [ "${#PIDS[@]}" -eq "$NGPU" ]; then
    wait_batch
  fi
}

for M in $MODELS; do
  for H in 1 3 5 8 10; do
    for OFF in 25 40 60; do
      run "$M" "$H" "$OFF" 300 fixcand
    done
  done
done
if [ "${#PIDS[@]}" -gt 0 ]; then
  wait_batch
fi

# fixed-model-calls protocol (calls ~ samples*H; anchor at H=5,s=300)
for M in $MODELS; do
  for H in 1 3 5 8 10; do
    S=$((1500 / H))
    for OFF in 25 40 60; do
      run "$M" "$H" "$OFF" "$S" fixcalls
    done
  done
done
if [ "${#PIDS[@]}" -gt 0 ]; then
  wait_batch
fi
echo "GATE A MATRIX DONE $(date --iso-8601=seconds)"
