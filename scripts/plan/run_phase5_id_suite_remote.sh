#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-outputs/lghl_phase5_id_diagnostics}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
SWM_TORCH_THREADS="${SWM_TORCH_THREADS:-2}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SWM_TORCH_THREADS}}"

export CUDA_VISIBLE_DEVICES
export SWM_TORCH_THREADS
export OMP_NUM_THREADS

mkdir -p "${OUT_ROOT}"

echo "[phase5] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[phase5] SWM_TORCH_THREADS=${SWM_TORCH_THREADS}"
echo "[phase5] OUT_ROOT=${OUT_ROOT}"

echo "[phase5] behavior: horizon/goal-offset diagonal alignment"
"${PYTHON_BIN}" scripts/plan/lghl_phase5_id_behavior.py \
  --output-dir "${OUT_ROOT}/behavior_alignment" \
  --goal-offsets 5,10,15,25,50 \
  --horizons 1,2,3,5,8,10 \
  --receding-mode diagonal \
  --warm-start true \
  --num-eval 50 \
  --eval-budget 50 \
  --seed 42 \
  --device cuda

echo "[phase5] behavior: H=5 warm/cold receding comparison"
"${PYTHON_BIN}" scripts/plan/lghl_phase5_id_behavior.py \
  --output-dir "${OUT_ROOT}/behavior_h5_warmcold" \
  --goal-offsets 25 \
  --horizons 5 \
  --receding 1,2,3,5 \
  --receding-mode list \
  --warm-start true,false \
  --num-eval 50 \
  --eval-budget 50 \
  --seed 42 \
  --device cuda

echo "[phase5] action-quality: ID frontier by plan_horizon"
for plan_horizon in 1 2 3 5; do
  "${PYTHON_BIN}" scripts/plan/latent_action_quality_phase4.py \
    --output-dir "${OUT_ROOT}/action_quality_plan_h${plan_horizon}" \
    --shift id \
    --eval-ks 0,1,2,3,4,5,6,7,8,9,10 \
    --plan-horizon "${plan_horizon}" \
    --num-samples 100 \
    --num-candidates 256 \
    --goal-offset 50 \
    --seed 42 \
    --device cuda
done

echo "[phase5] done"
