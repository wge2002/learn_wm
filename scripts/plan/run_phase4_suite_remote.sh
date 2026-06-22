#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/code/wge/stable-worldmodel}"
PYTHON="${PYTHON:-/root/miniconda/envs/lerobot/bin/python}"
GPU="${GPU:-6}"
THREADS="${THREADS:-2}"

cd "$REPO"

export CUDA_VISIBLE_DEVICES="$GPU"
export SWM_TORCH_THREADS="$THREADS"
export OMP_NUM_THREADS="$THREADS"

echo "[phase4-suite] repo=$REPO python=$PYTHON gpu=$GPU threads=$THREADS"

echo "[phase4-suite] contact/event analysis"
"$PYTHON" scripts/plan/geometry_contact_phase4.py \
  --num-samples 200 \
  --max-env-steps 50 \
  --goal-offset 50 \
  --action-block 5 \
  --seed 42 \
  --output-dir outputs/lghl_phase4_contact_n200_goal50

echo "[phase4-suite] action-quality analysis"
"$PYTHON" scripts/plan/latent_action_quality_phase4.py \
  --policy quentinll/lewm-pusht \
  --dataset-name pusht_expert_train.h5 \
  --num-samples 100 \
  --max-k 10 \
  --eval-ks 1,3,5,10 \
  --plan-horizon 5 \
  --goal-offset 50 \
  --action-block 5 \
  --num-candidates 256 \
  --batch-size 16 \
  --seed 42 \
  --device cuda \
  --output-dir outputs/lghl_phase4_action_quality_n100_c256_goal50

echo "[phase4-suite] warm_start ablation"
"$PYTHON" scripts/plan/lghl_warm_start_ablation.py \
  --repo "$REPO" \
  --policy quentinll/lewm-pusht \
  --num-eval 50 \
  --goal-offset 25 \
  --eval-budget 50 \
  --horizon 5 \
  --receding 1,3,5 \
  --warm-start true,false \
  --action-block 5 \
  --seed 42 \
  --gpus "$GPU" \
  --max-workers 1 \
  --threads-per-run "$THREADS" \
  --output-dir outputs/lghl_phase4_warm_start_n50

echo "[phase4-suite] done"
