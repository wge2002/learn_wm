#!/usr/bin/env bash
# Step B sweep: mono (param ladder) vs MoE (regime gate on state), 3 seeds.
# Distributes independent runs across GPUs 0-7. Usage:
#   bash scripts/plan/run_regime_stepB.sh <train_npz> <eval_contact_npz> <out_root> [epochs]
set -euo pipefail

DATA=${1:?train npz}
EVAL=${2:?eval contact npz}
ROOT=${3:?out root}
EPOCHS=${4:-60}
PY=.venv/bin/python
NGPU=${NGPU:-8}

mkdir -p "$ROOT"
# arm:experts:hidden
CONFIGS=(
  "mono:1:512"
  "mono:1:1024"
  "moe:2:512"
  "moe:3:512"
  "moe:4:512"
)
SEEDS=(0 1 2)

i=0
for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r arm K hidden <<< "$cfg"
  for s in "${SEEDS[@]}"; do
    gpu=$(( i % NGPU ))
    tag="${arm}_K${K}_h${hidden}_s${s}"
    echo "[launch] $tag -> GPU $gpu"
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    CUDA_VISIBLE_DEVICES=$gpu SWM_TORCH_THREADS=3 OMP_NUM_THREADS=3 \
      $PY scripts/plan/regime_moe_stepB.py \
      --data "$DATA" --eval-contact "$EVAL" \
      --arm "$arm" --experts "$K" --hidden "$hidden" --epochs "$EPOCHS" \
      --seed "$s" --output-dir "$ROOT/$tag" \
      > "$ROOT/$tag.log" 2>&1 &
    i=$(( i + 1 ))
    # keep at most NGPU jobs in flight
    if (( i % NGPU == 0 )); then wait; fi
  done
done
wait
echo "[done] all Step B runs finished -> $ROOT"
