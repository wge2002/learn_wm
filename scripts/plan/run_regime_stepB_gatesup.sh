#!/usr/bin/env bash
# Step B round A: weak gate-supervision sweep. Does a LEARNABLE gate, given a
# small contact prior (Step A read-out), close the drift gap toward the oracle
# upper bound (mse@10 0.320) vs the contact-blind learned gate (0.403)?
# Usage: bash scripts/plan/run_regime_stepB_gatesup.sh <train_npz> <eval_npz> <out_root> [epochs]
set -euo pipefail

DATA=${1:?train npz}
EVAL=${2:?eval contact npz}
ROOT=${3:?out root}
EPOCHS=${4:-60}
PY=.venv/bin/python
NGPU=${NGPU:-8}

mkdir -p "$ROOT"
# gate-sup weight : gate-input
CONFIGS=(
  "0.1:state"
  "0.3:state"
  "1.0:state"
  "3.0:state"
  "1.0:both"
)
SEEDS=(0 1 2)

i=0
for cfg in "${CONFIGS[@]}"; do
  IFS=: read -r gs gin <<< "$cfg"
  for s in "${SEEDS[@]}"; do
    gpu=$(( i % NGPU ))
    tag="gs${gs}_${gin}_s${s}"
    echo "[launch] $tag -> GPU $gpu"
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    CUDA_VISIBLE_DEVICES=$gpu SWM_TORCH_THREADS=3 OMP_NUM_THREADS=3 \
      $PY scripts/plan/regime_moe_stepB.py \
      --data "$DATA" --eval-contact "$EVAL" \
      --arm moe --experts 2 --gate-input "$gin" --gate-sup "$gs" \
      --epochs "$EPOCHS" --seed "$s" --output-dir "$ROOT/$tag" \
      > "$ROOT/$tag.log" 2>&1 &
    i=$(( i + 1 ))
    if (( i % NGPU == 0 )); then wait; fi
  done
done
wait
echo "[done] gate-sup sweep -> $ROOT"
