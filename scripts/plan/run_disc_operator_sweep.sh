#!/usr/bin/env bash
# Fan out the discrete-operator vs continuous sweep across GPUs 1-7 (GPU0 busy).
# Each cell is a tiny predictor; GPUs run their queue serially.
# Usage: run_disc_operator_sweep.sh <data.npz> <out_root> [epochs]
set -u
DATA="$1"; OUT="$2"; EPOCHS="${3:-60}"
PY=.venv/bin/python
GPUS=(1 2 3 4 5 6 7)
mkdir -p "$OUT"
LOG="$OUT/orchestrator.log"
echo "=== sweep start $(date) data=$DATA epochs=$EPOCHS ===" | tee "$LOG"

# build cell list: "arm K U seed"
CELLS=()
for s in 0 1 2; do
  for u in 1 5; do
    CELLS+=("cont 0 $u $s")
    for k in 4 8 16 32; do
      CELLS+=("disc $k $u $s")
      CELLS+=("disc_c $k $u $s")
    done
  done
done
echo "total cells: ${#CELLS[@]}" | tee -a "$LOG"

# assign cells round-robin to GPUs; each GPU's queue runs in its own subshell
declare -A QUEUE
i=0
for c in "${CELLS[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  QUEUE[$g]="${QUEUE[$g]:-}|$c"
  i=$((i+1))
done

for g in "${GPUS[@]}"; do
  (
    IFS='|' read -ra mine <<< "${QUEUE[$g]}"
    for c in "${mine[@]}"; do
      [ -z "$c" ] && continue
      read -r arm K U seed <<< "$c"
      tag="${arm}_K${K}_U${U}_s${seed}"
      odir="$OUT/$tag"
      mkdir -p "$odir"
      CUDA_VISIBLE_DEVICES=$g SWM_TORCH_THREADS=2 OMP_NUM_THREADS=2 \
        $PY scripts/plan/disc_operator_train.py \
          --data "$DATA" --output-dir "$odir" \
          --arm "$arm" --codebook "$K" --unroll "$U" --seed "$seed" \
          --epochs "$EPOCHS" --device cuda \
          > "$odir/run.log" 2>&1
      echo "[gpu$g] done $tag rc=$? $(date +%H:%M:%S)" >> "$LOG"
    done
  ) &
done
wait
echo "=== sweep done $(date) ===" | tee -a "$LOG"
