#!/usr/bin/env bash
# DLC worker entry point for the preregistered K1/K5 paired training wave.
set -Eeuo pipefail

REPO=/mnt/home/gewang/code/learn_wm
export PY=/mnt/home/gewang/venv-clean/bin/python
export DS=/mnt/home/gewang/data/learn_wm/pusht_expert_train.h5
export STABLEWM_HOME=/mnt/home/gewang/swmhome/learn_wm
export RUN_TAG=${RUN_TAG:-controlled_metric_paired_20260810}
export OUT=${OUT:-$REPO/outputs/$RUN_TAG}
export PHASES=${PHASES:-init,train}
export SEEDS=${SEEDS:-"7 13 42"}
export EPOCHS=${EPOCHS:-30}
export NGPU=${NGPU:-8}
export GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
export WORKERS=${WORKERS:-6}
export PREFETCH=${PREFETCH:-2}
export USE_PIXEL_SIDECAR=false
export GPU_IMAGE_PREPROCESS=true
export SWM_TORCH_THREADS=${SWM_TORCH_THREADS:-2}
export HYDRA_FULL_ERROR=1

cd "$REPO"
test -f "$DS"
test -x "$PY"
test "$NGPU" -eq 8
test "$GPU_IDS" = 0,1,2,3,4,5,6,7

current_commit=$(git rev-parse HEAD)
if [ -n "${EXPECTED_COMMIT:-}" ] && [ "$current_commit" != "$EXPECTED_COMMIT" ]; then
  echo "expected commit $EXPECTED_COMMIT, found $current_commit" >&2
  exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "tracked repository changes detected; refusing formal training" >&2
  exit 2
fi

mkdir -p "$OUT"
printf '%s\n' "$current_commit" > "$OUT/source_commit.txt"
git status --porcelain > "$OUT/source_status.txt"
nvidia-smi -L
"$PY" -c 'import stable_worldmodel, torch; print("swm ok |", torch.__version__)'

(
  while true; do
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
      --format=csv,noheader,nounits >> "$OUT/gpu.csv"
    sleep 30
  done
) &
sampler_pid=$!
trap 'kill "$sampler_pid" 2>/dev/null || true' EXIT

set +e
bash scripts/plan/run_controlled_metric_paired.sh \
  2>&1 | tee "$OUT/dlc_wave.log"
return_code=${PIPESTATUS[0]}
set -e

echo "=== paired wave rc=$return_code ==="
echo "=== throughput per model ==="
for log in "$OUT"/train_*.log; do
  [ -f "$log" ] || continue
  printf '  %-48s %s\n' "$(basename "$log" .log)" \
    "$(grep -oE '[0-9.]+it/s' "$log" | tail -3 | tr '\n' ' ')"
done
echo "=== GPU utilization ==="
awk -F', ' '{u[$1]+=$2; m[$1]=($3>m[$1]?$3:m[$1]); n[$1]++}
  END{for(i in u) printf "  gpu%s util_mean=%d%% mem_max=%.1fGiB\n", i, u[i]/n[i], m[i]/1024}' \
  "$OUT/gpu.csv" 2>/dev/null | sort

chown -R 10011:10011 \
  "$OUT" \
  "$STABLEWM_HOME/checkpoints/paired_initializations/$RUN_TAG" \
  "$STABLEWM_HOME/checkpoints"/cm_k1_s*"_$RUN_TAG" \
  "$STABLEWM_HOME/checkpoints"/cm_k5_s*"_$RUN_TAG" \
  2>/dev/null || true

test "$return_code" -eq 0
echo "CONTROLLED-METRIC DLC TRAINING PASSED"
