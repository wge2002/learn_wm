#!/bin/bash
# Overnight LeWM two-goal control sweep (resilient: one detached orchestrator).
# Tiny smoke first; only proceed to the full multi-seed/horizon sweep if it works.
cd /code/wge/stable-worldmodel
PY=/root/miniconda/envs/lerobot/bin/python
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=5
LOG=outputs/overnight_lewm/orchestrator.log
mkdir -p outputs/overnight_lewm
echo "=== overnight start $(date) ===" > $LOG

run () {  # tag + args
  tag=$1; shift
  od=outputs/overnight_lewm/$tag
  mkdir -p $od
  echo ">>> $(date) running $tag : $*" >> $LOG
  $PY scripts/plan/lewm_twogoal_control.py "$@" --device cuda --output-dir $od >> $od/run.log 2>&1
  echo "<<< $(date) done $tag rc=$?" >> $LOG
  grep -E "sep=|discrete -|reach rate|between-ness" $od/RESULT.txt 2>/dev/null >> $LOG || echo "   NO RESULT for $tag" >> $LOG
}

# 1) smoke gate (far pairing) -------------------------------------------------
run smoke --goal-pairing far --n-train 50 --n-eval 24 --n-samples 128 --pred-epochs 30 --horizon 8
if [ ! -f outputs/overnight_lewm/smoke/RESULT.txt ]; then
  echo "SMOKE FAILED -> abort full sweep" >> $LOG; exit 1
fi
echo "smoke OK -> full sweep" >> $LOG

# 2) full sweep: far pairing, horizons {8,16}, seeds {0,1,2} -------------------
for H in 8 16; do
  for S in 0 1 2; do
    run far_H${H}_s${S} --goal-pairing far --horizon $H --seed $S \
        --n-train 400 --n-eval 128 --n-samples 512 --pred-epochs 200 --lam 1.0
  done
done
echo "=== overnight done $(date) ===" >> $LOG
