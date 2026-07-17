#!/bin/bash
# Gate A: horizon-matching matrix (zero training, existing checkpoints).
# K_train x H_plan x goal_offset, fixed-candidates protocol (s=300).
# Fixed-model-calls protocol: rerun the H_plan rows with
#   solver.num_samples = round(300 * 5 / H)  (matches candidates x horizon).
# Usage: NGPU=4 bash run_gateA_matrix.sh   (set PY/DS for the A100 box)
set -u
PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
OUT=${OUT:-outputs/week1}
mkdir -p $OUT
NGPU=${NGPU:-4}
I=0
run() { # MODEL H OFF SAMPLES TAG
  CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY scripts/plan/eval_wm.py \
    policy=${1}_eval seed=42 +plan_config.history_len=3 \
    plan_config.horizon=$2 eval.goal_offset_steps=$3 solver.num_samples=$4 \
    eval.num_eval=50 eval.video=false eval.dataset_name=$DS \
    output.filename=gateA_${1}_h${2}_off${3}_${5}.txt > $OUT/gateA_${1}_h${2}_off${3}_${5}.log 2>&1 &
  I=$((I + 1))
  [ $((I % NGPU)) -eq 0 ] && wait
}
MODELS="iter2_baseline pd_d192_k2 pd_d192_k3 iter2_multistep pd_d192_k10"
for M in $MODELS; do
  for H in 1 3 5 8 10; do
    for OFF in 25 40 60; do
      run $M $H $OFF 300 fixcand
    done
  done
done
wait
# fixed-model-calls protocol (calls ~ samples*H; anchor at H=5,s=300)
for M in $MODELS; do
  for H in 1 3 5 8 10; do
    S=$((1500 / H))
    for OFF in 25 40 60; do
      run $M $H $OFF $S fixcalls
    done
  done
done
wait
echo "GATE A MATRIX DONE $(date)"
