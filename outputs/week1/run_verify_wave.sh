#!/bin/bash
# Anchor+dose verification wave (= Horizon-Bundle baseline-2 construction).
# 6 trainings, 1 GPU each on A100 (fits batch 128); then far-goal eval + certs.
# Usage: NGPU=4 bash run_verify_wave.sh
set -u
PY=${PY:-python}
DS=${DS:-$HOME/.stable_worldmodel/pusht_expert_train.h5}
OUT=${OUT:-outputs/week1}
mkdir -p $OUT
NGPU=${NGPU:-4}
I=0
train() { # NAME SEED CONFIG EXTRA
  CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY scripts/train/lewm.py --config-name $3 \
    output_model_name=$1 subdir=$1 seed=$2 $4 \
    trainer.max_epochs=30 trainer.devices=1 \
    loader.num_workers=6 loader.prefetch_factor=2 \
    data.dataset.name=$DS > $OUT/train_$1.log 2>&1 &
  I=$((I + 1))
  [ $((I % NGPU)) -eq 0 ] && wait
}
# wave 1 (4 runs) + wave 2 (2 runs)
train mix_d192_g10_s1 1    lewm_mix "wm.mix_gamma=1.0"
train mix_d192_g10_s7 7    lewm_mix "wm.mix_gamma=1.0"
train mix_d192_g05    3072 lewm_mix "wm.mix_gamma=0.5"
train mix_d192_g20    3072 lewm_mix "wm.mix_gamma=2.0"
train pd_d192_k5_s7   7    lewm_multistep ""
train pd_d192_k1_s7   7    lewm "data.dataset.num_steps=4"
wait
echo "TRAININGS DONE $(date)"

# eval: near+far goals x 3 eval seeds + certificates
for NAME in mix_d192_g10_s1 mix_d192_g10_s7 mix_d192_g05 mix_d192_g20 pd_d192_k5_s7 pd_d192_k1_s7 mix_d192_g10 mix_d192_g03 mix_d192_g01; do
  $PY outputs/pd/make_eval_dir.py $NAME 2>/dev/null
  for OFF in 25 40; do
    for SEED in 42 123 7; do
      CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY scripts/plan/eval_wm.py \
        policy=${NAME}_eval seed=$SEED +plan_config.history_len=3 \
        eval.goal_offset_steps=$OFF eval.num_eval=50 eval.video=false \
        eval.dataset_name=$DS \
        output.filename=verify_${NAME}_off${OFF}_seed${SEED}.txt \
        > $OUT/plan_${NAME}_off${OFF}_s${SEED}.log 2>&1 &
      I=$((I + 1))
      [ $((I % NGPU)) -eq 0 ] && wait
    done
  done
done
wait
for NAME in mix_d192_g10_s1 mix_d192_g10_s7 mix_d192_g05 mix_d192_g20; do
  CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY scripts/plan/regime_stepB_eval_data.py \
    --policy ${NAME}_eval --num-samples 4000 --seed 2025 \
    --output outputs/pd/probe_${NAME}.npz > $OUT/dump_${NAME}.log 2>&1
  CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY outputs/gauge/jac_spectrum.py \
    --policy ${NAME}_eval --data outputs/pd/probe_${NAME}.npz \
    --out outputs/gauge/jac_${NAME}.json > $OUT/jac_${NAME}.log 2>&1
  CUDA_VISIBLE_DEVICES=$((I % NGPU)) $PY outputs/gauge/separation.py \
    --policy ${NAME}_eval --data outputs/pd/probe_${NAME}.npz --perturb first \
    --out outputs/gauge/sepfirst_${NAME}.json > $OUT/sep_${NAME}.log 2>&1
  $PY outputs/pd/probe_r2.py --data outputs/pd/probe_${NAME}.npz \
    --out outputs/pd/probe_${NAME}.json > $OUT/r2_${NAME}.log 2>&1
  I=$((I + 1))
done
echo "VERIFY WAVE ALL DONE $(date)"
