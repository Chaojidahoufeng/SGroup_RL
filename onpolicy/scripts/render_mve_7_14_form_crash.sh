#!/bin/sh
env="MVE"
scenario="relative_formation"  # simple_speaker_listener # simple_spread
num_landmarks=2
num_agents=3
algo="mappo"
exp="7_14_form_crash"
seed_max=2

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=1 \
    python render/render_mve.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --episode_length 25 \
    --render_episodes 5 \
    --model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MVE/relative_formation/mappo/7_14_form_crash/run4/models" \
    --use_recurrent_policy \
    --usegui
done