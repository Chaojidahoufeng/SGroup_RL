#!/bin/sh
env="MVE"
scenario="predator"  # predator-prey
algo="mappo"
exp="MVE-08-12-train-predator-R"
seed_max=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 \
    python render/render_mve.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --user_name "yuzi" \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --episode_length 25 \
    --render_episodes 5 \
    --model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MVE/predator/rmappo/MVE-08-12-train-predator/run13/models" \
    --use_recurrent_policy \
    --usegui \
    --num_agents 1 \
    --num_landmarks 2 \
    --num_obstacles 2
    #--wandb_name "tartrl"
    #--use_wandb
done