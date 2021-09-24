#!/bin/sh
env="MVE"
scenario="yyz3"  # simple_speaker_listener # simple_spread
num_landmarks=1
num_agents=4
num_obstacles=0
algo="mappo"
exp="MVE-09-12-train-yyz3-4agent"
seed=2

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python train/train_mve.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --user_name "yuzi" \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --num_obstacles ${num_obstacles}\
    --seed ${seed} \
    --n_training_threads 4 \
    --n_rollout_threads 32 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 10000000 \
    --ppo_epoch 10 --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --use_recurrent_policy \
    --direction_alpha 0.1 \
    --ideal_side_len 2.5 \
    --form_rew_weight 0.0 \
    --spring_rew_weight 1.0
    # --usegui
#--wandb_name "tartrl"
#--use_wandb