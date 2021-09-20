#!/bin/sh
env="MVE"
scenario="yyz3-avoid"  # simple_speaker_listener # simple_spread
num_landmarks=1
num_agents=4
num_obstacles=2
algo="mappo"
exp="MVE-09-19-train-4a-mo-form-avoid2"
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
    --num_env_steps 1000000 \
    --ppo_epoch 10 --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --use_recurrent_policy \
    --direction_alpha 0.1 \
    --ideal_side_len 2 \
    --form_rew_weight 20.0 \
    --spring_rew_weight 0.0 \
    --motion_rew_weight 2.0
    # --usegui
#--wandb_name "tartrl"
#--use_wandb