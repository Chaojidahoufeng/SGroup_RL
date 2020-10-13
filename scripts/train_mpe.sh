#!/bin/sh
env="MPE"
scenario="simple_spread"
num_landmarks=3
num_agents=3
algo="check_wandb"
seed_max=3

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, seeds is ${seed_max}"
CUDA_VISIBLE_DEVICES=7 python train_mpe.py --env_name ${env} --algorithm_name ${algo} --scenario_name ${scenario} --num_agents ${num_agents} --num_landmarks ${num_landmarks} --seed ${seed_max} --n_rollout_threads 20 --num_mini_batch 1 --episode_length 25 --num_env_steps 4000 --ppo_epoch 15  --gain 0.01 --recurrent_policy
echo "training is done!"
