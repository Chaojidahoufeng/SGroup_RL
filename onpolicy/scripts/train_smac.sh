#!/bin/sh
env="StarCraft2"
map="MMM2"
algo="rmappo"
exp="ippo"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed 7 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 50000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --user_name "zoeyuchao" --use_centralized_V
done
