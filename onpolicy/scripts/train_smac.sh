#!/bin/sh
env="StarCraft2"
map="6h_vs_8z"
algo="mappo"
exp="originalstate_mlp_mustalive"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=2 python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed 4 --n_training_threads 1 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --use_eval --add_center_xy --user_name "zoeyuchao" --use_mustalive --use_recurrent_policy
done
