#!/bin/sh
env="Highway"
scenario="highway-v0"
task="defend"
n_defenders=1
n_attackers=0
n_dummies=0
algo="mappo"
exp="debug"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do

    CUDA_VISIBLE_DEVICES=0 python render/render_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_render_rollout_threads 1 --horizon 40 --use_render --use_wandb --model_dir "models" --render_episodes 100 --dt 1 --simulation_frequency 5 --use_recurrent

done
