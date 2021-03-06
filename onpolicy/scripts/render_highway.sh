#!/bin/sh
env="Highway"
scenario="highway-v0"
task="defend"
n_defenders=1
n_attackers=1
n_dummies=0
algo="rmappo"
exp="attacker10LoadDefenderPretrained-defender1-npc0"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=0 python render/render_highway.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --task_type ${task} --n_attackers ${n_attackers} --n_defenders ${n_defenders} --n_dummies ${n_dummies} --seed ${seed} --n_training_threads 1 --n_render_rollout_threads 1 --horizon 40 --use_render --save_gifs --use_wandb --model_dir "/home/jimmy/work/RL/env/onpolicy/onpolicy/scripts/results/attack-task-attacker10LoadDefenderPretrained-defender1-npc0/files" --render_episodes 100 --dt 1 --simulation_frequency 5 --collision_reward -1
done
