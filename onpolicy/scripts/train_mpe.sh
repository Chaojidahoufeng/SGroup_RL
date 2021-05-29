#!/bin/sh
env="MPE"
scenario="rel_formation_only" 
num_landmarks=3
num_agents=3
algo="rmappo"
exp="debug"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    srun -p gpu -N 1 -n 1 --export=ALL\ 
    python -u train/train_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed 50 \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --num_mini_batch 1 \
    --episode_length 250 \
    --num_env_steps 20000000 \
    --ppo_epoch 10 \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "mapping"
done
