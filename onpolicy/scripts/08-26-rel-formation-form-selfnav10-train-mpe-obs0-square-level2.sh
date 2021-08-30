#!/bin/sh
env="MPE"
scenario="rel_formation_form_error" 
num_landmarks=1
num_agents=4
algo="rmappo"
exp="08-26-rel-formation-form-selfnav10-train-mpe-obs0-square-level2"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed 50 \
    --n_training_threads 4 \
    --n_rollout_threads 32 \
    --num_mini_batch 1 \
    --episode_length 250 \
    --num_env_steps 30000000 \
    --ppo_epoch 10 \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "mapping" \
    --avoid-rew-weight 5 \
    --form-rew-weight 0.05 \
    --nav-rew-weight 0 \
    --self-nav-rew-weight 10 \
    --num_static_obs 0 \
    --map-max-size 3600 \
    --static_obs_intensity 0e-6 \
    --topo_type "square" \
    --use_softmax_last \
    --model_dir "/home/yanyz/data/MARL-yuzi/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/rmappo/08-26-rel-formation-form-selfnav10-train-mpe-obs0-square/run1/models"
done