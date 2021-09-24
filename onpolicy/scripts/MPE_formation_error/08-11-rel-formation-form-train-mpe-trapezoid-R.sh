#!/bin/sh
env="MPE"
scenario="rel_formation_form_error_trapezoid" 
num_landmarks=1
num_agents=4
algo="rmappo"
exp="08-11-rel-formation-form-nav-train-mpe-trapezoid-R"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    let "seed=$seed+1"
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=0 python render/render_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --num_mini_batch 1 \
    --episode_length 250 \
    --num_env_steps 50000000 \
    --ppo_epoch 10 \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --user_name "mapping" \
    --avoid-rew-weight 5 \
    --form-rew-weight 0.05 \
    --nav-rew-weight 0 \
    --num_static_obs 0 \
    --model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error_trapezoid/rmappo/08-11-rel-formation-form-nav-train-mpe-trapezoid/run1/models" \
    --save_gifs \
    --use_render \
    --eval_interval 250
done