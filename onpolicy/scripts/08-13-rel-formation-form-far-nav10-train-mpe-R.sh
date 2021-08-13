#!/bin/sh
env="MPE"
scenario="rel_formation_form_error_far" 
num_landmarks=1
num_agents=4
algo="rmappo"
exp="08-13-rel-formation-form-far-nav10-train-mpe-R"
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
    --avoid-rew-weight 5 \
    --form-rew-weight 0.05 \
    --nav-rew-weight 500 \
    --num_static_obs 5 \
    --use_render \
    --episode_length 250 \
    --render_episodes 5 \
    --num_static_obs 0 \
    --model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error_far/rmappo/08-13-rel-formation-form-far-nav10-train-mpe/run1/models" \
    --save_gifs \
    --eval_interval 250
done