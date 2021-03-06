#!/bin/sh
env="MPE"
scenario="rel_formation_form_error" 
num_landmarks=1
num_agents=3
algo="rmappo"
exp="08-18-rel-formation-form-selfnav10-train-mpe-obs1-triangle-R"
seed=12

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
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
--nav-rew-weight 0 \
--self-nav-rew-weight 10 \
--num_static_obs 10 \
--use_render \
--episode_length 400 \
--render_episodes 1 \
--num_static_obs 0 \
--model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/rmappo/08-18-rel-formation-form-selfnav10-train-mpe-obs1-triangle/run1/models" \
--save_gifs \
--eval_interval 250 \
--map-max-size 3600 \
--static_obs_intensity 2e-6 \
--topo_type "square"