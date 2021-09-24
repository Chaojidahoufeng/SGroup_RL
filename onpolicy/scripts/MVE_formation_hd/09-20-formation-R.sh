#!/bin/sh
env="MVE"
scenario="formation_hd_boundless_env"  # predator-prey
algo="mappo"
exp="MVE-09-20-formation-hd-boundless-env-test-R"
num_landmarks=3
num_agents=3
num_obstacles=0
seed=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed}"
echo "seed is ${seed}:"
CUDA_VISIBLE_DEVICES=1 \
python ../render/render_mve.py \
--env_name ${env} \
--algorithm_name ${algo} \
--experiment_name ${exp} \
--scenario_name ${scenario} \
--user_name "yuzi" \
--seed ${seed} \
--n_training_threads 1 \
--n_rollout_threads 1 \
--use_render \
--episode_length 10 \
--render_episodes 5 \
--model_dir "/home/yanyz/yanyz/gitlab/onpolicy/onpolicy/scripts/results/MVE/formation_hd_boundless_env/mappo/MVE-09-20-formation-hd-boundless-env-test/run1/models" \
--use_recurrent_policy \
--usegui \
--num_agents ${num_agents} \
--num_landmarks ${num_landmarks} \
--num_obstacles ${num_obstacles} \
--ideal_side_len 2.0 \
#--save_gifs
#--wandb_name "tartrl"
#--use_wandb