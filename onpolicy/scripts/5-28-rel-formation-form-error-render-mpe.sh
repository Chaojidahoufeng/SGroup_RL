#!/bin/sh
env="MPE"
scenario="rel_formation_form_error"
num_landmarks=1
num_agents=4
algo="rmappo"
exp="5-28-rel-formation-form-error-comm-avoid-5-form-0_05-dist-0"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    CUDA_VISIBLE_DEVICES=4 python render/render_mpe.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --episode_length 250 \
    --render_episodes 5 \
    --model_dir "~/data/gitlab/onpolicy/onpolicy/scripts/results/MPE/rel_formation_form_error/5-28-rel-formation-form-error-comm-avoid-5-form-0_05-dist-0/run-14/models/"
done
