#!/bin/sh
env="StarCraft2"
map="2s_vs_1sc"
#algo="win_2s3z_xavier_uniform_attn"
algo="ok_2s_vs_1sc_attn1642"
seed_max=1

echo "env is ${env}, map is ${map}, algo is ${algo}, seed is ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train_sc.py --env_name ${env} --algorithm_name ${algo} --map_name ${map} --seed ${seed} --n_rollout_threads 8 --num_mini_batch 1 --episode_length 400 --num_env_steps 10000000 --ppo_epoch 15 --use_clipped_value_loss --cuda --attn --attn_heads 2
    echo "training is done!"
done
