#!/bin/sh
env="Hanabi"
hanabi="Hanabi-Small"
num_agents=2
#algo="value0_samedim_fullminimal2_parallel1000_length80_batch5_ppo15_lr7e-4_512_1mlp2gru_relu_100M"
algo="check"
seed_max=1
ulimit -n 22222

echo "env is ${env}, algo is ${algo}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=6 python train/train_hanabi.py --env_name ${env} --algorithm_name ${algo} --hanabi_name ${hanabi} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1000 --num_mini_batch 5 --episode_length 80 --num_env_steps 100000000 --ppo_epoch 15 --gain 0.01 --lr 7e-4 --hidden_size 512 --layer_N 2 --eval --n_eval_rollout_threads 100 
    echo "training is done!"
done
