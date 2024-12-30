#!/bin/bash

envs=("Walker2d-v5" "Ant-v5" "Humanoid-v5" "HalfCheetah-v5" "Hopper-v5" "LunarLanderContinuous-v3")
algo="crossq"

for env in "${envs[@]}"; do
    python src/train.py --algo $algo --env_name $env --total_env_step 1000000 \
        --buffer_size 1000000 --actor_log_std_min -20 --batch_size 256 \
        --eval_interval 5000 --critic_tau 0.005 --alpha_lr 3e-4 --num_layers 3 \
        --critic_lr 3e-4 --actor_lr 3e-4 --init_temperature 0.2 --hidden_dim 256 \
        --reward_scale 1 --upd 1
done
