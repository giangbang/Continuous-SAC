#!/bin/bash

envs=("Walker2d-v3" "Ant-v2" "Humanoid-v2" "HalfCheetah-v4" "Hopper-v3" "LunarLanderContinuous-v2")
algo="crossq"

for env in "${envs[@]}"; do
    python src/train.py --algo $algo --env_name $env --total_env_step 1000000 \
        --buffer_size 1000000 --actor_log_std_min -20 --batch_size 256 \
        --eval_interval 5000 --critic_tau 0.005 --alpha_lr 3e-4 --num_layers 3 \
        --critic_lr 3e-4 --actor_lr 3e-4 --init_temperature 0.2 --hidden_dim 256 \
        --reward_scale 1 --upd 1
done
