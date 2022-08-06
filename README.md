# Continuous-SAC-Pytorch

Reproduce results from [Continuous SAC paper](https://arxiv.org/pdf/1812.05905.pdf).

This repo is based on several SAC implementations, mainly [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [author's implementation](https://github.com/haarnoja/sac) and [SAC-Continuous-Pytorch](https://github.com/XinJingHao/SAC-Continuous-Pytorch).

## How to run
```
python train.py --env_name HalfCheetah-v4 --total_env_step 1000000 --buffer_size 1000000 --actor_log_std_min -20 --batch_size 256 --eval_interval 5000 --critic_tau 0.005 --alpha_lr 3e-4 --num_layers 3 --critic_lr 3e-4 --actor_lr 3e-4 --init_temperature 1 --hidden_dim 256 --reward_scale .2 --train_freq 1 --gradient_steps 1
```

## Results

![avatar](https://github.com/giangbang/Continuous-SAC/blob/master/results/sac.png)  
