# Continuous-SAC-Pytorch

Reproduce results from [Continuous SAC paper](https://arxiv.org/pdf/1812.05905.pdf).

This repo is based on several SAC implementations, mainly [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [author's implementation](https://github.com/haarnoja/sac) and [SAC-Continuous-Pytorch](https://github.com/XinJingHao/SAC-Continuous-Pytorch).

Update 12/28/24: [CrossQ](https://arxiv.org/abs/1902.05605) added.

## Installation
After cloning the repo, install requirements by running
```
pip install -r requirements.txt
```
or it can be installed with `pip`
```
pip install git+https://github.com/giangbang/Continuous-SAC.git
```

## How to run
```
python src/train.py --env_name HalfCheetah-v4 --total_env_step 1000000 --buffer_size 1000000 --actor_log_std_min -20 --batch_size 256 --eval_interval 5000 --critic_tau 0.005 --alpha_lr 3e-4 --num_layers 3 --critic_lr 3e-4 --actor_lr 3e-4 --init_temperature 1 --hidden_dim 256 --reward_scale 1 --upd 1
```
Some benchmark environments from `gym`, for example `mujoco` or `RacingCar` and `LunarLanderContinuous`, need to be installed separately from by `pip install gymnasium[mujoco]` or `pip install gymnasium[box2d]`.

It can also be run from terminal by the following command from the entry point, if installed by `setup.py`
```
sac_continuous --env_name HalfCheetah-v4 --total_env_step 1_000_000
```
add `--algo crossq` to run with CrossQ.

## Results

Most of the experiments used the same hyper-parameters shown in the table. Set `seed` to `-1` to use random seed every run.

| Hyper params  |Value  |  Hyper params   | Value  | 
|----------|:-------------:|----------|:-------------:|
| `reward_scale`                   | 1.0     | `critic_lr`  | 0.0003 |
| `buffer_size`                    | 1000000 |`critic_tau`                     | 0.005 |
| `start_step`                     | 1000    |`actor_lr`                       | 0.0003 |
| `total_env_step`                 | 1000000 |`actor_log_std_min`              | -20.0 | 
| `batch_size`                     | 256     |`actor_log_std_max`              | 2 |
| `hidden_dim`                     | 256     |`num_layers`                     | 3  |
| `upd`                 	       | 1       |`discount`                       | 0.99   |
| `algo`                           | (`sac`\|`crossq`)|`init_temperature`               | 0.2 |
| `eval_interval`                  | 5000    |`alpha_lr`                       | 0.0003 |
| `num_eval_episodes`              | 10      |`seed`                           | -1  |


SAC and CrossQ results are shown below
![avatar](/assets/plt.png)  
## Comments
Here are some critical minor implementation details but are crucial to achieve the desired performance; 

For SAC:
- Handle done separately by truncation and termination. SAC performs much worse in some environment when we do not correctly implement this (about 2k rewards in difference in `Half-Cheetah`).
- Using ReLU activation function slightly increases the performance, compared to using Tanh. I suspect that the three layer Tanh Activation network are not powerful enough to learn the value function of tasks with high reward range like Mujoco.
- Using `eps=1e-5` in Adam Optimizer does not provide any significant boost as suggested in `stable-baselines3`.
- Initial temperature of `alpha` (entropy coefficient) can largely impact the final performance (than one might expect). In `Half-Cheetah`, `alpha` starting with the values of 0.2 and 1 can yield a gap ~ 1-2k in final performance.
- Changing `actor_log_std_min` from -20 to -10 can sometimes reduce the performance, but this might not be consistent through out seeds

For CrossQ;
- Batch Renorm is the key factor for stable training without a target network and is the most tricky part. The running mean/variance of batch renorm should __only__ be recorded when the networks are _trained_. For example, critic should be in _eval_ mode when optimizing the actor.
Enabling Batch renorm in actor does not yield good results in my experiments.
