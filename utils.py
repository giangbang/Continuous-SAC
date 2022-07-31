import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training SAC continuous')
    # environment
    parser.add_argument('--env_name', default='LunarLanderContinuous-v2')
    parser.add_argument('--frame_stack', default=3, type=int)
    # replay buffer
    parser.add_argument('--buffer_size', default=1000000, type=int)
    # train
    parser.add_argument('--start_step', default=1000, type=int)
    parser.add_argument('--total_env_step', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--hidden_dim', default=64, type=int)
    # eval
    parser.add_argument('--eval_interval', default=10000, type=int)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    # actor
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    
    parser.add_argument('--num_layers', default=4, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')

    args, unknown = parser.parse_known_args()
    return args


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True