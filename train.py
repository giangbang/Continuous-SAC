from buffer import ReplayBuffer
from sac import SAC
import torch
import numpy as np
import gym
import argparse

def evaluate(env, agent, n_rollout = 10):   
    tot_rw = 0
    for _ in range(n_rollout):
        state = env.reset()
        done = False
        agent.eval()
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32))
            action = agent.select_action(state).reshape(-1)
            
            next_state, reward, done, _ = env.step(action)
            tot_rw += reward
            state = next_state
    agent.train()
    return tot_rw / n_rollout
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training SAC continuous')
    
    parser.add_argument('--env_name', type=str, default='LunarLanderContinuous-v2',
                        help='Environment name')
    parser.add_argument('--total_env_step', type=int, default=1e1,
                        help='Total number of env steps')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='Max replay buffer size')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='Log eval info every `eval_interval` env steps')
    parser.add_argument('--start_step', type=int, default=1000,
                        help='Random staring steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size (sampling from replay buffer)')
    
    args, unknown = parser.parse_known_args()
    
    
    env = gym.make(args.env_name)
    
    action_shape      = env.action_space.shape
    observation_shape = env.observation_space.shape
    
    sac_agent = SAC(observation_shape[0], action_shape[0])
    buffer    = ReplayBuffer(observation_shape, action_shape, 
                args.buffer_size, args.batch_size)
                
    print('Action dim: {} | Observation dim: {}'.format(observation_shape, action_shape))
    
    his = []
    loss = []
    
    state = env.reset()
    for env_step in  range(int(args.total_env_step)):
        if env_step < args.start_step: 
            action = env.action_space.sample()
        else :
            state = torch.from_numpy(np.array(state,dtype=np.float32))
            action = sac_agent.select_action(state).reshape(-1)
        
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, next_state, done)
        
        loss.append(sac_agent.update(buffer))
        
        state = next_state
        env_step += 1
        if done: 
            state = env.reset()
        if (env_step + 1) % args.eval_interval == 0:
            eval_return = evaluate(gym.make(args.env_name), sac_agent)
            his.append(eval_return)
            print('mean reward after {} env step: {:.2f}'.format(env_step+1, eval_return))
            print('critic loss: {:.2f} | actor loss: {:.2f} | alpha loss: {:.2f}'.format(
                    *np.mean(list(*zip(*loss[-10:])), axis=-1)
                    ))
            
    import matplotlib.pyplot as plt
    plt.plot(his)
    plt.savefig('res.png')
    