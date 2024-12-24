"""This file is not well-maintained and contains ugly code
since it should not be copied to other projects
"""

from buffer import ReplayBuffer
from sac import SAC
import torch
import numpy as np
import gymnasium as gym
from utils import parse_args, pprint, seed_everything
from logger import Logger


def evaluate(env, agent, n_rollout=10):
    tot_rw = 0
    for _ in range(n_rollout):
        state, _ = env.reset()
        done = False
        agent.eval()
        while not done:
            state = torch.from_numpy(np.array(state, dtype=np.float32))
            action = agent.select_action(state).reshape(-1)

            next_state, reward, terminated, truncated, _ = env.step(action)
            tot_rw += reward
            state = next_state
            done = terminated or truncated
    agent.train()
    return tot_rw / n_rollout


def main():
    args = parse_args()
    logger = Logger()
    logger.add_run_command()

    if args.seed > 0:
        seed_everything(args.seed)

    # creating the training and testing environments here,
    # all the wrapping and preprocessing should be placed here.
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)

    assert isinstance(env.action_space, gym.spaces.Box), "Only support continuous env"

    action_shape = env.action_space.shape
    observation_shape = env.observation_space.shape

    sac_agent = SAC(observation_shape[0], action_shape[0], **vars(args))
    buffer = ReplayBuffer(
        observation_shape, action_shape, args.buffer_size, args.batch_size
    )

    def get_batch():
        if args.sample_buffer == "without_replace":
            return buffer.sample_without_replace()
        elif args.sample_buffer == "with_replace":
            return buffer.sample()
        else:
            raise NotImplementedError()

    pprint(vars(args))
    print(
        "Action dim: {} | Observation dim: {}".format(action_shape, observation_shape)
    )

    his = []
    loss = []

    train_returns = 0
    num_network_update = 0

    state, _ = env.reset()
    for env_step in range(int(args.total_env_step)):
        if env_step < args.start_step:
            action = env.action_space.sample()
        else:
            state = torch.from_numpy(np.array(state, dtype=np.float32))
            action = sac_agent.select_action(state).reshape(-1)

        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.add(state, action, reward, next_state, terminated, truncated, info)
        train_returns += reward

        if env_step > args.batch_size:
            while num_network_update * args.upd <= env_step + 1:
                loss.append(sac_agent.update(get_batch=get_batch))
                num_network_update += 1

        state = next_state
        done = terminated or truncated
        if done:
            state, _ = env.reset()
            logger.add_scalar("train/returns", train_returns, env_step)
            train_returns = 0
        if (env_step + 1) % args.eval_interval == 0:
            eval_return = evaluate(test_env, sac_agent, args.num_eval_episodes)
            his.append(eval_return)
            print(
                "mean reward after {} env step: {:.2f}".format(
                    env_step + 1, eval_return
                )
            )
            print(
                "critic loss: {:.2f} | actor loss: {:.2f} | alpha loss: {:.2f} | alpha: {:.2f}".format(
                    *np.mean(list(zip(*loss[-10:])), axis=-1),
                    sac_agent.log_ent_coef.exp().item()
                )
            )
            logger.add_scalar("eval/returns", eval_return, env_step)
        if (env_step + 1) % 5000 == 0:
            logger.log_stdout()
        if (env_step + 1) % 100 == 0 and env_step > args.batch_size:
            logger.add_scalar("train/entropy", sac_agent.entropy, env_step)
            logger.add_scalar("train/fps", logger.fps(), env_step)

    logger.close()

    import matplotlib.pyplot as plt

    x, y = np.linspace(0, args.total_env_step, len(his)), his
    plt.plot(x, y)
    plt.title(args.env_name)
    plt.savefig("res.png")

    import pandas as pd

    data_dict = {
        "rollout/ep_rew_mean": y,
        "time/total_timesteps": x,
    }  # formated as stable baselines
    df = pd.DataFrame(data_dict)

    df.to_csv("sac_continuous_progress.csv", index=False)


if __name__ == "__main__":
    main()
