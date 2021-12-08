from pettingzoo.mpe import reference


import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter
import torch as T
from ppo_shared import random_policy
from ppo_shared import Agent
from ppo_shared import actions_to_discrete
from dotmap import DotMap
from tqdm import tqdm

params = {
    # exerpiment details
    "chkpt_dir": "tmp/ppo",
    # PPO memory
    "total_memory": 25,
    "batch_size": 5,
    # learning hyper parameters actor critic models
    "n_epochs": 5,
    "alpha": 1e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "policy_clip": 0.1,
    "entropy": 0.01,
}


class Utils:
    def __init__(self) -> None:

        self.n_episodes = 5000
        self.max_cycles = 50

        self.set_seed()

        run_num = 1

        log_dir = "runs/ppo_shared/"
        self.logger = SummaryWriter(str(log_dir))

    def set_seed(self, seed=1000):
        torch.manual_seed = seed
        np.random.seed(seed)


def run(config: Utils):

    args = DotMap(params)
    env = reference.parallel_env()

    policy = Agent(args)
    steps = 0

    for ep_i in tqdm(range(0, config.n_episodes)):
        observation = env.reset()
        env.render()
        rewards, dones = 0, False
        total_reward = 0

        for step in range(config.max_cycles - 1):

            actions = {}
            to_remember = {}

            for agent in env.agents:
                obs = observation[agent]
                obs_batch = np.concatenate(
                    [
                        obs["current_velocity"],
                        obs["landmarks"],
                        obs["goal"],
                        obs["communication"],
                    ]
                )
                obs_batch = T.tensor([obs_batch], dtype=T.float)

                (
                    move_probs,
                    communicate_probs,
                    move_action,
                    communicate_action,
                    value,
                ) = policy.choose_action(obs_batch)

                to_remember[agent] = (
                    obs_batch,
                    move_action,
                    move_probs,
                    communicate_action,
                    communicate_probs,
                    value,
                )

                actions[agent] = actions_to_discrete(
                    move_action, communicate_action
                ).item()

            observation, rewards, dones, infos = env.step(actions)
            end_reward = 0
            for agent in env.agents:
                if agent == "agent_0":
                    memory = policy.memory1
                else:
                    memory = policy.memory2
                policy.remember(
                    to_remember[agent][0],
                    to_remember[agent][1],
                    to_remember[agent][2],
                    to_remember[agent][3],
                    to_remember[agent][4],
                    to_remember[agent][5],
                    -rewards[agent],
                    dones[agent],
                    memory,
                )
                end_reward += rewards[agent]
                # break

            env.render()

            steps += 1

            if step % args.total_memory == 0:
                for agent in env.agents:
                    if agent == "agent_0":
                        memory = policy.memory1
                    else:
                        memory = policy.memory2
                    policy.learn(memory)

        config.logger.add_scalar("rewards/end_reward", end_reward, (ep_i + 1))

        # break


if __name__ == "__main__":
    config = Utils()
    run(config)
