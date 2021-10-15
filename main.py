from pettingzoo.mpe import reference


import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter

from policy import random_policy
from policy import Agent

from dotmap import DotMap

parameters = {
    # exerpiment details
    "chkpt_dir": "tmp/ppo",
    # PPO memory
    "total_memory": 25,
    "batch_size": 5,
    # learning hyper parameters actor critic models
    "n_epochs": 5,
    "alpha": 1e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "policy_clip": 0.1,
    "entropy": 0.01,
}


class parameters:
    def __init__(self) -> None:

        self.n_episodes = 50
        self.max_cycles = 25

        self.set_seed()

        run_num = 1

        log_dir = "runs/logs/"
        self.logger = SummaryWriter(str(log_dir))

    def set_seed(self, seed=1000):
        torch.manual_seed = seed
        np.random.seed(seed)


def run(config: parameters):

    args = DotMap(parameters)
    env = reference.parallel_env()

    agent1 = Agent(args)
    agent2 = Agent(args)

    for ep_i in range(0, config.n_episodes):

        observations = env.reset()
        env.render()
        rewards, dones = 0, False
        for step in range(config.max_cycles):
            actions = {agent: random_policy() for agent in env.agents}
            observations, rewards, dones, infos = env.step(actions)
            env.render()


if __name__ == "__main__":
    config = parameters()
    run(config)
