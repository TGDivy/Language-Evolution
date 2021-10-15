from pettingzoo.mpe import simple_reference_v2


import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter

from policy import random_policy


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

    env = simple_reference_v2.parallel_env()

    for ep_i in range(0, config.n_episodes):

        obs = env.reset()
        env.render()

        for step in range(config.max_cycles):
            actions = {agent: random_policy() for agent in env.agents}
            observations, rewards, dones, infos = env.step(actions)
            env.render()


if __name__ == "__main__":
    config = parameters()
    run(config)
