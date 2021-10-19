from pettingzoo.mpe import reference
from pettingzoo.mpe import simple_v2

import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter
import torch as T

from policies import simple_ppo
import policies
from policies.base import base_policy, Args
from pettingzoo.utils.conversions import to_parallel_wrapper

from tqdm import tqdm


class Run:
    def __init__(
        self, args: Args, environment: to_parallel_wrapper, agents: base_policy
    ) -> None:

        self.args = args
        self.set_seed(self.args.seed)
        self.logger = SummaryWriter(args.log_dir)

        self.agents = agents
        self.env = environment

    def set_seed(self, seed=1000):
        torch.manual_seed = seed
        np.random.seed(seed)

    def run(self):
        total_steps = 0

        for ep_i in tqdm(range(0, self.n_episodes)):
            observation = self.env.reset()
            self.env.render()
            reward, dones = 0, False
            total_reward = 0

            for step in range(self.max_cycles - 1):

                actions = self.agents(observation, reward, dones, total_steps)
                observation, reward, dones, infos = self.env.step(actions)
                self.agents.store(reward, dones)

                self.env.render()
                total_steps += 1

            # self.config.logger.add_scalar("rewards/end_reward", end_reward, (ep_i + 1))


if __name__ == "__main__":
    args = Args(**simple_ppo.args)
    environment = simple_v2.parallel_env()
    policy = simple_ppo.policy()

    r = Run(args, environment, policy)

    r.run()
