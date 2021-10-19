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
from pettingzoo.mpe._mpe_utils.simple_env import make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from tqdm import tqdm
import os


class Run:
    def __init__(
        self, args: Args, environment: to_parallel_wrapper, agents: base_policy
    ) -> None:

        self.args = args
        self.set_seed(self.args.seed)
        print(args.log_dir)

        self.logger = args.logger

        self.agents = agents
        self.env = environment

        hyperparameters = {
            "hparm/" + key: value
            for key, value in vars(args).items()
            if isinstance(value, (int, float))
        }
        self.logger.add_hparams(
            hyperparameters, hyperparameters, run_name=args.exp_name
        )

    def set_seed(self, seed=1000):
        torch.manual_seed = seed
        np.random.seed(seed)

    def run(self):
        total_steps = 0
        args = self.args

        for ep_i in tqdm(range(0, args.n_episodes)):
            observation = self.env.reset()
            self.env.render()
            rewards, dones = 0, False
            for step in range(args.max_cycles - 1):

                actions = self.agents.action(observation)
                observation, rewards, dones, infos = self.env.step(actions)
                self.agents.store(rewards, dones)

                self.env.render()
                total_steps += 1

            reward = sum([reward for reward in rewards.values()])
            self.logger.add_scalar("rewards/end_reward", reward, (ep_i + 1))


if __name__ == "__main__":
    model_args = simple_ppo.args
    model_args["exp_name"] = "less_epochs"
    args = Args(**model_args)

    env = simple_v2.parallel_env(max_cycles=args.max_cycles)
    policy = simple_ppo.policy(args)

    r = Run(args, env, policy)

    r.run()
