from pettingzoo.mpe import reference


import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter
import torch as T
from policy import random_policy
from policy import Agent
from policy import actions_to_discrete
from dotmap import DotMap
from tqdm import tqdm

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

    policies = {"agent_0": Agent(args), "agent_1": Agent(args)}

    for ep_i in tqdm(range(0, config.n_episodes)):
        observation = env.reset()
        env.render()
        rewards, dones = 0, False

        for step in range(config.max_cycles):

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
                obs_batch = T.tensor([obs_batch])

                (
                    move_probs,
                    communicate_probs,
                    move_action,
                    communicate_action,
                    value,
                ) = policies[agent].choose_action(obs_batch)

                to_remember[agent] = (
                    obs_batch,
                    move_action,
                    move_probs,
                    communicate_action,
                    communicate_probs,
                    value,
                )

                actions[agent] = actions_to_discrete(move_action, communicate_action)

            observation, rewards, dones, infos = env.step(actions)

            for agent in env.agents:
                policies[agent].remember(
                    to_remember[agent][0],
                    to_remember[agent][1],
                    to_remember[agent][2],
                    to_remember[agent][3],
                    to_remember[agent][4],
                    to_remember[agent][5],
                    rewards[agent],
                    dones[agent],
                )

            env.render()

        for agent in env.agents:
            policies[agent].learn()


if __name__ == "__main__":
    config = parameters()
    run(config)
