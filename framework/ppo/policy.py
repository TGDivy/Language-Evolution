import numpy as np
from framework.utils.base import base_policy
from framework.ppo.ppo import Agent
import torch as T
from torch.utils.tensorboard import SummaryWriter


class ppo_policy(base_policy):
    def __init__(self, args, n_agents, input_shape, num_layers, num_filters) -> None:
        self.args = args

        self.agents = []
        for _ in range(n_agents):
            agent = Agent(args, input_shape, num_layers, num_filters)
            self.agents.append(agent)

        self.to_remember = {}

    def add_logger(self, logger: SummaryWriter):
        self.logger = logger

    def action(self, observations):

        actions = []
        for i, agent in enumerate(self.agents):

            obs = observations[i]

            obs_batch = T.tensor(np.array([obs]), dtype=T.float, device="cuda")

            (
                move_probs,
                communicate_probs,
                move_action,
                communicate_action,
                value,
            ) = agent.choose_action(obs_batch)

            self.to_remember[i] = (
                obs_batch,
                move_action,
                move_probs,
                communicate_action,
                communicate_probs,
                value,
            )
            if self.args.communicate:
                action = (
                    move_action.item() * communicate_action.item() + move_action.item()
                )
            else:
                action = move_action.item()
            actions.append(action)

        return actions, (value, move_probs)

    def store(self, rewards, dones):

        for i, agent in enumerate(self.agents):
            agent.remember(
                self.to_remember[i][0],
                self.to_remember[i][1],
                self.to_remember[i][2],
                self.to_remember[i][3],
                self.to_remember[i][4],
                self.to_remember[i][5],
                T.tensor(rewards[i]),
                dones[i],
            )

            if agent.memory.counter == self.args.total_memory:
                agent.learn()
