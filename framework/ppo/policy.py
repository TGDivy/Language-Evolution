from framework.utils.base import base_policy
from framework.ppo.ppo import Agent
import torch as T


class ppo_policy(base_policy):
    def __init__(self, args, n_agents, input_shape, num_layers, num_filters) -> None:
        self.args = args

        self.agents = []
        for _ in range(n_agents):
            agent = Agent(args, input_shape, num_layers, num_filters)
            self.agents.append(agent)

        self.to_remember = {}

        # self.added_graph = False
        # if not self.added_graph:
        #     self.logger.add_graph(self.agent1.ppo, obs_batch)
        #     self.added_graph = True

    def action(self, observations):

        actions = []
        for i, agent in enumerate(self.agents):

            obs = observations[i]

            obs_batch = T.tensor([obs], dtype=T.float, device="cuda")

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
            actions.append(move_action.item())

        return actions

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
