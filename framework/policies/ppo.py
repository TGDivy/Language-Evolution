import numpy as np
import torch as T
from torch import nn
from torch import optim
import os
from torch.nn import Softmax
from torch.distributions.categorical import Categorical
from torch.nn import MSELoss
from torch.nn import HuberLoss
from dotmap import DotMap
from framework.utils.base import base_policy, Args
from pettingzoo import ParallelEnv
from framework.model_arc import ACNetwork
from framework.utils.base import base_policy
from torch.utils.tensorboard import SummaryWriter


class ppo_policy(base_policy):
    def __init__(
        self,
        args,
        n_agents,
        action_space,
        input_shape,
        num_layers,
        num_filters,
        lr,
        device,
    ) -> None:
        self.args = args

        self.agents = []
        for _ in range(n_agents):
            agent = Agent(
                args, action_space, input_shape, num_layers, num_filters, device, lr
            )
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
                action_p,
                action,
                value,
            ) = agent.choose_action(obs_batch)

            self.to_remember[i] = (
                obs_batch,
                action_p,
                action,
                value,
            )
            actions.append(action.item())

        return actions, (value, action_p)

    def store(self, total_steps, obs, rewards, dones):

        for i, agent in enumerate(self.agents):
            agent.remember(
                self.to_remember[i][0],
                self.to_remember[i][1],
                self.to_remember[i][2],
                self.to_remember[i][3],
                T.tensor(rewards[i]),
                dones[i],
            )

            if agent.memory.counter == self.args.total_memory:
                agent.learn()


class PPOMemory:
    def __init__(self, input_size, batch_size, total_memory):
        self.batch_size = batch_size
        self.total_memory = total_memory
        self.input_size = input_size[-1]
        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.observations)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]
        dic = {
            "observations": self.observations,
            "action": (
                self.action_p,
                self.action,
            ),
            "rewards": (
                self.vals,
                self.rewards,
                self.dones,
            ),
            "batches": T.tensor(batches),
        }

        return dic

    def store_memory(self, observations, action_p, action, vals, reward, done):
        self.observations[self.counter] = observations[0]

        self.action_p[self.counter] = action_p
        self.action[self.counter] = action
        self.vals[self.counter] = vals
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
        self.counter += 1

    def clear_memory(self):
        self.observations = T.zeros(self.total_memory, self.input_size)
        self.action_p = T.zeros(self.total_memory, 1)
        self.action = T.zeros(self.total_memory, 1)
        self.vals = T.zeros(self.total_memory)
        self.rewards = T.zeros(self.total_memory)
        self.dones = T.zeros(self.total_memory)
        self.counter = 0


class Agent:
    def __init__(
        self, args, action_space, input_shape, num_layers, num_filters, device, lr
    ):
        self.gamma = args.gamma
        self.policy_clip = args.policy_clip
        self.n_epochs = args.n_epochs
        self.gae_lambda = args.gae_lambda
        self.entropy = args.entropy

        self.ppo = ACNetwork(
            action_space, input_shape, num_layers, num_filters, device, lr
        )
        self.memory = PPOMemory(input_shape, args.batch_size, args.total_memory)
        # self.huber = HuberLoss(reduction="mean", delta=1.0)
        self.mse = MSELoss(reduction="mean")
        self.device = self.ppo.device

    def remember(self, observations, action_p, action, vals, reward, done):
        self.memory.store_memory(observations, action_p, action, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.ppo.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.ppo.load_checkpoint()

    def choose_action(self, observations):

        action, value = self.ppo(observations)

        action_dist = Categorical(action)
        action = action_dist.sample()
        action_p = action_dist.log_prob(action)

        return action_p, action, value

    def advantage(self, reward_arr, values, dones_arr):
        advantage = np.zeros(len(reward_arr), dtype=np.float16)
        for t in range(len(reward_arr) - 1):
            discount = 0.99
            a_t = 0
            for k in range(t, len(reward_arr) - 1):
                a_t += discount * (
                    reward_arr[k]
                    + self.gamma * values[k + 1] * (1 - int(dones_arr[k]))
                    - values[k]
                )
                discount *= self.gamma * self.gae_lambda
            advantage[t] = a_t
        advantage = T.tensor(advantage).to(self.device)
        return advantage

    def learn(self):
        dic = self.memory.generate_batches()
        vals_arr, reward_arr, dones_arr = dic["rewards"]
        (action_p_arr, action_arr) = dic["action"]
        action_p_arr = (
            action_p_arr.clone().detach().requires_grad_(True).to(self.device)
        )
        action_arr = action_arr.clone().detach().requires_grad_(True).to(self.device)
        observations_arr = (
            dic["observations"].clone().detach().requires_grad_(True).to(self.device)
        )

        values = vals_arr.clone().detach().requires_grad_(True).to(self.device)

        for _ in range(self.n_epochs):
            n_states = len(observations_arr)
            batch_start = np.arange(0, n_states, self.memory.batch_size)
            indices = np.arange(n_states, dtype=np.int64)
            np.random.shuffle(indices)
            batches = [indices[i : i + self.memory.batch_size] for i in batch_start]

            advantage = self.advantage(reward_arr, values, dones_arr)

            for batch in batches:
                # print("LEARNINGGG !!")
                total_loss = 0
                #### game inputs/states
                observations = observations_arr[batch]
                adv = advantage[batch]
                #### action
                actions = action_arr[batch]
                old_probs = action_p_arr[batch]
                #### Actor Critic Run
                action, critic_value = self.ppo(observations)

                #### Critic Loss
                critic_value = T.squeeze(critic_value)
                returns = adv - values[batch]

                critic_loss = self.mse(critic_value, returns)
                total_loss += critic_loss * 0.5

                #### Unit, City Actors Loss
                dist = Categorical(action)
                new_probs = dist.log_prob(actions)
                prob_ratio = (new_probs - old_probs).exp()

                weighted_probs = prob_ratio * adv.reshape(-1, 1)
                weighted_clipped_probs = T.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * adv.reshape(-1, 1)

                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                total_loss += actor_loss
                total_loss += self.entropy * T.mean(dist.entropy())

                total_loss.backward()
                self.ppo.optimizer.step()
                self.ppo.optimizer.zero_grad()

        self.memory.clear_memory()
