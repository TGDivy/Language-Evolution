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


class MADDPG:
    def __init__(
        self,
        n_agents,
        n_actions,
        action_space,
        input_shape,
        num_layers,
        num_filters,
        device,
        lr,
    ):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions

        for agent_idx in range(self.n_agents):
            self.agents.append(
                Agent(
                    agent_idx,
                    action_space,
                    input_shape,
                    num_layers,
                    num_filters,
                    device,
                    lr,
                )
            )

    def choose_action(self, raw_obs):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            action = agent.choose_action(raw_obs[agent_idx])
            actions.append(action)
        return actions

    def learn(self, memory):
        if not memory.ready():
            return

        (
            actor_states,
            states,
            actions,
            rewards,
            actor_new_states,
            states_,
            dones,
        ) = memory.sample_buffer()

        device = self.agents[0].AC.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)
            new_action, val = agent.targetAC.forward(new_states)
            all_agents_new_actions.append(new_action)

            states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            action, _ = agent.targetAC.forward(states)
            all_agents_new_mu_actions.append(action)
            old_agents_actions.append(actions[agent_idx])

            agent.AC.optimizer.zero_grad()

        new_actions = T.cat(all_agents_new_actions, dim=1)
        mu = T.cat(all_agents_new_mu_actions, dim=1)
        old_actions = T.cat(old_agents_actions, dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.targetAC.forward(states_, new_actions).flatten()
            critic_value_[dones[:, 0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:, agent_idx] + agent.gamma * critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            actor_loss.backward(retain_graph=True)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.step()
            agent.update_network_parameters()


class MultiAgentReplayBuffer:
    def __init__(
        self, max_size, critic_dims, actor_dims, n_actions, n_agents, batch_size
    ):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.n_agents = n_agents
        self.actor_dims = actor_dims
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.state_memory = np.zeros((self.mem_size, critic_dims))
        self.new_state_memory = np.zeros((self.mem_size, critic_dims))
        self.reward_memory = np.zeros((self.mem_size, n_agents))
        self.terminal_memory = np.zeros((self.mem_size, n_agents), dtype=bool)

        self.init_actor_memory()

    def init_actor_memory(self):
        self.actor_state_memory = []
        self.actor_new_state_memory = []
        self.actor_action_memory = []

        for i in range(self.n_agents):
            self.actor_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i]))
            )
            self.actor_new_state_memory.append(
                np.zeros((self.mem_size, self.actor_dims[i]))
            )
            self.actor_action_memory.append(np.zeros((self.mem_size, self.n_actions)))

    def store_transition(self, raw_obs, state, action, reward, raw_obs_, state_, done):

        index = self.mem_cntr % self.mem_size

        for agent_idx in range(self.n_agents):
            self.actor_state_memory[agent_idx][index] = raw_obs[agent_idx]
            self.actor_new_state_memory[agent_idx][index] = raw_obs_[agent_idx]
            self.actor_action_memory[agent_idx][index] = action[agent_idx]

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        states = self.state_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        actor_states = []
        actor_new_states = []
        actions = []
        for agent_idx in range(self.n_agents):
            actor_states.append(self.actor_state_memory[agent_idx][batch])
            actor_new_states.append(self.actor_new_state_memory[agent_idx][batch])
            actions.append(self.actor_action_memory[agent_idx][batch])

        return (
            actor_states,
            states,
            actions,
            rewards,
            actor_new_states,
            states_,
            terminal,
        )

    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True


class Agent:
    def __init__(
        self,
        agent_idx,
        action_space,
        input_shape,
        num_layers,
        num_filters,
        device,
        lr,
    ):
        self.gamma = 0.95
        self.tau = 0.01
        self.n_actions = action_space
        self.agent_name = f"agent_{agent_idx}"
        self.AC = ACNetwork(
            action_space, input_shape, num_layers, num_filters, device, lr
        )
        self.targetAC = ACNetwork(
            action_space, input_shape, num_layers, num_filters, device, lr
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        action, value = self.AC(observation)
        # state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # actions = self.actor.forward(state)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = action + noise

        return action

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_AC_params = self.targetAC.named_parameters()
        AC_params = self.AC.named_parameters()

        target_AC_params = dict(target_AC_params)
        AC_params = dict(AC_params)
        for name in AC_params:
            AC_params[name] = (
                tau * AC_params[name].clone()
                + (1 - tau) * target_AC_params[name].clone()
            )

        self.targetAC.load_state_dict(AC_params)

    # def save_models(self):
    #     self.actor.save_checkpoint()
    #     self.target_actor.save_checkpoint()
    #     self.critic.save_checkpoint()
    #     self.target_critic.save_checkpoint()

    # def load_models(self):
    #     self.actor.load_checkpoint()
    #     self.target_actor.load_checkpoint()
    #     self.critic.load_checkpoint()
    #     self.target_critic.load_checkpoint()
