import torch as T
import torch as T
import torch.nn.functional as F
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from framework.utils.base import base_policy


class maddpg_policy(base_policy):
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

        actor_dims = []
        for i in range(n_agents):
            actor_dims.append(input_shape[1])
        critic_dims = sum(actor_dims)

        print(action_space)
        self.maddpg_agents = MADDPG(
            actor_dims,
            critic_dims,
            n_agents,
            action_space[0],
            fc1=64,
            fc2=64,
            alpha=0.01,
            beta=0.01,
            chkpt_dir="tmp/maddpg/",
        )

        self.memory = MultiAgentReplayBuffer(
            1000000, critic_dims, actor_dims, action_space[0], n_agents, batch_size=1024
        )

    def action(self, observation, evaluate=False):

        obs = T.tensor(np.array([observation]), dtype=T.float, device="cuda")

        actions = self.maddpg_agents.choose_action(obs, evaluate)
        self.to_remember = (
            observation,
            obs_list_to_state_vector(observation),
            actions,
        )
        # print(actions)
        # actions = [np.argmax(a) for a in actions]
        # print(actions)
        return actions, ([0], 0)

    def store(self, total_steps, obs, rewards, dones):

        self.memory.store_transition(
            self.to_remember[0],
            self.to_remember[1],
            self.to_remember[2],
            rewards,
            obs,
            obs_list_to_state_vector(obs),
            dones,
        )

        if total_steps % 100 == 0:
            self.maddpg_agents.learn(self.memory)


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state


class CriticNetwork(nn.Module):
    def __init__(
        self, beta, input_dims, fc1_dims, fc2_dims, n_agents, n_actions, name, chkpt_dir
    ):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc2_dims)
        self.fc2 = nn.Linear(fc2_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, 1)
        # self.activation = T.nn.SELU()
        self.activation = T.nn.ReLU()
        # self.activation = T.nn.Tanh()

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.to(self.device)

    def forward(self, state, action):
        x = T.cat([state, action], dim=1)
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        q = self.pi(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    def __init__(
        self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name, chkpt_dir
    ):
        super(ActorNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, fc1_dims)
        self.pi = nn.Linear(fc1_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        # self.activation = T.nn.SELU()
        self.activation = T.nn.ReLU()
        # self.activation = T.nn.Tanh()

        self.to(self.device)

    def forward(self, state):
        x = state
        x = self.activation(self.fc1(x))
        # x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        pi = self.pi(x)
        # pi = T.softmax(pi, dim=1)

        return pi

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class Agent:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_actions,
        n_agents,
        agent_idx,
        chkpt_dir,
        alpha=0.01,
        beta=0.01,
        fc1=64,
        fc2=64,
        gamma=0.95,
        tau=0.01,
    ):
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = "agent_%s" % agent_idx
        self.actor = ActorNetwork(
            alpha,
            actor_dims,
            fc1,
            fc1,
            n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + "_actor",
        )
        self.critic = CriticNetwork(
            beta,
            critic_dims,
            fc2,
            fc2,
            n_agents,
            n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + "_critic",
        )
        self.target_actor = ActorNetwork(
            alpha,
            actor_dims,
            fc1,
            fc2,
            n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + "_target_actor",
        )
        self.target_critic = CriticNetwork(
            beta,
            critic_dims,
            fc1,
            fc2,
            n_agents,
            n_actions,
            chkpt_dir=chkpt_dir,
            name=self.agent_name + "_target_critic",
        )

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        # print(observation)
        # print(T.tensor(observation, dtype=T.float))
        actions = self.actor.forward(observation)
        noise = T.rand(self.n_actions).to(self.actor.device)
        action = actions + noise * int((not evaluate))

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_state_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()


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
        # this introduces a bug: if we fill up the memory capacity and then
        # zero out our actor memory, the critic will still have memories to access
        # while the actor will have nothing but zeros to sample. Obviously
        # not what we intend.
        # In reality, there's no problem with just using the same index
        # for both the actor and critic states. I'm not sure why I thought
        # this was necessary in the first place. Sorry for the confusion!

        # if self.mem_cntr % self.mem_size == 0 and self.mem_cntr > 0:
        #    self.init_actor_memory()

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


class MADDPG:
    def __init__(
        self,
        actor_dims,
        critic_dims,
        n_agents,
        n_actions,
        scenario="simple",
        alpha=0.01,
        beta=0.01,
        fc1=64,
        fc2=64,
        gamma=0.99,
        tau=0.01,
        chkpt_dir="tmp/maddpg/",
    ):
        self.agents = []
        self.n_agents = n_agents
        self.n_actions = n_actions
        chkpt_dir += scenario
        for agent_idx in range(self.n_agents):
            self.agents.append(
                Agent(
                    actor_dims[agent_idx],
                    critic_dims,
                    n_actions,
                    n_agents,
                    agent_idx,
                    alpha=alpha,
                    beta=beta,
                    chkpt_dir=chkpt_dir,
                )
            )

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        for agent in self.agents:
            agent.save_models()

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        for agent in self.agents:
            agent.load_models()

    def choose_action(self, raw_obs, evaluate=False):
        actions = []
        for agent_idx, agent in enumerate(self.agents):
            # print(raw_obs.shape)
            action = agent.choose_action(raw_obs[:, agent_idx, :], evaluate)
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

        device = self.agents[0].actor.device

        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards, dtype=T.float).to(device)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], dtype=T.float).to(device)

            new_pi = agent.target_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actor_states[agent_idx], dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in enumerate(self.agents):
            agent.actor.optimizer.zero_grad()

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.target_critic.forward(states_, new_actions).flatten()
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
