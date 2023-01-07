from cmath import tanh
import numpy as np
import torch as T
from torch import nn, no_grad
from torch import optim
import os
import torch
from torch.nn import Softmax
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal
from tqdm import tqdm
from torch.nn import MSELoss
from torch.nn import HuberLoss
from dotmap import DotMap
from framework.utils.base import base_policy, Args
from pettingzoo import ParallelEnv
from framework.model_arc import ACNetwork
from framework.utils.base import base_policy
from torch.utils.tensorboard import SummaryWriter


class ppo_shared_use_future(base_policy):
    def __init__(self, args, writer):
        self.args = args

        self.agent = Agent(args, writer)

        self.n_agents = args.n_agents
        self.idx_starts = np.array([i * args.n_agents for i in range(0, args.num_envs)])

    def save_agents(self, PATH):
        self.agent.save(PATH)

    def load_agents(self, PATH):
        self.agent.load(PATH)

    def get_critic_obs(self, observations):
        val_obs_ = T.tensor(observations).reshape(self.args.num_envs, self.n_agents, -1)
        val_obs = T.zeros(
            (self.args.num_envs * self.n_agents, self.args.obs_space[0] * self.n_agents)
        )

        for i in range(self.args.num_envs * self.n_agents):
            an = i % self.n_agents
            av = i // self.n_agents
            full_obs = []
            for k in range(self.n_agents):
                full_obs.append(val_obs_[av][(k + an) % self.n_agents])
            val_obs[i] = T.hstack(full_obs)

        val_obs = val_obs.to("cuda")
        return val_obs

    def action(self, observations, new_episode=False, **kwargs):
        with T.no_grad():
            if new_episode:
                self.agent.ppo.eval()
                self.agent.ppo.init_hidden(observations.shape[0])
            # print(self.agent.ppo.actor_hidden[0][0])
            self.to_remember = []
            val_obs = self.get_critic_obs(observations)
            obs = T.tensor(observations, dtype=T.float, device="cuda")

            (action_p, actions, value) = self.agent.choose_action(obs, val_obs)
            actions = actions.squeeze()
            action_p = action_p.squeeze()
            value = value.squeeze()

            self.to_remember = (obs, val_obs, action_p, actions, value)

            # print(actions.squeeze())

            return actions.numpy()

    def action_evaluate(self, observations, new_episode):
        obs_batch = T.tensor(observations, dtype=T.float, device="cuda")
        if new_episode:
            self.agent.ppo.eval()
            self.agent.ppo.init_hidden(observations.shape[0])
        actions = self.agent.choose_action_evaluate(obs_batch)
        # actions = actions.squeeze()
        # print(actions)

        return actions[0].numpy()

    def store(self, total_steps, obs, rewards, dones):

        done = T.Tensor(dones)
        reward = T.tensor(rewards)
        self.agent.remember(
            self.to_remember[0],  # obs
            self.to_remember[1],  # valobs
            self.to_remember[2],  # action_p
            self.to_remember[3],  # actions
            self.to_remember[4],  # value
            reward,
            done,
        )

        if (self.agent.memory.counter) == self.args.episode_len:
            self.agent.memory.counter = 0
            self.agent.memory.cn += 1
        if self.agent.memory.cn == self.args.learn_n:
            self.agent.learn(total_steps)


# fmt:off
class PPOTrainer:
    def __init__(self, args, num_steps, num_envs, obs_space, gamma, gae_lambda):
        self.args = args
        self.num_steps = args.episode_len
        self.num_envs = num_envs
        self.batch_size = args.batch_size
        self.obs_space = obs_space
        self.gae = True
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear_memory()

    def create_training_data(self):
        b_obs = self.obs.to("cuda")
        b_val_obs = self.valobs.to("cuda")
        b_logprobs = self.logprobs.to("cuda")
        b_actions = self.actions.to("cuda")
        b_advantages = self.advantages.to("cuda")
        b_returns = self.returns.to("cuda")
        b_values = self.values.to("cuda")
        
        return b_obs, b_val_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values

    def store_memory(self, observations, val_observations, logprobs,action,vals,reward,done):
        c = self.counter
        start = self.cn * self.num_envs * self.args.n_agents
        end = (self.cn+1) * self.num_envs * self.args.n_agents
        self.obs[c,start:end] = observations
        self.valobs[c,start:end] = val_observations

        self.logprobs[c,start:end] = logprobs
        self.actions[c,start:end] = action
        self.values[c,start:end] = vals
        self.rewards[c,start:end] = reward
        self.dones[c,start:end] = done

        self.counter += 1

    def calculate_returns(self):
        with torch.no_grad():
            if self.gae:
                advantages = torch.zeros_like(self.rewards)
                lastgaelam = 0
                for t in reversed(range(self.num_steps-1)):
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                    delta = (
                        self.rewards[t]
                        + self.gamma * nextvalues * nextnonterminal
                        - self.values[t]
                    )
                    advantages[t] = lastgaelam = (
                        delta
                        + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                    )
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards)
                for t in reversed(range(self.num_steps)):
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = returns[t + 1]
                    returns[t] = (
                        self.rewards[t] + self.gamma * nextnonterminal * next_return
                    )
                advantages = returns - self.values

        self.returns = returns
        self.advantages = advantages

    def clear_memory(self):
        space = (self.num_steps, self.num_envs * self.args.n_agents * self.args.learn_n)

        self.obs = T.zeros(space + self.obs_space)
        self.valobs = T.zeros(space + (self.obs_space[0]*self.args.n_agents,))
        self.logprobs = T.zeros(space)
        self.actions = T.zeros(space)
        self.values = T.zeros(space)
        self.rewards = T.zeros(space)
        self.dones = T.zeros(space)
        self.counter = 0
        self.cn = 0
# fmt:on


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# class NNN(nn.Module):
#     def __init__(self, obs_shape, actors, action_space, hidden_size):
#         super(NNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.gru_layers = 1
#         self.inp_hid_size = np.array(obs_shape).prod()
#         self.actors = actors

#         act_fn = nn.ReLU

#         layer_filters = 128

#         val_obs_repr_size = 64

#         self.val_concater = nn.Sequential(
#             layer_init(nn.Linear(self.inp_hid_size, val_obs_repr_size)),
#             nn.Tanh(),
#         )

#         self.gru_critic = nn.GRU(
#             val_obs_repr_size, hidden_size, self.gru_layers, batch_first=False
#         )
#         self.gru_actor = nn.GRU(
#             self.inp_hid_size, hidden_size, self.gru_layers, batch_first=False
#         )

#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, 1), std=1.0),
#         )
#         self.actor = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, action_space), std=0.01),
#         )

#     def init_hidden(self, batch_size=1):
#         self.actor_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )
#         self.critic_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )

#     def get_value(self, val_x):
#         out = 0
#         for i in range(self.actors):
#             end = self.inp_hid_size * (i + 1)
#             start = self.inp_hid_size * i
#             base_x = val_x[:, :, start:end]
#             self_or_not = torch.zeros(base_x.size(0), base_x.size(1), 1).cuda() + (
#                 i != 0
#             )
#             self_or_not = torch.concat([base_x, self_or_not], 2)
#             out += self.val_concater(base_x)

#         out, self.critic_hidden = self.gru_critic(out, self.critic_hidden)
#         value = self.critic(out)
#         return value

#     def get_action(self, x):
#         out, self.actor_hidden = self.gru_actor(x, self.actor_hidden)
#         logits = self.actor(out)
#         probs = Categorical(logits=logits)
#         action = probs.sample()

#         return action, probs

#     def get_action_and_value(self, x, val_x, action_=None):
#         action, probs = self.get_action(x)
#         value = self.get_value(val_x)

#         prob = (
#             probs.log_prob(action_) if action_ is not None else probs.log_prob(action)
#         )

#         return (action, prob, probs.entropy(), value)


# class NNN(nn.Module):
#     def __init__(self, obs_shape, actors, action_space, hidden_size):
#         super(NNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.gru_layers = 1
#         self.inp_hid_size = np.array(obs_shape).prod()

#         act_fn = nn.ReLU
#         self.actors = actors

#         layer_filters = 128

#         self.base_info = 2 + 2 + 15  # self, obj, landmarks
#         self.agent_info = 2 + 3
#         self.extra_info = 30
#         self.critic_extra_info = 128

#         self.agent_concater = nn.Sequential(
#             layer_init(nn.Linear(self.agent_info, self.extra_info)),
#             nn.Tanh(),
#         )

#         self.val_concater = nn.Sequential(
#             layer_init(
#                 nn.Linear(self.base_info + self.extra_info, self.critic_extra_info)
#             ),
#             nn.Tanh(),
#         )

#         self.gru_critic = nn.GRU(
#             self.critic_extra_info, hidden_size, self.gru_layers, batch_first=False
#         )
#         self.gru_actor = nn.GRU(
#             self.base_info + self.extra_info,
#             hidden_size,
#             self.gru_layers,
#             batch_first=False,
#         )

#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, 1), std=1.0),
#         )
#         self.actor = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, action_space), std=0.01),
#         )

#     def init_hidden(self, batch_size=1):
#         self.actor_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )
#         self.critic_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )

#     def get_hidden_repr(self, x, agent_info):
#         vocab_size = 20
#         extra_info = 3
#         inp = torch.zeros(vocab_size + extra_info)
#         inp[x + extra_info] = 1
#         inp[:extra_info] = agent_info
#         return self.agent_concater(inp)

#     def get_value(self, val_x):
#         out = 0
#         for i in range(self.actors):
#             end = self.inp_hid_size * (i + 1)
#             start = self.inp_hid_size * i
#             base_x = val_x[:, :, start:end]

#             out_temp = 0
#             for j in range(self.actors - 1):
#                 start = (self.base_info) + self.agent_info * j
#                 end = (self.base_info) + self.agent_info * (j + 1)
#                 out_temp += self.agent_concater(base_x[:, :, start:end])
#             out_temp = out_temp / (self.actors - 1)
#             out_temp = torch.concat([base_x[:, :, 0 : self.base_info], out_temp], 2)
#             out += self.val_concater(out_temp)

#         out = out / (self.actors - 1)
#         out, self.critic_hidden = self.gru_critic(out, self.critic_hidden)
#         value = self.critic(out)
#         return value

#     def get_action(self, x):
#         out = 0
#         for i in range(self.actors - 1):
#             start = (self.base_info) + self.agent_info * i
#             end = (self.base_info) + self.agent_info * (i + 1)
#             out += self.agent_concater(x[:, :, start:end])
#         out = out / (self.actors - 1)
#         out = torch.concat([x[:, :, 0 : self.base_info], out], 2)

#         out, self.actor_hidden = self.gru_actor(out, self.actor_hidden)
#         logits = self.actor(out)
#         probs = Categorical(logits=logits)
#         action = probs.sample()

#         return action, probs

#     def get_action_and_value(self, x, val_x, action_=None):
#         action, probs = self.get_action(x)
#         value = self.get_value(val_x)

#         prob = (
#             probs.log_prob(action_) if action_ is not None else probs.log_prob(action)
#         )

#         return (action, prob, probs.entropy(), value)


# class NNN(nn.Module):
#     def __init__(self, obs_shape, actors, action_space, hidden_size):
#         super(NNN, self).__init__()

#         self.hidden_size = hidden_size
#         self.gru_layers = 1
#         inp_hid_size = np.array(obs_shape).prod()

#         act_fn = nn.ReLU

#         layer_filters = 256

#         self.gru_critic = nn.GRU(
#             inp_hid_size * actors, hidden_size, self.gru_layers, batch_first=False
#         )
#         self.gru_actor = nn.GRU(
#             inp_hid_size, hidden_size, self.gru_layers, batch_first=False
#         )

#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, 1), std=1.0),
#         )

#         self.actor = nn.Sequential(
#             layer_init(nn.Linear(hidden_size, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#         )

#         self.action = nn.Sequential(
#             layer_init(nn.Linear(layer_filters + inp_hid_size, layer_filters)),
#             layer_init(nn.Linear(layer_filters, action_space), std=0.01),
#         )

#         self.future = nn.Sequential(
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, layer_filters)),
#             act_fn(),
#             layer_init(nn.Linear(layer_filters, inp_hid_size), std=0.01),
#         )

#     def init_hidden(self, batch_size=1):
#         self.actor_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )
#         self.critic_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
#             "cuda"
#         )

#     def get_futures(self, x, n):
#         out = x
#         futures = []
#         for i in range(n):
#             out, self.actor_hidden = self.gru_actor(out, self.actor_hidden)
#             out = self.actor(out)
#             out = self.future(out)
#             futures.append(out)

#         return futures

#     def get_value(self, val_x):
#         out, self.critic_hidden = self.gru_critic(val_x, self.critic_hidden)
#         value = self.critic(out)
#         return value

#     def get_action(self, x):
#         out, self.actor_hidden = self.gru_actor(x, self.actor_hidden)
#         out = self.actor(out)
#         future = self.future(out)
#         out = torch.concat([out, future], dim=2)
#         logits = self.action(out)
#         probs = Categorical(logits=logits)
#         action = probs.sample()

#         return action, probs, future

#     def get_action_and_value(self, x, val_x, action_=None):
#         action, probs, future = self.get_action(x)
#         value = self.get_value(val_x)

#         prob = (
#             probs.log_prob(action_) if action_ is not None else probs.log_prob(action)
#         )

#         return (action, prob, probs.entropy(), value, future)


class NNN(nn.Module):
    def __init__(self, obs_shape, actors, action_space, hidden_size):
        super(NNN, self).__init__()

        self.hidden_size = hidden_size
        self.gru_layers = 1
        inp_hid_size = np.array(obs_shape).prod()

        act_fn = nn.ReLU

        layer_filters = 256

        self.gru = nn.GRU(inp_hid_size, hidden_size, self.gru_layers, batch_first=False)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(10, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, 1)),
        )

        self.common = nn.Sequential(
            layer_init(nn.Linear(hidden_size, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, 10)),
            act_fn(),
        )

        self.action = nn.Sequential(
            layer_init(nn.Linear(10 + inp_hid_size, layer_filters)),
            layer_init(nn.Linear(layer_filters, action_space), std=0.01),
        )

        self.future = nn.Sequential(
            layer_init(nn.Linear(10, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, layer_filters)),
            act_fn(),
            layer_init(nn.Linear(layer_filters, inp_hid_size), std=0.01),
        )

    def init_hidden(self, batch_size=1):
        self.actor_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
            "cuda"
        )
        self.critic_hidden = T.zeros(self.gru_layers, batch_size, self.hidden_size).to(
            "cuda"
        )

    def get_hidden(self, x):
        out, self.actor_hidden = self.gru(x, self.actor_hidden)
        out = self.common(out)
        return out

    def get_futures(self, x, n):
        out = x
        futures = []
        for i in range(n):
            out, self.actor_hidden = self.gru_actor(out, self.actor_hidden)
            out = self.actor(out)
            out = self.future(out)
            futures.append(out)

        return futures

    # def get_value(self, val_x):
    #     out, self.critic_hidden = self.gru(val_x, self.critic_hidden)
    #     value = self.critic(out)
    #     return value

    def get_action(self, x, value=False):
        out, self.actor_hidden = self.gru(x, self.actor_hidden)
        out = self.common(out)
        future = self.future(out)

        if value:
            val = self.critic(out)
        out = torch.concat([out, future], dim=2)
        logits = self.action(out)
        probs = Categorical(logits=logits)
        action = probs.sample()

        if value:
            return action, probs, future, val

        return action, probs, future

    def get_action_and_value(self, x, val_x, action_=None):
        action, probs, future, value = self.get_action(x, value=True)
        # value = self.get_value(val_x)

        prob = (
            probs.log_prob(action_) if action_ is not None else probs.log_prob(action)
        )

        return (action, prob, probs.entropy(), value, future)


class Agent:
    def __init__(
        self,
        args,
        writer: SummaryWriter,
    ):
        self.args = args

        self.writer = writer
        action_space = args.action_space
        self.ppo = NNN(args.obs_space, args.n_agents, action_space, args.hidden_size)
        print(self.ppo)
        self.memory = PPOTrainer(
            args,
            args.episode_len,
            args.num_envs,
            args.obs_space,
            args.gamma,
            args.gae_lambda,
        )
        self.ppo.to(args.device)
        self.optimizer = optim.Adam(
            self.ppo.parameters(), lr=args.learning_rate, eps=1e-5
        )

    # fmt:off
    def remember(self, observations, val_obs, action_p, action, vals, reward, done):
        self.memory.store_memory(observations, val_obs, action_p, action, vals, reward, done)

    def save(self, PATH):
        torch.save(self.ppo.state_dict(), PATH+f"/agent_{0}")
        print(f"Save model agent_{0} at {PATH}")
    
    def load(self, PATH):
        fname = PATH+f"/agent_{0}"
        if os.path.isfile(fname):
            self.ppo.load_state_dict(torch.load(fname), strict=False)
            print(f"Load model agent_{0} at {fname}")
        else:
            fname = PATH+f"/agent_{0}"
            self.ppo.load_state_dict(torch.load(fname), strict=False)
            print(f"Load model agent_{0} at {fname}")
    # fmt:on
    def choose_action_evaluate(self, obs):
        with torch.no_grad():
            obs_space = np.array(self.args.obs_space).prod()
            obs = obs.reshape(1, -1, obs_space)
            action, _, _ = self.ppo.get_action(obs)
            return action.cpu()

    def choose_action(self, observations, val_obs):
        with torch.no_grad():
            obs_space = np.array(self.args.obs_space).prod()
            observations = observations.reshape(1, -1, obs_space)
            val_obs = val_obs.reshape(1, -1, obs_space * self.args.n_agents)

            (
                action,
                probs,
                _,
                value,
                _,
            ) = self.ppo.get_action_and_value(observations, val_obs)

            return (
                probs.cpu(),
                action.cpu(),
                value.cpu(),
            )

    def learn(self, global_step):
        self.ppo.train()

        args = self.args
        self.memory.calculate_returns()
        clipfracs = []

        (
            b_obs,
            b_val_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
        ) = self.memory.create_training_data()

        total_pg_loss = 0
        total_v_loss = 0
        mseloss = torch.nn.MSELoss()

        for epoch in range(args.update_epochs):
            self.ppo.init_hidden(b_obs.shape[1])
            self.optimizer.zero_grad()

            (_, newlogprob, entropy, newvalue, future) = self.ppo.get_action_and_value(
                b_obs, b_val_obs, b_actions.long()
            )
            newvalue = newvalue.squeeze()

            logratio = newlogprob - b_logprobs
            ratio = logratio.exp()

            # if True: #Future Loss
            # n = 3
            # futures = self.ppo.get_futures(b_obs, n)
            floss = 0
            floss = mseloss(b_obs[1:], future[:-1])
            # for i in range(n):
            #     floss += mseloss(b_obs[i + 1 :], futures[i][: -i - 1])

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                # old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                # if approx_kl > 0.02:
                #     print("too large kl", epoch)
                #     break
                clipfracs += [
                    ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                ]

            if args.norm_adv:
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

            pg_loss1 = -b_advantages * ratio
            pg_loss2 = -b_advantages * torch.clamp(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            total_pg_loss += pg_loss
            if args.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns) ** 2
                v_clipped = b_values + torch.clamp(
                    newvalue - b_values,
                    -args.clip_coef,
                    args.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
            total_v_loss += v_loss

            entropy_loss = entropy.mean()
            loss = (
                pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + floss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.ppo.parameters(), args.max_grad_norm)
            self.optimizer.step()

        y_pred, y_true = (
            b_values.reshape(-1).cpu().numpy(),
            b_returns.reshape(-1).cpu().numpy(),
        )
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        div_term = args.update_epochs

        self.writer.add_scalar(
            f"losses/value_loss", total_v_loss.item() / div_term, global_step
        )
        self.writer.add_scalar(
            f"losses/policy_loss", total_pg_loss.item() / div_term, global_step
        )
        self.writer.add_scalar(f"losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar(f"losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar(f"losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar(f"losses/explained_variance", explained_var, global_step)
        self.writer.add_scalar(f"losses/Floss", floss, global_step)

        self.memory.clear_memory()
