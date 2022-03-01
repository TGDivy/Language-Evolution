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

from torch.nn import MSELoss
from torch.nn import HuberLoss
from dotmap import DotMap
from framework.utils.base import base_policy, Args
from pettingzoo import ParallelEnv
from framework.model_arc import ACNetwork
from framework.utils.base import base_policy
from torch.utils.tensorboard import SummaryWriter


class ppo_rnn_policy_shared(base_policy):
    def __init__(self, args, writer):
        self.args = args

        self.agent = Agent(args, writer)
        self.to_remember = []

        # fmt:off
        self.critic_hidden = T.zeros(args.num_envs * self.args.n_agents, args.hidden_size, device=args.device)
        self.actor_hidden = T.zeros(args.num_envs * self.args.n_agents, args.hidden_size, device=args.device)
        
        self.critic_hidden_test = T.zeros(self.args.n_agents, args.hidden_size, device=args.device)
        self.actor_hidden_test = T.zeros(self.args.n_agents, args.hidden_size, device=args.device)
        # fmt:on

    def add_logger(self, logger: SummaryWriter):
        self.logger = logger

    def action(self, observations, **kwargs):

        obs_batch = T.tensor(observations, dtype=T.float, device="cuda")

        (
            action_p,
            action,
            value,
            self.actor_hidden,
            self.critic_hidden,
        ) = self.agent.choose_action(obs_batch, self.actor_hidden, self.critic_hidden)
        value = T.squeeze(value)

        self.to_remember = (obs_batch, action_p, action, value)

        return action.numpy(), (value, action_p)

    def action_evaluate(self, observation, new_episode):
        self.critic_hidden_test *= 0 if new_episode else 1
        self.actor_hidden_test *= 0 if new_episode else 1
        obs_batch = T.tensor(observation, dtype=T.float, device="cuda")
        (
            action_p,
            action,
            value,
            self.actor_hidden_test,
            self.critic_hidden_test,
        ) = self.agent.choose_action(
            obs_batch, self.actor_hidden_test, self.critic_hidden_test
        )
        value = T.squeeze(value)

        self.to_remember = (obs_batch, action_p, action, value)

        return action.numpy(), (value, action_p)

    def store(self, total_steps, obs, rewards, dones):

        dones = T.Tensor(dones)
        self.agent.remember(
            self.to_remember[0],
            self.to_remember[1],
            self.to_remember[2],
            self.to_remember[3],
            T.tensor(rewards),
            dones,
            self.critic_hidden,
            self.actor_hidden,
        )
        # fmt:off
        self.critic_hidden *= dones.reshape(self.args.num_envs * self.args.n_agents, 1).to(self.args.device)
        self.actor_hidden *= dones.reshape(self.args.num_envs * self.args.n_agents, 1).to(self.args.device)
        # fmt:on
        if self.agent.memory.counter == self.args.num_steps:
            self.agent.learn(total_steps)


# fmt:off
class PPOTrainer:
    def __init__(self, args, num_steps, num_envs, obs_space, gamma, gae_lambda):
        self.args = args
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.batch_size = num_steps * num_envs
        self.obs_space = obs_space
        self.gae = True
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear_memory()

    def create_training_data(self):
        b_obs = self.obs.reshape((-1,) + self.obs_space).to("cuda")
        b_logprobs = self.logprobs.reshape(-1).to("cuda")
        b_actions = self.actions.reshape((-1,)).to("cuda")
        b_advantages = self.advantages.reshape(-1).to("cuda")
        b_returns = self.returns.reshape(-1).to("cuda")
        b_values = self.values.reshape(-1).to("cuda")
        b_actor_hidden = self.actor_hidden.reshape(-1,self.args.hidden_size).to("cuda")
        b_critic_hidden = self.critic_hidden.reshape(-1,self.args.hidden_size).to("cuda")
        b_inds = np.arange(self.batch_size)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_actor_hidden, b_critic_hidden, b_inds

    def store_memory(self,observations,logprobs,action,vals,reward,done,actor_hidden,critic_hidden):
        self.obs[self.counter] = observations[0]
        self.logprobs[self.counter] = logprobs
        self.actions[self.counter] = action
        self.values[self.counter] = vals
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
        self.actor_hidden[self.counter] = actor_hidden
        self.critic_hidden[self.counter] = critic_hidden
        self.counter += 1

    def calculate_returns(self):
        with torch.no_grad():
            if self.gae:
                advantages = torch.zeros_like(self.rewards)
                lastgaelam = 0
                for t in reversed(range(self.num_steps - 1)):
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
                for t in reversed(range(self.num_steps - 1)):
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    next_return = returns[t + 1]
                    returns[t] = (
                        self.rewards[t] + self.gamma * nextnonterminal * next_return
                    )
                advantages = returns - self.values

        self.returns = returns
        self.advantages = advantages

    def clear_memory(self):
        space = (self.num_steps, self.num_envs * self.args.n_agents)

        self.obs = T.zeros(space + self.obs_space)
        self.logprobs = T.zeros(space)
        self.actions = T.zeros(space)
        self.values = T.zeros(space)
        self.rewards = T.zeros(space)
        self.dones = T.zeros(space)
        self.actor_hidden = T.zeros(space + (self.args.hidden_size,))
        self.critic_hidden = T.zeros(space + (self.args.hidden_size,))
        self.counter = 0
# fmt:on


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NNN(nn.Module):
    def __init__(self, obs_shape, action_space, hidden_size):
        super(NNN, self).__init__()

        inp_hid_size = hidden_size + np.array(obs_shape).prod()

        self.input_to_hidden_actor = nn.Sequential(
            layer_init(nn.Linear(inp_hid_size, hidden_size)), nn.ReLU()
        )
        self.input_to_hidden_critic = nn.Sequential(
            layer_init(nn.Linear(inp_hid_size, hidden_size)), nn.ReLU()
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(inp_hid_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(inp_hid_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, hidden_actor, hidden_critic, action=None):
        # print("*" * 100)
        # print(x, x.shape)
        # print(hidden_actor, hidden_actor.shape)
        inp_hidden_actor = torch.hstack([x, hidden_actor])
        # print(inp_hidden_actor.shape)
        inp_hidden_critic = torch.hstack([x, hidden_critic])

        new_hidden_actor = self.input_to_hidden_actor(inp_hidden_actor)
        new_hidden_critic = self.input_to_hidden_actor(inp_hidden_critic)

        logits = self.actor(inp_hidden_actor)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()

        return (
            action,
            probs.log_prob(action),
            probs.entropy(),
            self.critic(inp_hidden_critic),
            new_hidden_actor,
            new_hidden_critic,
        )


class Agent:
    def __init__(
        self,
        args,
        writer,
    ):
        self.args = args

        self.writer = writer
        action_space = 50
        self.ppo = NNN(args.obs_space, action_space, args.hidden_size)
        self.memory = PPOTrainer(
            args,
            args.num_steps,
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
    def remember(self, observations, action_p, action, vals, reward, done, critic_h, actor_h):
        self.memory.store_memory(observations, action_p, action, vals, reward, done, critic_h, actor_h)
    
    def save_models(self):
        print("... saving models ...")
        self.ppo.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.ppo.load_checkpoint()
    # fmt:on
    def choose_action(self, observations, actor_hidden, critic_hidden):
        with torch.no_grad():
            (
                action,
                probs,
                entropy,
                value,
                new_hidden_actor,
                new_hidden_critic,
            ) = self.ppo.get_action_and_value(observations, actor_hidden, critic_hidden)

        return (
            probs.cpu(),
            action.cpu(),
            value.cpu(),
            new_hidden_actor,
            new_hidden_critic,
        )

    def learn(self, global_step):
        # print("-" * 100)

        args = self.args
        self.memory.calculate_returns()
        (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_actor_hidden,
            b_critic_hidden,
            b_inds,
        ) = self.memory.create_training_data()

        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _, _ = self.ppo.get_action_and_value(
                    b_obs[mb_inds],
                    b_actor_hidden[mb_inds],
                    b_critic_hidden[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ppo.parameters(), args.max_grad_norm)
                self.optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar(
            "charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step
        )
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        # self.writer.add_scalar(
        #     "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        # )

        self.memory.clear_memory()
