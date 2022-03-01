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


class ppo_policy3_shared(base_policy):
    def __init__(self, args, writer):
        self.args = args

        self.agent = Agent(args, writer)
        self.to_remember = []

    def add_logger(self, logger: SummaryWriter):
        self.logger = logger

    def action(self, observations, **kwargs):

        obs_batch = T.tensor(observations, dtype=T.float, device="cuda")

        (
            action_p,
            action,
            value,
        ) = self.agent.choose_action(obs_batch)
        value = T.squeeze(value)

        self.to_remember = (
            obs_batch,
            action_p,
            action,
            value,
        )

        return action, (value, action_p)

    def store(self, total_steps, obs, rewards, dones):

        dones = T.Tensor(dones)
        self.agent.remember(
            self.to_remember[0],
            self.to_remember[1],
            self.to_remember[2],
            self.to_remember[3],
            T.tensor(rewards),
            dones,
        )

        if self.agent.memory.counter == self.args.num_steps:
            self.agent.learn(total_steps)


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
        b_inds = np.arange(self.batch_size)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_inds

    def store_memory(self, observations, logprobs, action, vals, reward, done):
        self.obs[self.counter] = observations[0]
        self.logprobs[self.counter] = logprobs
        self.actions[self.counter] = action
        self.values[self.counter] = vals
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
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
        self.counter = 0


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class NNN(nn.Module):
    def __init__(self, obs_shape, action_space):
        super(NNN, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(obs_shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_space), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent:
    def __init__(
        self,
        args,
        writer,
    ):
        self.args = args

        self.writer = writer
        action_space = 50
        self.ppo = NNN(args.obs_space, action_space)
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

    def remember(self, observations, action_p, action, vals, reward, done):
        self.memory.store_memory(observations, action_p, action, vals, reward, done)

    def save_models(self):
        print("... saving models ...")
        self.ppo.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.ppo.load_checkpoint()

    def choose_action(self, observations):
        with torch.no_grad():
            action, probs, entropy, value = self.ppo.get_action_and_value(observations)

        return probs.cpu(), action.cpu(), value.cpu()

    def learn(self, global_step):
        args = self.args
        self.memory.calculate_returns()
        (
            b_obs,
            b_logprobs,
            b_actions,
            b_advantages,
            b_returns,
            b_values,
            b_inds,
        ) = self.memory.create_training_data()

        clipfracs = []

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.ppo.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
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
