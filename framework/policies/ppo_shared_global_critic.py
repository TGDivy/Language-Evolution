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


class ppo_shared_critic(base_policy):
    def __init__(self, args, writer):
        self.args = args

        self.agent = Agent(args, writer)
        self.n_agents = args.n_agents
        self.idx_starts = np.array([i * args.n_agents for i in range(0, args.num_envs)])

        # fmt:off
        self.actor_hidden = T.zeros(self.args.n_agents, args.num_envs, args.hidden_size, device=args.device)
        self.actor_hidden_test = T.zeros(self.args.n_agents, args.hidden_size, device=args.device)
        self.critic_hidden = T.zeros(self.args.n_agents, args.num_envs, args.hidden_size, device=args.device)
        self.critic_hidden_test = T.zeros(self.args.n_agents, args.hidden_size, device=args.device)
        # fmt:on

    def action(self, observations, **kwargs):
        self.to_remember = []
        actions = []
        val_obs = T.tensor(observations, device="cuda")
        val_obs = T.hstack([val_obs[self.idx_starts + i] for i in range(self.n_agents)])
        for i in range(self.n_agents):

            obs_batch = T.tensor(
                observations[self.idx_starts + i], dtype=T.float, device="cuda"
            )

            (
                action_p,
                action,
                value,
                self.actor_hidden[i],
                self.critic_hidden[i],
            ) = self.agent.choose_action(
                obs_batch, val_obs, i, self.actor_hidden[i], self.critic_hidden[i]
            )
            value = T.squeeze(value)

            self.to_remember.append((obs_batch, action_p, action, value, val_obs))
            actions.append(action.numpy())

        all_actions = []
        for i in range(len(self.idx_starts)):
            for a in actions:
                all_actions.append(a[i])

        return all_actions

    def action_evaluate(self, observations, new_episode):
        self.actor_hidden_test *= 0 if new_episode else 1
        self.to_remember = []
        actions = []
        val_obs = T.tensor(observations, device="cuda")
        val_obs = T.concat([val_obs[i] for i in range(self.n_agents)])
        for i in range(self.n_agents):

            obs_batch = T.tensor(observations[i], dtype=T.float, device="cuda")

            (
                action_p,
                action,
                value,
                self.actor_hidden_test[i],
                self.critic_hidden_test[i],
            ) = self.agent.choose_action(
                obs_batch,
                val_obs,
                i,
                self.actor_hidden_test[i],
                self.critic_hidden_test[i],
            )

            actions.append(action.item())
        # print(actions)
        return actions

    def store(self, total_steps, obs, rewards, dones):

        for i, to_rem in enumerate(self.to_remember):
            done = T.Tensor(dones[self.idx_starts + i])
            reward = T.tensor(rewards[self.idx_starts + i])
            self.agent.remember(
                i,
                to_rem[0],
                to_rem[1],
                to_rem[2],
                to_rem[3],
                to_rem[4],
                reward,
                done,
                self.actor_hidden[i],
                self.critic_hidden[i],
            )
            # fmt:off
            self.actor_hidden *= done.reshape(-1,1).to(self.args.device)
            # fmt:on

        if (self.agent.memory.counter // self.n_agents) == self.args.num_steps:
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

    def create_training_data(self, i):
        b_obs = self.obs[i].reshape((-1,) + self.obs_space).to("cuda")
        b_val_obs = self.valobs[i].reshape((-1,) + (self.obs_space[0]*self.args.n_agents,)).to("cuda")
        b_logprobs = self.logprobs[i].reshape(-1).to("cuda")
        b_actions = self.actions[i].reshape((-1,)).to("cuda")
        b_advantages = self.advantages[i].reshape(-1).to("cuda")
        b_returns = self.returns[i].reshape(-1).to("cuda")
        b_values = self.values[i].reshape(-1).to("cuda")
        b_actor_hidden = self.actor_hidden[i].reshape(-1,self.args.hidden_size).to("cuda")
        b_critic_hidden = self.critic_hidden[i].reshape(-1,self.args.hidden_size).to("cuda")
        
        b_inds = np.arange(self.batch_size)
        return b_obs, b_val_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_actor_hidden, b_critic_hidden, b_inds

    def store_memory(self, i, observations, val_observations, logprobs,action,vals,reward,done,actor_hidden, critic_hidden):
        c = self.counter//self.args.n_agents
        self.obs[i, c] = observations
        self.valobs[i, c] = val_observations

        self.logprobs[i, c] = logprobs
        self.actions[i, c] = action
        self.values[i, c] = vals
        self.rewards[i, c] = reward
        self.dones[i, c] = done
        self.actor_hidden[i, c] = actor_hidden
        self.critic_hidden[i, c] = critic_hidden
        self.counter += 1

    def calculate_returns(self):
        with torch.no_grad():
            if self.gae:
                advantages = torch.zeros_like(self.rewards)
                for i in range(self.args.n_agents):
                    lastgaelam = 0
                    for t in reversed(range(self.num_steps - 1)):
                        nextnonterminal = 1.0 - self.dones[i][t + 1]
                        nextvalues = self.values[i][t + 1]
                        delta = (
                            self.rewards[i][t]
                            + self.gamma * nextvalues * nextnonterminal
                            - self.values[i][t]
                        )
                        advantages[i][t] = lastgaelam = (
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
        space = (self.args.n_agents, self.num_steps, self.num_envs)

        self.obs = T.zeros(space + self.obs_space)
        self.valobs = T.zeros(space + (self.obs_space[0]*self.args.n_agents,))
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
    def __init__(self, obs_shape, actors, action_space, hidden_size):
        super(NNN, self).__init__()

        inp_hid_size = hidden_size + np.array(obs_shape).prod()

        self.input_to_hidden_actor = nn.Sequential(
            layer_init(nn.Linear(inp_hid_size, hidden_size)), nn.Tanh()
        )
        self.input_to_hidden_critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    hidden_size + np.array(obs_shape).prod() * actors, hidden_size
                )
            ),
            nn.Tanh(),
        )

        layer_filters = 256

        self.critic = nn.Sequential(
            layer_init(
                nn.Linear(
                    hidden_size + np.array(obs_shape).prod() * actors, layer_filters
                )
            ),
            nn.Tanh(),
            layer_init(nn.Linear(layer_filters, layer_filters)),
            nn.Tanh(),
            layer_init(nn.Linear(layer_filters, 1), std=1.0),
        )
        self.actors = nn.ModuleList()
        for i in range(actors):
            self.actors.append(
                nn.Sequential(
                    layer_init(nn.Linear(inp_hid_size, layer_filters)),
                    nn.Tanh(),
                    layer_init(nn.Linear(layer_filters, layer_filters)),
                    nn.Tanh(),
                    layer_init(nn.Linear(layer_filters, action_space), std=0.01),
                )
            )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self, x, val_x, actor, hidden_actor, hidden_critic, action=None
    ):
        inp_hidden_actor = torch.hstack([x, hidden_actor])
        inp_hidden_critic = torch.hstack([val_x, hidden_critic])

        new_hidden_actor = self.input_to_hidden_actor(inp_hidden_actor)
        new_hidden_critic = self.input_to_hidden_critic(inp_hidden_critic)

        logits = self.actors[actor](inp_hidden_actor)
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
        self.ppo = NNN(args.obs_space, args.n_agents, action_space, args.hidden_size)
        print(self.ppo)
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
    def remember(self, i, observations, action_p, action, vals, val_obs, reward, done, actor_h, critic_h):
        self.memory.store_memory(i, observations, val_obs, action_p, action, vals, reward, done, actor_h, critic_h)
    
    def save_models(self):
        print("... saving models ...")
        self.ppo.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.ppo.load_checkpoint()
    # fmt:on
    def choose_action(self, observations, val_obs, i, actor_hidden, critic_hidden):
        with torch.no_grad():
            (
                action,
                probs,
                entropy,
                value,
                new_hidden_actor,
                new_hidden_critic,
            ) = self.ppo.get_action_and_value(
                observations, val_obs, i, actor_hidden, critic_hidden
            )

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
        clipfracs = []

        for i in range(args.n_agents):

            (
                b_obs,
                b_val_obs,
                b_logprobs,
                b_actions,
                b_advantages,
                b_returns,
                b_values,
                b_actor_hidden,
                b_critic_hidden,
                b_inds,
            ) = self.memory.create_training_data(i)

            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    (
                        _,
                        newlogprob,
                        entropy,
                        newvalue,
                        _,
                        _,
                    ) = self.ppo.get_action_and_value(
                        b_obs[mb_inds],
                        b_val_obs[mb_inds],
                        i,
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
                    loss = (
                        pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.ppo.parameters(), args.max_grad_norm)
                    self.optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

        self.writer.add_scalar(f"losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar(f"losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar(f"losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar(f"losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar(f"losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar(f"losses/explained_variance", explained_var, global_step)

        self.memory.clear_memory()

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        self.writer.add_scalar(
            "charts/learning_rate",
            self.optimizer.param_groups[0]["lr"],
            global_step,
        )
