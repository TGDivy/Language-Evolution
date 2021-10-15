import numpy as np
from stable_baselines3 import PPO
import torch as T
from torch import nn
from torch import optim
import os
from torch.nn import Softmax
from torch.distributions.categorical import Categorical
from torch.nn import HuberLoss


def actions_to_discrete(movement, symbol):
    return movement + symbol * 5


def random_policy():
    movement = np.random.randint(0, 4)
    symbol = np.random.randint(0, 9)
    return actions_to_discrete(movement, symbol)


class PPO(nn.Module):
    def __init__(self, args):
        super(PPO, self).__init__()
        alpha = args.alpha * 4
        self.checkpoint_file = os.path.join(args.chkpt_dir, "actor_torch_ppo")

        self.landmarks = 3
        self.inputs = 2 + self.landmarks * 5 + 3

        self.observation = nn.Sequential(
            nn.Linear(self.inputs, self.inputs * 4),
            nn.ReLU(),
            nn.Linear(self.inputs * 4, self.inputs * 2),
            nn.ReLU(),
            nn.Linear(self.inputs * 2, self.inputs),
            nn.ReLU(),
        )

        self.move = nn.Linear(self.inputs, 5)
        self.communicate = nn.Linear(self.inputs, 10)

        self.critic = nn.Sequential(
            nn.Linear(self.inputs, self.inputs * 2),
            nn.ReLU(),
            nn.Linear(self.inputs * 2, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observations):
        dist = self.softmax()

        value = self.critic()

        return dist, value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class PPOMemory:
    def __init__(self, args):
        self.agents = 2
        self.landmarks = 3
        self.batch_size = args.batch_size
        self.total_memory = args.total_memory
        self.clear_memory()

    def generate_batches(self):
        n_states = 25
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        dic = {
            "states": (T.tensor(self.states), T.tensor(self.stats)),
            "units": (
                T.tensor(self.unit_vals),
                T.tensor(self.unit_actions),
                T.tensor(self.unit_probs),
            ),
            "cities": (
                T.tensor(self.city_vals),
                T.tensor(self.city_actions),
                T.tensor(self.city_probs),
            ),
            "rewards": (
                T.tensor(self.vals),
                T.tensor(self.rewards),
                T.tensor(self.dones),
            ),
            "batches": batches,
        }

        return dic

    def store_memory(
        self,
        stats,
        state,
        unit_vals,
        unit_actions,
        unit_probs,
        city_vals,
        city_actions,
        city_probs,
        vals,
        reward,
        done,
    ):
        self.states[self.counter] = state[0]
        self.stats[self.counter] = stats[0]

        self.unit_vals[self.counter] = unit_vals[0]
        self.unit_actions[self.counter] = unit_actions
        self.unit_probs[self.counter] = unit_probs

        self.city_vals[self.counter] = city_vals[0]
        self.city_actions[self.counter] = city_actions
        self.city_probs[self.counter] = city_probs

        self.vals[self.counter] = vals
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
        self.counter += 1

    def clear_memory(self):
        self.states = T.zeros(self.total_memory, 12, 16, 16)
        self.stats = T.zeros(self.total_memory, 11)

        self.unit_vals = T.zeros(self.total_memory, self.max_controlled_units, 5)
        self.unit_actions = T.zeros(self.total_memory, self.max_controlled_units)
        self.unit_probs = T.zeros(self.total_memory, self.max_controlled_units)

        self.city_vals = T.zeros(self.total_memory, self.max_controlled_cities, 4)
        self.city_actions = T.zeros(self.total_memory, self.max_controlled_cities)
        self.city_probs = T.zeros(self.total_memory, self.max_controlled_cities)

        self.vals = T.zeros(self.total_memory)
        self.rewards = T.zeros(self.total_memory)
        self.dones = T.zeros(self.total_memory)
        self.counter = 0


class Agent:
    def __init__(self, args):
        self.gamma = args.gamma
        self.policy_clip = args.policy_clip
        self.n_epochs = args.n_epochs
        self.gae_lambda = args.gae_lambda
        self.entropy = args.entropy

        self.ppo = PPO(args)
        self.memory = PPOMemory(args)
        self.huber = HuberLoss(reduction="mean", delta=1.0)
        self.device = self.ppo.device

    def remember(
        self,
        stats,
        state,
        unit_vals,
        unit_actions,
        unit_probs,
        city_vals,
        city_actions,
        city_probs,
        vals,
        reward,
        done,
    ):
        self.memory.store_memory(
            stats,
            state,
            unit_vals,
            unit_actions,
            unit_probs,
            city_vals,
            city_actions,
            city_probs,
            vals,
            reward,
            done,
        )

    def save_models(self):
        print("... saving models ...")
        self.ppo.save_checkpoint()

    def load_models(self):
        print("... loading models ...")
        self.ppo.load_checkpoint()

    def choose_action(self, stats, state, unit_val, city_val):

        unit_dist, city_dist, value = self.ppo(stats, state, unit_val, city_val)

        unit_dist = Categorical(unit_dist)
        unit_action = unit_dist.sample()
        unit_probs = unit_dist.log_prob(unit_action)

        city_dist = Categorical(city_dist)
        city_action = city_dist.sample()
        city_probs = city_dist.log_prob(city_action)

        return unit_action, unit_probs, city_action, city_probs, value

    def advantage(self, reward_arr, values, dones_arr):
        advantage = np.zeros(len(reward_arr), dtype=np.float16)
        for t in range(len(reward_arr) - 1):
            discount = 1
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
        for _ in range(self.n_epochs):
            dic = self.memory.generate_batches()

            vals_arr, reward_arr, dones_arr = dic["rewards"]
            unit_val_arr, unit_action_arr, old_unit_prob_arr = dic["units"]
            city_val_arr, city_action_arr, old_city_prob_arr = dic["cities"]

            state_arr, stats_arr = dic["states"]

            values = vals_arr.to(self.device)

            advantage = self.advantage(reward_arr, values, dones_arr)

            for batch in dic["batches"]:
                total_loss = 0
                #### game inputs/states
                states = state_arr[batch].to(self.device)
                stats = stats_arr[batch].to(self.device)
                adv = advantage[batch]
                #### Unit
                unit_vals = unit_val_arr[batch].to(self.device)
                old_unit_probs = old_unit_prob_arr[batch].to(self.device)
                unit_actions = unit_action_arr[batch].to(self.device)
                #### City
                city_vals = city_val_arr[batch].to(self.device)
                old_city_probs = old_city_prob_arr[batch].to(self.device)
                city_actions = city_action_arr[batch].to(self.device)
                #### Actor Critic Run
                unit_dist, city_dist, critic_value = self.ppo(
                    stats, states, unit_vals, city_vals
                )

                #### Critic Loss
                critic_value = T.squeeze(critic_value)
                returns = advantage[batch] + values[batch]
                critic_loss = self.huber(critic_value, returns)
                total_loss += critic_loss * 0.5

                #### Unit, City Actors Loss
                for (dist, actions, old_probs) in [
                    (unit_dist, unit_actions, old_unit_probs),
                    (city_dist, city_actions, old_city_probs),
                ]:
                    dist = Categorical(dist)
                    new_probs = dist.log_prob(actions)
                    prob_ratio = new_probs.exp() / old_probs.exp()

                    weighted_probs = prob_ratio * adv.reshape(-1, 1)
                    weighted_clipped_probs = T.clamp(
                        prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                    ) * adv.reshape(-1, 1)

                    actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                    total_loss += actor_loss
                    total_loss += self.entropy * T.mean(dist.entropy())

                self.ppo.optimizer.zero_grad()
                total_loss.backward()
                self.ppo.optimizer.step()

        self.memory.clear_memory()
