import numpy as np
import torch as T
from torch import nn
from torch import optim
import os
from torch.nn import Softmax
from torch.distributions.categorical import Categorical
from torch.nn import HuberLoss
from model_arc import PolicyNetwork as PPO


def actions_to_discrete(movement, symbol):
    return movement + symbol * 5


def random_policy():
    movement = np.random.randint(0, 4)
    symbol = np.random.randint(0, 9)
    return actions_to_discrete(movement, symbol)


class PPOMemory:
    def __init__(self, args):
        self.agents = 2
        self.landmarks = 3
        self.batch_size = args.batch_size
        self.total_memory = args.total_memory
        self.clear_memory()

    def generate_batches(self):
        n_states = len(self.observations)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        dic = {
            "observations": T.tensor(self.observations),
            "action": (
                T.tensor(self.actions_move),
                T.tensor(self.probs_move),
                T.tensor(self.actions_communicate),
                T.tensor(self.probs_communicate),
            ),
            "rewards": (
                T.tensor(self.vals),
                T.tensor(self.rewards),
                T.tensor(self.dones),
            ),
            "batches": T.tensor(batches),
        }

        return dic

    def store_memory(
        self,
        observations,
        move,
        probs_move,
        communicate,
        probs_communicate,
        vals,
        reward,
        done,
    ):
        self.observations[self.counter] = observations[0]

        self.actions_move[self.counter] = move
        self.probs_move[self.counter] = probs_move
        self.actions_communicate[self.counter] = communicate
        self.probs_communicate[self.counter] = probs_communicate

        self.vals[self.counter] = vals
        self.rewards[self.counter] = reward
        self.dones[self.counter] = done
        self.counter += 1

    def clear_memory(self):
        inputs = 3 * 5 + 10 + 3 + 2
        self.observations = T.zeros(self.total_memory, inputs)

        self.actions_move = T.zeros(self.total_memory, 1)
        self.probs_move = T.zeros(self.total_memory, 1)
        self.actions_communicate = T.zeros(self.total_memory, 1)
        self.probs_communicate = T.zeros(self.total_memory, 1)

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
        observations,
        move,
        probs_move,
        communicate,
        probs_communicate,
        vals,
        reward,
        done,
    ):
        self.memory.store_memory(
            observations,
            move,
            probs_move,
            communicate,
            probs_communicate,
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

    def choose_action(self, observations):

        move, communicate, value = self.ppo(observations)

        move_dist = Categorical(move)
        move_action = move_dist.sample()
        move_probs = move_dist.log_prob(move_action)

        communicate_dist = Categorical(communicate)
        communicate_action = communicate_dist.sample()
        communicate_probs = communicate_dist.log_prob(communicate_action)

        return move_probs, communicate_probs, move_action, communicate_action, value

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
            (
                actions_move_arr,
                probs_move_arr,
                actions_communicate_arr,
                probs_communicate_arr,
            ) = dic["action"]
            observations_arr = dic["observations"]

            values = vals_arr.to(self.device)

            advantage = self.advantage(reward_arr, values, dones_arr)

            for batch in dic["batches"]:
                total_loss = 0
                #### game inputs/states
                observations = observations_arr[batch].to(self.device)
                adv = advantage[batch]
                #### action
                actions_move = actions_move_arr[batch].to(self.device)
                actions_communicate = actions_communicate_arr[batch].to(self.device)
                old_move_probs = probs_move_arr[batch].to(self.device)
                old_communicate_probs = probs_communicate_arr[batch].to(self.device)
                #### Actor Critic Run
                move, communicate, critic_value = self.ppo(observations)

                #### Critic Loss
                critic_value = T.squeeze(critic_value)
                returns = advantage[batch] + values[batch]
                critic_loss = self.huber(critic_value, returns)
                total_loss += critic_loss * 0.5

                #### Unit, City Actors Loss
                for (dist, actions, old_probs) in [
                    (move, actions_move, old_move_probs),
                    (communicate, actions_communicate, old_communicate_probs),
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

                total_loss.backward()
                self.ppo.optimizer.step()
                self.ppo.optimizer.zero_grad()

        self.memory.clear_memory()
