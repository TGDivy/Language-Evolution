import torch
from torch import nn
from torch import optim
import os
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_video
from framework.utils.base import base_policy
import shutil
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import io
import seaborn as sns


class ExperimentBuilder(nn.Module):
    def __init__(
        self,
        args,
        train_environment,
        test_environment,
        Policy: base_policy,
        experiment_name,
        logfolder,
        videofolder,
        episode_len,
        steps,
        logger: SummaryWriter,
    ):
        super(ExperimentBuilder, self).__init__()

        self.Policy = Policy
        self.args = args
        self.train_env = train_environment
        self.test_env = test_environment
        self.episode_len = episode_len
        self.steps = steps

        self.experiment_name = experiment_name
        self.experiment_logs = logfolder
        self.experiment_videos = videofolder

        self.logger = logger

        self.best_score = -1000

    def save_video(self, step, N=2):
        import time

        episode_len = self.episode_len
        env = VecVideoRecorder(
            self.test_env,
            self.experiment_videos,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len - 1,
            name_prefix=f"{self.experiment_name}-{step}",
        )
        import time

        for _ in range(N):
            obs = env.reset()
            for i in range(episode_len - 1):
                # time.sleep(0.5)
                act = self.Policy.action_evaluate(obs, new_episode=i == 0)
                obs, _, _, _ = env.step(act)

        env.close()
        videos = []
        for i in range(N):
            start = i * (episode_len - 1)
            end = (i + 1) * (episode_len - 1)
            path = os.path.join(
                self.experiment_videos,
                f"{self.experiment_name}-{step}-step-{start}-to-step-{end}.mp4",
            )
            video = read_video(path)
            v = video[0][None, :]
            videos.append(v)

        videos = torch.concat(videos)
        self.logger.add_video(
            f"{self.experiment_name}", videos, global_step=step, fps=10
        )

    def score(self, step):
        N = 50
        env = self.test_env
        end_rewards = []
        mean_episode_reward = []
        agent_end_reward = [[] for i in range(self.args.n_agents)]

        n_agents = self.args.n_agents

        comm = []
        for _ in range(N):
            obs = env.reset()
            rewards = []
            communication = []
            for i in range(self.episode_len):
                act = self.Policy.action_evaluate(obs, new_episode=i == 0)
                # for i, a in enumerate(act):
                communication.append(act // 5)
                obs, reward, _, _ = env.step(act)
                rewards.append(np.mean(reward))
            comm.append(communication)
            mean_episode_reward.append(np.sum(rewards))
            end_rewards.append(reward)
            for i, r in enumerate(reward):
                agent_end_reward[i].append(r)

        self.analyze_comms(comm, step)

        self.logger.add_scalar(f"dev/End_reward", np.mean(end_rewards), step)
        self.logger.add_scalar("dev/Episode_return", np.mean(mean_episode_reward), step)

        for i, ereward in enumerate(agent_end_reward):
            self.logger.add_scalar(f"dev/agent_{i}", np.mean(ereward), step)

        if self.best_score < np.mean(end_rewards):
            self.best_score = np.mean(end_rewards)

        env.close()

    def analyze_comms(self, comms, step):
        comms = np.array(comms, dtype=int)  # 50, 25, 3

        total_unique_symbols_uttered = len(np.unique(comms))
        n_agents = self.args.n_agents
        self.logger.add_scalar(f"comm/vocab_size", total_unique_symbols_uttered, step)

        for i in range(n_agents):
            tusu_agent = len(np.unique(comms[:, :, i]))
            self.logger.add_scalar(f"comm/vocab_size_agent_{i}", tusu_agent, step)

        average_symbols_uttered_ep = (
            np.sum(comms != 0) / np.size(comms) * self.episode_len
        )

        self.logger.add_scalar(f"comm/symbols_per_ep", average_symbols_uttered_ep, step)

        for i in range(n_agents):
            comms_a = comms[:, :, i]
            asue = np.sum(comms_a != 0) / np.size(comms_a) * self.episode_len
            self.logger.add_scalar(f"comm/symbols_per_ep_agent_{i}", asue, step)

        episodes = 10
        vocab = self.args.action_space // 5
        array = comms[:episodes, :, :]
        dummy = np.zeros((episodes, self.episode_len, n_agents + 1), dtype=np.int)
        dummy[:, :, :n_agents] = array
        dummy = np.hstack([dummy[i] for i in range(episodes)])

        pallete = sns.color_palette("crest", vocab)
        labels = "*ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        arrayShow = np.array([[pallete[i] for i in j] for j in dummy])
        patches = [
            mpatches.Patch(color=pallete[i], label=labels[i]) for i in range(0, vocab)
        ]

        fig = plt.figure(figsize=(10, 10))
        a = plt.imshow(arrayShow)
        plt.legend(handles=patches, loc=5, borderaxespad=-5.0)
        self.logger.add_figure("comm/utterances", fig, step)

    def run_experiment(self):

        total_steps = 0
        observation = self.train_env.reset()
        rewards, dones = 0, False

        for step in tqdm(range(0, self.steps + 1), position=1):
            if (step) % (self.steps // 50) == 0:
                self.score(step)

            new_episode = (step % self.args.episode_len) == 0
            actions = self.Policy.action(observation, new_episode=new_episode)

            observation, rewards, dones, infos = self.train_env.step(actions)

            self.Policy.store(total_steps, observation, rewards, dones)

            total_steps += 1
            if (step + 1) % (self.steps // 5) == 0:
                self.save_video(step)

        self.save_video(1e6, N=10)
