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


class ExperimentBuilder(nn.Module):
    def __init__(
        self,
        train_environment,
        test_environment,
        Policy: base_policy,
        experiment_name,
        logfolder,
        videofolder,
        episode_len,
        logger: SummaryWriter,
    ):
        super(ExperimentBuilder, self).__init__()

        self.Policy = Policy

        self.train_env = train_environment
        self.test_env = test_environment
        self.episode_len = episode_len
        print("System learnable parameters")
        for name, value in self.named_parameters():
            print(name, value.shape)

        self.experiment_name = experiment_name
        self.experiment_logs = logfolder
        self.experiment_videos = videofolder

        self.logger = logger

    def save_model(self, model_save_dir, model_save_name, index):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state[
            "network"
        ] = self.state_dict()  # save network parameter and other variables.
        torch.save(
            self.state,
            f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(index))),
        )  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(
            f=os.path.join(
                model_save_dir, "{}_{}".format(model_save_name, str(model_idx))
            )
        )
        self.load_state_dict(state_dict=state["network"])
        return state, state["best_val_model_idx"], state["best_val_model_acc"]

    def logging(self):
        pass

    def save_video(self, id):
        import time

        episode_len = self.episode_len
        env = VecVideoRecorder(
            self.test_env,
            self.experiment_videos,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len - 1,
            name_prefix=f"{self.experiment_name}-{id}",
        )
        N = 2
        import time

        for _ in range(N):
            obs = env.reset()
            for i in range(episode_len - 1):
                # time.sleep(0.5)
                act, _ = self.Policy.action(obs, new_episode=i == 0, evaluate=True)
                obs, _, _, _ = env.step(act)
        env.close()

        videos = []
        for i in range(N):
            start = i * (episode_len - 1)
            end = (i + 1) * (episode_len - 1)
            path = os.path.join(
                self.experiment_videos,
                f"{self.experiment_name}-{id}-step-{start}-to-step-{end}.mp4",
            )
            video = read_video(path)
            v = video[0][None, :]
            videos.append(v)

        videos = torch.concat(videos)
        self.logger.add_video(f"{self.experiment_name}-{id}", videos, fps=10)

    def score(self, step):
        N = 50
        env = self.test_env
        rewards = []
        for _ in range(N):
            obs = env.reset()
            for i in range(self.episode_len):
                act, _ = self.Policy.action(obs, new_episode=i == 0, evaluate=True)
                act = act.numpy()
                obs, reward, _, _ = env.step(act)
            rewards.append(reward)

        self.logger.add_scalar("rewards/Val_end_reward", np.mean(rewards), step)
        env.close()

    def run_experiment(self):

        total_steps = 0
        observation = self.train_env.reset()
        rewards, dones = 0, False

        steps = 300000

        for step in tqdm(range(0, steps)):
            actions, (value, move_probs) = self.Policy.action(
                observation, new_episode=step == 0
            )

            observation, rewards, dones, infos = self.train_env.step(actions)
            # self.env.render()

            self.Policy.store(total_steps, observation, rewards, dones)

            total_steps += 1

            # self.logger.add_scalar("stats/move_prob", move_probs.exp(), total_steps)

            if (step + 1) % (10000) == 0:
                self.score(step)

            if (step + 1) % (steps // 5) == 0:
                self.save_video(step)

        # self.save_video("final")
