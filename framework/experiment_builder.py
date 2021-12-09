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


class ExperimentBuilder(nn.Module):
    def __init__(
        self,
        environment,
        Policy: base_policy,
        experiment_name,
        logfolder,
        videofolder,
        n_episodes,
        episode_len,
        logger: SummaryWriter,
    ):
        super(ExperimentBuilder, self).__init__()

        self.Policy = Policy

        self.env = environment
        self.episode_len = episode_len
        print("System learnable parameters")
        for name, value in self.named_parameters():
            print(name, value.shape)

        self.experiment_name = experiment_name
        self.experiment_logs = logfolder
        self.experiment_videos = videofolder

        self.logger = logger
        self.n_episodes = n_episodes

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
            self.env,
            self.experiment_videos,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len - 1,
            name_prefix=f"{self.experiment_name}-{id}",
        )
        N = 2
        for _ in range(N):
            obs = env.reset()
            for i in range(episode_len - 1):
                act, _ = self.Policy.action(obs)
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

    def run_experiment(self):

        total_steps = 0

        for ep_i in tqdm(range(0, self.n_episodes)):
            observation = self.env.reset()
            rewards, dones = 0, False

            for step in range(self.episode_len - 1):

                actions, (value, move_probs) = self.Policy.action(observation)

                observation, rewards, dones, infos = self.env.step(actions)

                self.Policy.store(total_steps, observation, rewards, dones)

                # self.env.render()
                total_steps += 1

                self.logger.add_scalars(
                    "stats/value_reward",
                    {"value": value[0], "reward": rewards[0]},
                    total_steps,
                )
                self.logger.add_scalar("stats/move_prob", move_probs, total_steps)

            if (ep_i + 1) % (self.n_episodes // 10) == 0:
                self.save_video(ep_i)
            self.logger.add_scalar("rewards/end_reward", rewards[0], (ep_i + 1))

        self.save_video("final")
