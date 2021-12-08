import torch
from torch import nn
from torch import optim
import os
from tqdm import tqdm
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv


class ExperimentBuilder(nn.Module):
    def __init__(
        self,
        environment,
        model,
        experiment_name,
        n_episodes,
        episode_len,
        use_gpu,
        lr,
    ):
        super(ExperimentBuilder, self).__init__()

        self.experiment_name = experiment_name
        self.model = model

        self.env = environment
        self.episode_len = episode_len
        print("System learnable parameters")
        for name, value in self.named_parameters():
            print(name, value.shape)

        # Generate the directory names
        self.experiment_folder = os.path.join(
            experiment_name, os.path.abspath(experiment_name)
        )
        self.experiment_logs = os.path.abspath(
            os.path.join(self.experiment_folder, "result_outputs")
        )
        self.experiment_saved_models = os.path.abspath(
            os.path.join(self.experiment_folder, "saved_models")
        )

        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.0

        if not os.path.exists(self.experiment_folder):
            # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
            os.mkdir(self.experiment_logs)  # create the experiment log directory
            os.mkdir(self.experiment_saved_models)
            # create the experiment saved models directory

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

    def save_video(self, episode_len, id):
        import time

        env = VecVideoRecorder(
            self.env,
            self.experiment_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=episode_len * 3,
            name_prefix=f"{self.experiment_name}-{id}",
        )
        for _ in range(3):
            obs = env.reset()
            done = False
            actions = []

            for i in range(episode_len):
                act = self.model.predict(obs, deterministic=True)[0]
                actions.append(str(act))
                obs, _, _, _ = env.step(act)
                time.sleep(0.05)
            env.close()

    def run_experiment(self):

        total_steps = 0

        for ep_i in tqdm(range(0, self.n_episodes)):
            observation = self.env.reset()
            rewards, dones = 0, False
            observation = torch.tensor(observation, device="cuda")
            for step in range(self.episode_len - 1):
                actions = self.model.choose_action(observation)
                act = [actions[2].item()]
                print(act, observation)
                # print(self.env.action_space)
                observation, rewards, dones, infos = self.env.step(act)
                observation = torch.tensor(observation, device="cuda")
                self.env.render()
                # self.model.store(rewards, dones)
                total_steps += 1

            # reward = sum([reward for reward in rewards.values()])
            # self.logger.add_scalar("rewards/end_reward", reward, (ep_i + 1))
