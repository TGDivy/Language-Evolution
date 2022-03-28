from matplotlib.collections import PolyCollection
from framework.experiment_builder_iterated import ExperimentBuilderIterated
from framework.utils.arg_extractor import get_args
from iterated_learning.ppo_shared_use_future import language_learner_agents
import numpy as np
import random
import torch

from scenarios import complex_ref, full_ref, iterated

# from torch.profiler import profile, record_function, ProfilerActivity
import wandb
import shutil
import supersuit as ss

import psutil
import os
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import sys


def get_environments(args):
    env = iterated

    landmark_ind = [i for i in range(6)]
    landmark_all = [i for i in range(6)]
    random.shuffle(landmark_ind)
    landmark_ind = landmark_ind[0:4]

    env_learn = env.parallel_env(landmark_ind=landmark_ind)
    env_learn = ss.pad_observations_v0(env_learn)
    env_learn = ss.pettingzoo_env_to_vec_env_v1(env_learn)

    env_test_learn = ss.concat_vec_envs_v1(env_learn, 1)
    env_learn = ss.concat_vec_envs_v1(env_learn, args.num_envs, psutil.cpu_count() - 1)
    env_learn.seed(args.seed)

    env_test_all = env.parallel_env(landmark_ind=landmark_all)
    env_test_all = ss.pad_observations_v0(env_test_all)
    env_test_all = ss.pettingzoo_env_to_vec_env_v1(env_test_all)
    env_test_all = ss.concat_vec_envs_v1(env_test_all, 1)

    obs = env_learn.reset()
    args.action_space = env_learn.action_space.n
    args.obs_space = env_learn.observation_space.shape
    args.n_agents = 2
    args.landmark_ind = landmark_ind

    print(landmark_ind)

    print(
        f"Observation shape: {env_learn.observation_space.shape}, Action space: {env_learn.action_space}, all_obs shape: {obs.shape}"
    )

    return env_learn, env_test_learn, env_test_all, args


def iterated_learning(
    args, logger, experiment_name, experiment_videos, experiment_saved_models
):
    args.device = "cuda"

    for i, j in enumerate(range(1, 10)):

        agent_names = [i, j]

        # setup environment ###########################################
        env_learn, env_test_learn, env_test_all, args = get_environments(args)
        logger.add_text(
            "possible_types",
            str(args.landmark_ind),
        )

        ############### MODEL ########################################
        Policy = language_learner_agents

        Policy = Policy(args, logger, agent_names)
        PATH = experiment_saved_models
        Policy.load_agents(PATH)
        ###############################################################

        exp = ExperimentBuilderIterated(
            args=args,
            train_environment=env_learn,
            test_environment=env_test_learn,
            test_all_env=env_test_all,
            Policy=Policy,
            experiment_name=experiment_name,
            logfolder=experiment_videos,
            experiment_saved_models=experiment_saved_models,
            videofolder=experiment_videos,
            episode_len=args.episode_len,
            steps=args.total_timesteps,
            logger=logger,
            agent_names=agent_names,
        )
        logger.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        exp.run_experiment()

        env_learn.close()
        env_test_learn.close()
        env_test_all.close()

    logger.close()
    os._exit(0)


def main():
    args = get_args()  # get arguments from command line
    # Generate Directories##########################
    experiment_name = f"{args.model}-{args.env}-{args.experiment_name}"
    experiment_folder = os.path.join(os.path.abspath("experiments"), experiment_name)
    experiment_logs = os.path.abspath(os.path.join(experiment_folder, "result_outputs"))
    experiment_videos = os.path.abspath(os.path.join(experiment_folder, "videos"))
    experiment_saved_models = os.path.abspath(
        os.path.join(experiment_folder, "saved_models")
    )

    if os.path.exists(experiment_folder):
        shutil.rmtree(experiment_folder)

    os.mkdir(experiment_folder)  # create the experiment directory
    os.mkdir(experiment_logs)  # create the experiment log directory
    os.mkdir(experiment_saved_models)
    os.mkdir(experiment_videos)
    ################################################
    if args.wandb:
        wandb.init(
            project="iterated_language_evolution",
            entity=None,
            sync_tensorboard=True,
            config=vars(args),
            name=experiment_name,
            monitor_gym=True,
            save_code=True,
            dir=os.path.abspath("experiments"),
        )
    logger = SummaryWriter(experiment_logs)

    print("\n*****Parameters*****")
    space = " "
    print(
        "\n".join(
            [
                f"--- {param}: {(20-len(param))*space} {value}"
                for param, value in vars(args).items()
            ]
        )
    )
    print("*******************")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    iterated_learning(
        args, logger, experiment_name, experiment_videos, experiment_saved_models
    )


if __name__ == "__main__":
    main()
