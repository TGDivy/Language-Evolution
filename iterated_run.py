from matplotlib.collections import PolyCollection
from framework.experiment_builder import ExperimentBuilder
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
            project="language_evolution",
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

    # logger.add_hparams(vars(args), {"rewards/end_reward": 0})

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    # setup environment ###########################################
    env = full_ref
    N = 2

    env = env.parallel_env(N=N)
    args.n_agents = env.max_num_agents
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    single_env = ss.concat_vec_envs_v1(env, 1)
    parrallel_env = ss.concat_vec_envs_v1(env, args.num_envs, psutil.cpu_count() - 1)
    parrallel_env.seed(args.seed)
    obs = parrallel_env.reset()
    args.action_space = parrallel_env.action_space.n
    print(
        f"Observation shape: {env.observation_space.shape}, Action space: {parrallel_env.action_space}, all_obs shape: {obs.shape}"
    )
    env.close()

    args.obs_space = env.observation_space.shape
    args.device = "cuda"

    ############### MODEL ########################################
    Policy = language_learner_agents

    Policy = Policy(args, logger)
    if args.load_weights_name:
        PATH = os.path.abspath("experiments") + args.load_weights_name + "/saved_models"
        Policy.load_agents(PATH)
    ###############################################################

    exp = ExperimentBuilder(
        args=args,
        train_environment=parrallel_env,
        test_environment=single_env,
        Policy=Policy,
        experiment_name=experiment_name,
        logfolder=experiment_videos,
        experiment_saved_models=experiment_saved_models,
        videofolder=experiment_videos,
        episode_len=args.episode_len,
        steps=args.total_timesteps,
        logger=logger,
    )
    logger.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    exp.run_experiment()
    single_env.close()
    parrallel_env.close()
    logger.close()

    os._exit(0)


if __name__ == "__main__":
    main()
