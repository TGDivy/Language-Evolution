from matplotlib.collections import PolyCollection
from framework.experiment_builder import ExperimentBuilder
from framework.utils.arg_extractor import get_args
import numpy as np
import random
import torch
from pettingzoo.mpe import (
    simple_v2,
    simple_reference_v2,
    simple_spread_v2,
)
from scenarios import complex_ref, full_ref

# from torch.profiler import profile, record_function, ProfilerActivity
import wandb
import shutil
import supersuit as ss
from framework.policies.ppo import ppo_policy
from framework.policies.ppo3 import ppo_policy3
from framework.policies.maddpg import maddpg_policy
from framework.policies.ppo_rec import ppo_rec_policy
from framework.policies.ppo3_shared import ppo_policy3_shared
from framework.policies.ppo_rnn_shared import ppo_rnn_policy_shared
from framework.policies.ppo_shared_critic import ppo_shared_critic
from framework.policies.ppo_shared_global_critic import ppo_shared_global_critic
from framework.policies.ppo_shared_global_critic_rec import ppo_shared_global_critic_rec
from framework.policies.ppo_shared_global_critic_rec_larg import (
    ppo_shared_global_critic_rec_large,
)
import os
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")
import sys

if __name__ == "__main__":
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
    N = 2
    if args.env == "simple":
        env = simple_v2
    elif args.env == "communication":
        env = simple_reference_v2
    elif args.env == "complex_communication":
        env = complex_ref
    elif args.env == "full_communication_2":
        env = full_ref
        N = 2
    elif args.env == "full_communication_3":
        env = full_ref
        N = 3
    elif args.env == "full_communication_4":
        N = 4
        env = full_ref
    elif args.env == "spread":
        env = simple_spread_v2

    env = env.parallel_env(N=N)
    args.n_agents = env.max_num_agents
    env = ss.pad_observations_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    single_env = ss.concat_vec_envs_v1(env, 1)
    parrallel_env = ss.concat_vec_envs_v1(env, args.num_envs, min(8, args.num_envs))
    parrallel_env.seed(args.seed)
    obs = parrallel_env.reset()
    args.action_space = parrallel_env.action_space.n
    print(
        f"Observation shape: {env.observation_space.shape}, Action space: {parrallel_env.action_space}, all_obs shape: {obs.shape}"
    )
    env.close()

    args.obs_space = env.observation_space.shape
    args.device = "cuda"
    ##############################################################

    ############### MODEL ########################################
    if args.model == "ppo":
        Policy = ppo_policy
    elif args.model == "ppo-rec":
        Policy = ppo_rec_policy
    elif args.model == "maddpg":
        Policy = maddpg_policy
    elif args.model == "ppo_policy3":
        Policy = ppo_policy3
    elif args.model == "ppo_policy3_shared":
        Policy = ppo_policy3_shared
    elif args.model == "ppo_rnn_policy_shared":
        Policy = ppo_rnn_policy_shared
        args.hidden_size = 64
    elif args.model == "ppo_shared_critic":
        Policy = ppo_shared_critic
        args.hidden_size = 64
    elif args.model == "ppo_shared_global_critic":
        Policy = ppo_shared_global_critic
        args.hidden_size = 64
    elif args.model == "ppo_shared_global_critic_rec":
        Policy = ppo_shared_global_critic_rec
        args.hidden_size = 64
    elif args.model == "ppo_shared_global_critic_rec_large":
        Policy = ppo_shared_global_critic_rec_large
        args.hidden_size = 128
    Policy = Policy(args, logger)
    ###############################################################

    exp = ExperimentBuilder(
        args=args,
        train_environment=parrallel_env,
        test_environment=single_env,
        Policy=Policy,
        experiment_name=experiment_name,
        logfolder=experiment_videos,
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
    print("Running EXP!")

    exp.run_experiment()
    single_env.close()
    parrallel_env.close()
    logger.close()

    print("closing now!")
    # sys.exit()
    os._exit(0)
