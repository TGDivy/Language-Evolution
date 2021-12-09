from framework.experiment_builder import ExperimentBuilder
from framework.utils.arg_extractor import get_args
import numpy as np
import torch
from pettingzoo.mpe import (
    simple_v2,
    simple_reference_v2,
    simple_reference_v3,
    simple_push_v2,
    simple_spread_v2,
    simple_adversary_v2,
)
import shutil
import supersuit as ss
from framework.policies.ppo import ppo_policy
from framework.policies.maddpg import maddpg_policy
import os
from torch.utils.tensorboard import SummaryWriter

args = get_args()  # get arguments from command line

# Generate the directory names
experiment_folder = os.path.join(
    os.path.abspath("experiments"), f"{args.experiment_name}-{args.env}"
)
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

logger = SummaryWriter(experiment_logs)

# set seeds
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# setup environment
if args.env == "simple":
    env = simple_v2.parallel_env(max_cycles=args.episode_len)
elif args.env == "communication":
    env = simple_reference_v3.parallel_env(max_cycles=args.episode_len)
elif args.env == "spread":
    env = simple_spread_v2.parallel_env(max_cycles=args.episode_len)
elif args.env == "adversary":
    env = simple_adversary_v2.parallel_env(max_cycles=args.episode_len)

num_agents = env.max_num_agents
# print(env.action_spaces)
action_space = env.action_spaces["agent_0"].n
env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v0(game, 10, num_cpus=5, base_class='stable_baselines3')
obs = env.reset()

if args.model == "ppo":
    Policy = maddpg_policy(
        args,
        num_agents,
        action_space,
        obs.shape,
        args.num_layers,
        args.num_filters,
        args.lr,
        args.device,
    )
else:
    pass

exp = ExperimentBuilder(
    environment=env,
    Policy=Policy,
    logfolder=experiment_videos,
    videofolder=experiment_videos,
    n_episodes=args.n_episodes,
    episode_len=args.episode_len,
    logger=logger,
)
exp.run_experiment()
