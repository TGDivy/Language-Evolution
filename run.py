from framework.experiment_builder import ExperimentBuilder
from framework.utils.arg_extractor import get_args
import numpy as np
import random
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
from framework.policies.ppo_rec import ppo_rec_policy
import os
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore")

args = get_args()  # get arguments from command line

# Generate the directory names
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

logger.add_hparams(vars(args), {"rewards/end_reward": 0})

# set seeds
np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed
random.seed(args.seed)

# setup environment
if args.env == "simple":
    env = simple_v2
elif args.env == "communication":
    env = simple_reference_v2
elif args.env == "spread":
    env = simple_spread_v2
elif args.env == "adversary":
    env = simple_adversary_v2

env = env.parallel_env(args.episode_len, continuous_actions=False)
num_agents = env.max_num_agents
action_space = env.action_spaces["agent_0"].n
env = ss.pad_observations_v0(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)

num_envs = 4
game = ss.gym_vec_env_v0(env, num_envs, multiprocessing=True)

print(env.action_space)
print(env.action_space.n)
print(env.observation_space)

obs = env.reset()

if args.model == "ppo":
    Policy = ppo_policy
elif args.model == "ppo-rec":
    Policy = ppo_rec_policy
elif args.model == "maddpg":
    Policy = maddpg_policy
else:
    pass
Policy = Policy(
    args,
    num_agents,
    action_space,
    obs.shape,
    args.num_layers,
    args.num_filters,
    args.lr,
    args.device,
)

exp = ExperimentBuilder(
    environment=env,
    Policy=Policy,
    experiment_name=experiment_name,
    logfolder=experiment_videos,
    videofolder=experiment_videos,
    n_episodes=args.n_episodes,
    episode_len=args.episode_len,
    logger=logger,
)
exp.run_experiment()
