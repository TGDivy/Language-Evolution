from utils.arg_extractor import get_args
import numpy as np
import torch
from pettingzoo.mpe import simple_v2, simple_reference_v2, simple_reference_v3
import supersuit as ss
from ppo import simple_ppo

args = get_args()  # get arguments from command line

# set seeds
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# setup environment
if args:
    pass
env = simple_v2.parallel_env(max_cycles=80)
env = ss.pettingzoo_env_to_vec_env_v0(env)
# env = ss.concat_vec_envs_v0(game, 10, num_cpus=5, base_class='stable_baselines3')

if args.model = "ppo":
    simple_ppo.Agent(args)