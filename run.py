from framework.experiment_builder import ExperimentBuilder
from framework.utils.arg_extractor import get_args
import numpy as np
import torch
from pettingzoo.mpe import simple_v2, simple_reference_v2, simple_reference_v3
import supersuit as ss
from framework.ppo.policy import ppo_policy

args = get_args()  # get arguments from command line

# set seeds
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# setup environment
if args:
    pass
env = simple_v2.parallel_env(max_cycles=args.episode_len)
num_agents = env.max_num_agents

env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v0(game, 10, num_cpus=5, base_class='stable_baselines3')
obs = env.reset()

if args.model == "ppo":
    Policy = ppo_policy(args, num_agents, obs.shape, args.num_layers, args.num_filters)
else:
    pass

exp = ExperimentBuilder(
    environment=env,
    Policy=Policy,
    experiment_name=args.experiment_name,
    n_episodes=args.n_episodes,
    episode_len=args.episode_len,
)
exp.run_experiment()
