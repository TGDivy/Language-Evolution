from framework.experiment_builder import ExperimentBuilder
from framework.utils.arg_extractor import get_args
import numpy as np
import torch
from pettingzoo.mpe import simple_v2, simple_reference_v2, simple_reference_v3
import supersuit as ss
from framework.policies.ppo import ppo_policy
from framework.policies.maddpg import maddpg_policy

args = get_args()  # get arguments from command line

# set seeds
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

# setup environment
if args:
    pass
env = simple_v2.parallel_env(max_cycles=args.episode_len)
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
    experiment_name=args.experiment_name,
    n_episodes=args.n_episodes,
    episode_len=args.episode_len,
)
exp.run_experiment()
