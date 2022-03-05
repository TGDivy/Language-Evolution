import argparse
from distutils.util import strtobool
import os


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser()

    parser.add_argument("--learning-rate",type=float,default=2.5e-4,help="the learning rate of the optimizer",)
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=25000, help="total timesteps of the experiments",)
    parser.add_argument("--torch-deterministic",type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda",type=lambda x: bool(strtobool(x)),default=True,nargs="?",const=True,help="if toggled, cuda will be enabled by default"
    )
    parser.add_argument("--capture-video",type=lambda x: bool(strtobool(x)),default=False,nargs="?",const=True,help="weather to capture videos of the agent performances (check out `videos` folder)",)

    # Algorithm specific arguments
    parser.add_argument("--num-envs",type=int,default=4,help="the number of parallel game environments",)
    parser.add_argument("--num-steps",type=int,default=25,help="the number of steps to run in each environment per policy rollout",)
    parser.add_argument("--anneal-lr",type=lambda x: bool(strtobool(x)),default=True,nargs="?",const=True,help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae",type=lambda x: bool(strtobool(x)),default=True,nargs="?",const=True,help="Use GAE for advantage computation",)
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae-lambda",type=float,default=0.95,help="the lambda for the general advantage estimation",)
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4, help="the K epochs to update the policy")
    parser.add_argument("--norm-adv",type=lambda x: bool(strtobool(x)),default=True,nargs="?",const=True,help="Toggles advantages normalization",
    )
    parser.add_argument("--clip-coef",type=float,default=0.2,help="the surrogate clipping coefficient",)
    parser.add_argument("--clip-vloss",type=lambda x: bool(strtobool(x)),default=True,nargs="?",const=True,help="Toggles wheter or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--max-grad-norm",type=float,default=0.5,help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl",type=float,default=None,help="the target KL divergence threshold")

    parser.add_argument("--model",nargs="?",type=str,help="Policy to be used")
    parser.add_argument("--experiment_name",nargs="?",type=str,default="exp_1",help="Experiment name - to be used for building the experiment folder")
    parser.add_argument("--env",type=str,default="simple",help="environment for agent",)
    parser.add_argument("--episode_len", type=int, default=25)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on

    return args
