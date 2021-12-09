import argparse


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser(
        description="Welcome to the MLP course's Pytorch training and inference helper script"
    )

    parser.add_argument(
        "--batch_size",
        nargs="?",
        type=int,
        default=5,
        help="Batch_size for experiment",
    )

    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        default=7112018,
        help="Seed to use for random number generator for experiment",
    )
    parser.add_argument(
        "--n_episodes",
        nargs="?",
        type=int,
        default=100,
        help="n_episodes",
    )
    parser.add_argument(
        "--episode_len",
        nargs="?",
        type=int,
        default=25,
        help="episode_len",
    )
    parser.add_argument(
        "--n_epochs",
        nargs="?",
        type=int,
        default=3,
        help="n_epochs",
    )
    parser.add_argument(
        "--num_layers",
        nargs="?",
        type=int,
        default=7,
        help="num_layers",
    )
    parser.add_argument(
        "--num_filters",
        nargs="?",
        type=int,
        default=128,
        help="num_filters",
    )
    parser.add_argument(
        "--total_memory",
        nargs="?",
        type=int,
        default=10,
        help="num_filters",
    )
    parser.add_argument(
        "--device",
        nargs="?",
        type=str,
        default="cuda",
        help="num_filters",
    )
    parser.add_argument("--alpha", nargs="?", type=float, default=1e-4, help="alpha")
    parser.add_argument("--gamma", nargs="?", type=float, default=0.99, help="gamma")

    parser.add_argument(
        "--communicate", nargs="?", type=int, default=0, help="communicate"
    )

    parser.add_argument(
        "--gae_lambda",
        nargs="?",
        type=float,
        default=0.90,
        help="gae_lambda",
    )
    parser.add_argument(
        "--policy_clip",
        nargs="?",
        type=float,
        default=0.2,
        help="policy_clip",
    )
    parser.add_argument(
        "--lr",
        nargs="?",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--entropy",
        nargs="?",
        type=float,
        default=0.01,
        help="0.01",
    )

    parser.add_argument(
        "--model",
        nargs="?",
        type=str,
        help="Policy to be used",
    )
    parser.add_argument(
        "--experiment_name",
        nargs="?",
        type=str,
        default="exp_1",
        help="Experiment name - to be used for building the experiment folder",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="simple",
        help="environment for agent",
    )
    args = parser.parse_args()
    print(args)
    return args
