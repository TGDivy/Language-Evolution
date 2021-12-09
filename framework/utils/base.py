import os
from torch.utils.tensorboard import SummaryWriter


class base_policy:
    def __init__(self) -> None:
        pass

    def add_logger(self, logger):
        self.logger = logger

    def action(self, obeservation: dict):
        raise NotImplementedError()

    def store(self, total_steps, obs, rewards, dones):
        pass


class Args:
    def __init__(
        self,
        chkpt_dir="/",
        log_dir="run",
        exp_name="exp_name",
        n_episodes=5000,
        max_cycles=50,
        seed=7,
        n_epochs=5,
        alpha=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        policy_clip=0.1,
        entropy=0.01,
        batch_size=5,
        total_memory=25,
    ) -> None:

        self.chkpt_dir = chkpt_dir
        print(os.getcwd())
        self.log_dir = os.path.join(os.getcwd(), log_dir, exp_name)
        self.exp_name = exp_name
        print(self.log_dir)

        self.n_episodes = n_episodes
        self.max_cycles = max_cycles

        self.seed = seed

        ### model details
        self.total_memory = total_memory
        self.batch_size = batch_size

        self.n_epochs = n_epochs
        self.alpha = alpha
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.entropy = entropy

        self.logger = self.logging()

    def logging(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        return SummaryWriter(self.log_dir)
