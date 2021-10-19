class base_policy:
    def __init__(self) -> None:
        pass

    def action(self, obeservation: dict):
        raise NotImplementedError()

    def store(self, rewards, dones):
        pass


class Args:
    def __init__(
        self,
        chkpt_dir="/",
        log_dir="/run/",
        n_episodes=5000,
        max_cycles=25,
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
        self.log_dir = log_dir

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
