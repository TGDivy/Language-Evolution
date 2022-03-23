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
from framework.policies.ppo_rec_global_critic import ppo_rec_global_critic
from framework.policies.ppo_no_scaling_rec_global_critic import (
    ppo_no_scaling_rec_global_critic,
)
from framework.policies.ppo_attend_agent import ppo_attend_agent
from framework.policies.ppo_rec_global_critic_fixed import ppo_rec_global_critic_fixed
from framework.policies.ppo_shared_future import ppo_shared_future
from framework.policies.ppo_shared_use_future import ppo_shared_use_future


policies_dic = {
    "ppo_policy": ppo_policy,
    "ppo_policy3": ppo_policy3,
    "maddpg_policy": maddpg_policy,
    "ppo_shared_future": ppo_shared_future,
    "ppo_shared_use_future": ppo_shared_use_future,
}
