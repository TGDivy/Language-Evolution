from pettingzoo.mpe import simple_v2, simple_reference_v2, simple_reference_v3
import supersuit as ss
import numpy as np


if __name__ == "__main__":
    num_envs = 3
    env = simple_reference_v2.parallel_env(max_cycles=25, continuous_actions=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    game = ss.stable_baselines3_vec_env_v0(env, num_envs=2, multiprocessing=True)

    obs = game.reset()

    print(obs.shape)

    new_obs = game.step([[0, 0], [0, 0]])
