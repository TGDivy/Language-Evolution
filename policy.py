import numpy as np
from stable_baselines3 import PPO


def actions_to_discrete(movement, symbol):
    return movement + symbol * 5


def random_policy():
    movement = np.random.randint(0, 4)
    symbol = np.random.randint(0, 9)
    return actions_to_discrete(movement, symbol)
