import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
import random


class Scenario(BaseScenario):
    def make_world(self, N):
        world = World()
        # set any world properties first
        world.dim_c = 5 * N
        self.N = N
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(self.N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.N)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False

        world.colors = [
            np.array([1, 0.5, 0.5]),
            np.array([0, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5]),
            np.array([0.75, 0.5, 0.5]),
            np.array([0.25, 0.5, 0.5]),
        ]
        return world

    def swap(self, xs, a, b):
        xs[a], xs[b] = xs[b], xs[a]

    def derange(self, xs):
        for a in range(1, len(xs)):
            b = random.choice(range(0, a))
            self.swap(xs, a, b)
        return xs

    def reset_world(self, world, np_random):
        # Assign properties to landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = world.colors[i]
            # set random initial states
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # assign goals to agents
        x = self.derange([i for i in range(self.N)])
        # print(x, "done")

        for i, agent in enumerate(world.agents):
            agent.goal_a = world.agents[x[i]]
            agent.goal_b = np_random.choice(world.landmarks)
            # special colors for goals
            agent.goal_a.color = agent.goal_b.color
            # set random initial states
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

    def reward(self, agent, world):
        if agent.goal_a is None or agent.goal_b is None:
            agent_reward = 0.0
        else:
            agent_reward = np.sqrt(
                np.sum(np.square(agent.goal_a.state.p_pos - agent.goal_b.state.p_pos))
            )
        return -agent_reward

    def global_reward(self, world):
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)

    def observation(self, agent, world):
        # goal color

        # get other agent information
        other_agents = []
        for other in world.agents:
            if other is agent:
                continue
            other_agents.append(np.array(other.color[0]))
            other_agents.append(other.state.c)

        # get other landmarks information
        other_landmarks = []
        for entity in world.landmarks:
            other_landmarks.append(entity.state.p_pos - agent.state.p_pos)
            other_landmarks.append(np.array(entity.color[0]))

        # get goal info
        goal = np.array([agent.goal_a.color[0], agent.goal_b.color[0]])

        full = np.hstack([agent.state.p_vel] + [goal] + other_landmarks + other_agents)

        return full


class raw_env(SimpleEnv):
    def __init__(self, N, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata["name"] = "complex_reference"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
