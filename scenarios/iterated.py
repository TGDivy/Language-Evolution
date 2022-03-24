import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
import random

landmark_ind = [i for i in range(15)]


class Scenario(BaseScenario):
    def make_world(self, landmark_ind=landmark_ind):
        world = World()
        # set any world properties first
        world.dim_c = 10
        self.N = 2
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(self.N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = False

        # add landmarks

        world.colors = [
            np.array([1, 0.5, 0.5]),
            np.array([0, 0.5, 0.5]),
            np.array([0.33, 0.5, 0.5]),
            np.array([0.66, 0.5, 0.5]),
        ]
        world.lradi = [0.10, 0.15, 0.20]

        self.combinations = len(world.colors) * len(world.lradi)

        self.landmarks = []

        for i, color in enumerate(world.colors):
            for j, lradius in enumerate(world.lradi):
                landmark = Landmark()
                landmark.name = "landmark %d" % ((i * 3) + j)
                landmark.collide = False
                landmark.movable = False
                landmark.size = lradius
                landmark.color = color
                self.landmarks.append(landmark)

        self.landmarkN = 3
        self.landmark_ind = landmark_ind

        return world

    def swap(self, xs, a, b):
        xs[a], xs[b] = xs[b], xs[a]

    def derange(self, xs):
        for a in range(1, len(xs)):
            b = random.choice(range(0, a))
            self.swap(xs, a, b)
        return xs

    def reset_world(self, world, np_random):
        # Assign properties to landmarks.
        # Select landmarks.
        x = self.derange(self.landmark_ind)
        world.landmarks = self.landmarks[x[: self.landmarkN]]

        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        # assign goals to agents
        x = self.derange([i for i in range(self.N)])
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
            d_reward = 0.0
        else:
            diff = agent.goal_a.state.p_pos - agent.goal_b.state.p_pos
            distance = np.linalg.norm(diff)
            d_reward = distance if distance >= (agent.goal_b.size - agent.size) else 0

        comm_penalty = (np.argmax(agent.state.c) > 0) * 0.03
        agent_reward = -d_reward - comm_penalty
        return agent_reward

    def global_reward(self, world):
        all_rewards = sum(self.reward(agent, world) for agent in world.agents)
        return all_rewards / len(world.agents)

    def observation(self, agent, world):
        # get other agent information
        other_agents = []
        for other in world.agents:
            if other is agent:
                continue
            pos = other.state.p_pos - agent.state.p_pos
            other_agents.append(pos)
            other_agents.append(np.array(other.color[0]))
            other_agents.append(other.state.c)

        # get other landmarks information
        other_landmarks = [np.zeros(4) for i in range(5)]
        for i, entity in enumerate(world.landmarks):
            pos = entity.state.p_pos - agent.state.p_pos
            other_landmarks[i][0:2] = pos
            other_landmarks[i][2] = entity.color[0]
            other_landmarks[i][3] = entity.size

        # get goal info
        goal = np.array([agent.goal_b.size, agent.goal_b.color[0]])

        full = np.hstack([agent.state.p_vel] + [goal] + other_landmarks + other_agents)

        return full


class raw_env(SimpleEnv):
    def __init__(
        self, landmark_ind, local_ratio=0.5, max_cycles=25, continuous_actions=False
    ):
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(landmark_ind)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata["name"] = "complex_reference"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
