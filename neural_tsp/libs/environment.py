"""Gym environment for the TSP
"""

from typing import Tuple
from collections import namedtuple

import gym
from gym import spaces
import numpy as np
from scipy.spatial.distance import pdist, squareform

State = namedtuple("State", "locs, dist_matrix, order, dist")


class TSPEnvironment(gym.Env):
    """Environment representing the Traveling Salesman Problem (TSP)"""

    def __init__(self, num_locs_range: Tuple[int, int]):
        super(TSPEnvironment, self).__init__()
        self.num_locs_range = num_locs_range
        self.reset()

    def step(self, action):
        # Update internal state and calculate reward
        self._update_state(action)
        # Get experience tuple
        state = self._next_observation()
        reward = self.reward
        done = False
        info = dict()
        return state, reward, done, info

    def reset(self):
        """Reset the environment.
        
        Creates a new set of locations chosen from a uniform distribution,
        calculates their distance matrix, etc.
        """
        # Create stop locations
        self.num_locs = np.random.randint(*self.num_locs_range)
        self.locations = np.random.uniform(
            low=0, high=1, size=(self.num_locs, 2)
        ).astype(np.float32)
        # Calculate distance matrix between locations
        self.D = squareform(pdist(self.locations, metric="euclidean")).astype(
            np.float32
        )
        # Set initial visitiation order (in sequence by index by default)
        self.order = np.arange(self.num_locs)

        # Set up state space
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(
                    low=0, high=1, shape=(self.num_locs, 2), dtype=np.float32
                ),  # (x,y) location of stops
                spaces.Box(
                    low=0, high=self.D.max(), shape=self.D.shape, dtype=np.float32
                ),  # Distance matrix between locations
                spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_locs,)
                ),  # The previous visitation order
                spaces.Box(
                    low=0, high=self.D.max() * self.num_locs, shape=()
                ),  # The distance of the last iteration
            )
        )
        # Maybe include previous actions in state space?

        # Action space is vector of ranks (the order in which to visit nodes); could be any real number in principle
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_locs,))

        # Initialize rewards
        self._update_state(self.order)

        return self._next_observation()

    # def render(self):
    #     """Render a plot of the TSP circuit?"""
    #     import matplotlib.pyplot as plt

    #     fig, ax = plt.subplots()
    #     for i, loc in enumerate(self.locations):
    #         m = f"${i}$"
    #         ax.scatter(*loc, marker=m, s=100 / len(str(i)), color="black")
    #     locs = self.locations[self.order]
    #     ax.plot(*locs.T)
    #     ax.plot(*locs[((-1, 0),)].T, color="C0")
    #     plt.show()

    def close(self):
        pass

    def seed(self):
        pass

    def _next_observation(self):
        return State(self.locations, self.D, self.order, self.obj_value)

    def _update_state(self, action):
        # Get the visitation order by ranking the elements of <action>
        order = np.argsort(action)
        self.order = order
        # Get the total distance of the circuit
        idx = np.vstack([order, np.roll(order, -1)])
        dist = self.D[idx].sum()
        # Update reward/objective
        self.obj_value = dist
        self.reward = -dist
        return None

