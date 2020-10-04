"""Gym environment for the TSP
"""

from typing import Tuple
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import gym
from gym import spaces
import torch
from torch import Tensor

"""A tuple of different pieces that make up the "state"

locs (Tensor): A tensor of shape (N,2) where each row are (x,y) coordinates
    of a location in the circuit.
dmatrix (Tensor): An NxN matrix where the (i,j) element represents the
    distance from location i to location j. Optional
seq (Tensor): An integer vector of length N denoting the sequence of location
    visitations. Since each circuit is a closed loop, this is invariant under
    shifts of the elements (i.e. index i -> i + j mod N)
"""
State = namedtuple("State", "locs, dmatrix, seq, dist")


class FixedTSPEnv(gym.Env):
    """
    Environment representing the Traveling Salesman Problem (TSP) for a fixed
    set of locations.
    
    Parameters:
        locs (Tensor): A tensor of shape (N,2) where each row are (x,y) coordinates
            of a location in the circuit.
        dmatrix (Tensor): An NxN matrix where the (i,j) element represents the
            distance from location i to location j. Optional
    """

    def __init__(self, dmatrix: Tensor, locs: Tensor):
        super(FixedTSPEnv, self).__init__()
        assert dmatrix.ndim == 2, "`dmatrix` must be NxN tensor"
        assert (
            dmatrix.shape[0] == dmatrix.shape[1]
        ), f"`dmatrix` must be NxN tensor {dmatrix.shape}"
        self.dmatrix = dmatrix
        assert locs.ndim == 2, "`locs` must have shape (N,2)"
        assert locs.shape[1] == 2, "`locs` must have shape (N,2)"
        self.locs = locs
        # set initial state
        self.state = State(locs, dmatrix, None, None)

    def step(self, action, kind):
        """Accept an action, update the state of the environment, output next state.

        Arguments:
            action (Tensor): a real-valued vector of length N (the same length
                as the number of locations). The argsort of this vector gives
                the indices of the locations in the order they should be visited.
        Returns:
            tuple of (state, reward, done, info)
        """
        if kind == "re-order":
            return self._step_reorder(action)
        else:
            raise ValueError("invalid action `kind`")

    def _step_reorder(self, action):
        # Turn the input action into a sequence of stops
        seq = action.squeeze().argsort()
        # calculate the distance traversed by this sequence
        d = self.dmatrix[seq, torch.roll(seq, -1, dims=0)].sum()
        self.state = State(self.locs, self.dmatrix, seq, d)
        self.reward = -d
        self.done = False
        self.info = dict()
        return self.state, self.reward, self.done, self.info

    def reset(self) -> State:
        """Reset the environment. Since the state is fixed, this doesn't
        actually do anything.
        """
        pass

    def render(self) -> plt.Figure:
        """Returns a plot of the locations (and the sequence of stops if
        available).
        """
        fig = plt.figure()
        locs = self.locs.cpu().numpy()
        plt.scatter(locs[:, 0], locs[:, 1], color="black")
        if self.state.seq is not None:
            seq = self.state.seq.cpu().numpy()
            seq = np.concatenate([seq, [seq[0]]])
            plt.plot(locs[seq, 0], locs[seq, 1], color="C0", alpha=0.75)
        if self.state.dist is not None:
            plt.title(f"TSP circuit (dist = {self.state.dist:.3f})")
        else:
            plt.title("TSP circuit")
        plt.xlabel("x-coord")
        plt.ylabel("y-coord")
        return fig

    def close(self):
        pass

    def seed(self):
        pass
