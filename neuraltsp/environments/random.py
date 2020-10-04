"""Gym environment for the TSP
"""

### TODO: refactor FixedTSPEnv so that is is more like a base class
### that RandomTSPEnv can inherit from

from typing import Iterable, Union, Optional
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


class RandomTSPEnv(gym.Env):
    """
    Environment representing the Traveling Salesman Problem (TSP) which will
    randomly generate new problems on reset.
    
    Parameters:
        n_locs (int, list): An integer or sequence of integers denoting how
            many locations to generate.
        p_locs (None, list): If `n_locs` is a sequence, then you can optionally
            provide a sequence of positive weights to sample from `n_locs` with.
        device (torch.device): The device to place the generated tensors on
    """

    def __init__(
        self,
        n_locs: Union[int, Iterable[int]],
        p_locs: Optional[Iterable[int]] = None,
        device=None,
    ):
        super(RandomTSPEnv, self).__init__()
        # convert n_locs to an array
        if isinstance(n_locs, int):
            n_locs = [n_locs]
        n_locs = np.array(n_locs)
        assert n_locs.ndim == 1
        self.n_locs = n_locs
        # create p_locs is None
        if p_locs is None:
            p_locs = np.ones(len(n_locs))
        p_locs = np.array(p_locs)
        p_locs /= p_locs.sum()
        self.p_locs = p_locs
        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.reset()

    def step(self, action, kind):
        """Accept an action, update the state of the environment, output next state.
        """
        if kind == "re-order":
            return self._step_reorder(action)
        elif kind == "swap":
            return self._step_swap(action)
        else:
            raise ValueError("invalid action `kind`")

    def _step_swap(self, action):
        old_seq = self.state.seq
        seq = torch.clone(old_seq)
        seq[action[0]] = old_seq[action[1]]
        seq[action[1]] = old_seq[action[0]]
        d = self.dmatrix[seq, torch.roll(seq, -1, dims=0)].sum()
        self.state = State(self.locs, self.dmatrix, seq, d)
        self.reward = -d
        self.done = True
        self.info = dict()
        return self.state, self.reward, self.done, self.info

    def _step_reorder(self, seq):
        """
        Arguments:
            action (Tensor): a real-valued vector of length N (the same length
                as the number of locations). The argsort of this vector gives
                the indices of the locations in the order they should be visited.
        Returns:
            tuple of (state, reward, done, info)
        """
        assert seq.dtype == torch.long
        # calculate the distance traversed by this sequence
        d = self.dmatrix[seq, torch.roll(seq, -1, dims=0)].sum()
        self.state = State(self.locs, self.dmatrix, seq, d)
        self.reward = -d
        self.done = True
        self.info = dict()
        return self.state, self.reward, self.done, self.info

    def reset(self) -> State:
        """Reset the environment. Generates a new set of stops.
        """
        N = np.random.choice(self.n_locs, p=self.p_locs)
        self.locs = torch.randn(N, 2).to(self.device)
        self.dmatrix = (
            (self.locs.view(N, 1, 2) - self.locs.view(1, N, 2))
            .pow(2.0)
            .sum(dim=-1)
            .sqrt()
            .to(self.device)
        )
        self.state = State(self.locs, self.dmatrix, None, None)
        return self.state

    def render(self, number_points=False) -> plt.Figure:
        """Returns a plot of the locations (and the sequence of stops if
        available).
        """
        fig = plt.figure()
        locs = self.locs.cpu().numpy()
        if number_points:
            for i in range(len(locs)):
                marker = f"${{{str(i)}}}$"
                plt.scatter(locs[i, 0], locs[i, 1], color="black", marker=marker)
        else:
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
