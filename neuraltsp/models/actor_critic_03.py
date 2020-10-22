""" Actor-critic RL model

This model has a number of improvements over the naive first one that was used
for the first two experiments:
    * uses a 2D representation of actions rather than 1D
        * the idea is basically to map each node into the "complex plane", then
          sort them by their arguments (i.e. angle around the origin)
        * this representation is invariant to the "starting point" of the tour
    * convolutional layers act over both node _and_ edge space, rather than
      just nodes
        * since there is no "correct" adjacency matrix, the model can learn its
          own (and it can be multi-dimensional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuraltsp.nn.layers import NodeToNode, NodeToEdge, GraphConvolution, EdgeToNode


class Actor(nn.Module):
    """Network representing an agent's policy"""

    def __init__(self):
        super(type(self), self).__init__()
        self.gconv1 = GraphConvolution(node_in=2, node_out=10, edge_in=1, edge_out=10)
        # self.gconv2 = GraphConvolution(node_in=10, node_out=10, edge_in=10, edge_out=10)
        self.n2n3 = NodeToNode(node_in=10, node_out=2)
        self.e2n3 = EdgeToNode(edge_in=10, node_out=2)

    def forward(self, A, B, V, E):
        V, E = [F.relu(X) for X in self.gconv1(A, B, V, E)]
        # V, E = F.relu(self.gconv2(A, B, V, E))
        V = self.n2n3(A, V) + self.e2n3(B, E)
        return V


class Critic(nn.Module):
    """Network representing the Q-value under an agent's policy"""

    def __init__(self):
        super(type(self), self).__init__()
        self.gconv1 = GraphConvolution(node_in=4, node_out=10, edge_in=1, edge_out=10)
        # self.gconv2 = GraphConvolution(node_in=10, node_out=10, edge_in=10, edge_out=10)
        self.n2n3 = NodeToNode(node_in=10, node_out=2)
        self.e2n3 = EdgeToNode(edge_in=10, node_out=2)

    def forward(self, A, B, V, E, a):
        V = torch.cat([V, a], dim=-1)
        V, E = [F.relu(X) for X in self.gconv1(A, B, V, E)]
        # V, E = F.relu(self.gconv2(A, B, V, E))
        V = self.n2n3(A, V) + self.e2n3(B, E)
        return torch.sum(V)
