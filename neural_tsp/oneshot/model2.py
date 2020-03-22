import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_tsp.libs.layers import NodeToNode


class Actor(nn.Module):
    """Network representing an agent's policy"""

    def __init__(self):
        super(type(self), self).__init__()
        self.conv1 = NodeToNode(node_in=2, node_out=10)
        self.conv2 = NodeToNode(node_in=10, node_out=10)
        self.conv3 = NodeToNode(node_in=10, node_out=1)

    def forward(self, A, V):
        V = F.relu(self.conv1(A, V))
        V = F.relu(self.conv2(A, V))
        V = self.conv3(A, V)
        return V


class Critic(nn.Module):
    """Network representing the Q-value under an agent's policy"""

    def __init__(self):
        super(type(self), self).__init__()
        self.conv1 = NodeToNode(node_in=3, node_out=10)
        self.conv2 = NodeToNode(node_in=10, node_out=10)
        self.conv3 = NodeToNode(node_in=10, node_out=1)
        self.output_bias = nn.Parameter(
            torch.tensor(0.0, dtype=torch.float), requires_grad=True
        )

    def forward(self, A, V, action):
        V = torch.cat([V, action], dim=2)
        V = F.relu(self.conv1(A, V))
        V = F.relu(self.conv2(A, V))
        V = self.conv3(A, V)
        return -torch.sum(V) + self.output_bias
