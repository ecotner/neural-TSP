"""Custom neural network layers
"""

import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """Generalized graph convolution layer.

    Computes a convolution over an entire graph, including both node and edge
    weights.
    
    Arguments:
        node_in [int]: dimension of input node features
        node_out [int]: dimension of output node features
        edge_in [int]: dimension of input edge features
        edge_out [int]: dimension of output edge features
    """

    def __init__(self, node_in: int, node_out: int, edge_in: int, edge_out: int):
        super(GraphConvolution, self).__init__()
        self.Wvv = torch.randn(  # Node to node
            (node_in, node_out), requires_grad=True, dtype=float
        )
        self.Wev = torch.randn(  # Edge to node
            (edge_in, node_out), requires_grad=True, dtype=float
        )
        self.Wve = torch.randn(  # Node to edge
            (node_in, edge_out), requires_grad=True, dtype=float
        )
        self.bv = torch.randn(  # Node bias
            (1, 1, node_out), requires_grad=True, dtype=float
        )
        self.be = torch.randn(  # Edge bias
            (1, 1, edge_out), requires_grad=True, dtype=float
        )

        self.weights = {"Wvv": self.Wvv, "Wev": self.Wev, "Wve": self.Wve}
        self.biases = {"bv": self.bv, "be": self.be}
        for W in self.weights.values():
            nn.init.xavier_uniform_(W)
        for b in self.biases.values():
            nn.init.xavier_uniform_(b)

    def forward(self, A, B, V, E):
        """Computes forward pass of graph convolution

        This is a two-part generalized convolution, where we separately
        compute the node/edge representations of the next layer based on the
        node/edge representations of the previous layer.

        Arguments:
            A [Tensor]: Adjacency matrix of the graph. Acts as a diffusion
                operator on the space of nodes. Is a square matrix with
                dimensions of (|V|, |V|).
            B [Tensor]: Incidence matrix of the graph. Acts as a gradient
                operator on the space of nodes. It is a matrix with dimensions 
                (|E|, |V|).
            V [Tensor]: Node feature matrix. Must have dimensions of
                (batch, |V|, node_in).
            E [Tensor]: Edge feature matrix. Must have dimensions of
                (batch, |E|, edge_in)
        Returns:
            (Tensor, Tensor): Two tensors representing the output node (0th
                element) and edge (1th element) representations. The node
                tensor has dimensions (batch, |V|, node_out), and the edge
                tensor has dimensions (batch, |E|, edge_out).
        """
        # FIXME: use of torch.einsum is supposed to be really slow; try to
        # replace with matmul or tensordot if possible
        Vvv = torch.einsum("ba,xai,ij->xbj", A, V, self.Wvv)
        Vev = torch.einsum("xai,ab,ij->xbj", E, B, self.Wev)
        Vout = Vvv + Vev + self.bv
        Eout = torch.einsum("ab,xbi,ij->xaj", B, V, self.Wve) + self.be
        return Vout, Eout
