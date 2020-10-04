"""Custom neural network layers for graph convolutions
"""

import torch
import torch.nn as nn


class NodeToNode(nn.Module):
    """Graph convolution layer over nodes"""

    def __init__(self, node_in: int, node_out: int):
        super(type(self), self).__init__()
        self.W = nn.Parameter(
            torch.randn((node_in, node_out)).type(torch.float), requires_grad=True,
        )
        self.b = nn.Parameter(
            torch.zeros(size=(1, 1, node_out)).type(torch.float), requires_grad=True
        )
        nn.init.xavier_uniform_(self.W)

    def forward(self, A, V):
        """Computes forward pass of graph convolution

        Arguments:
            A [Tensor]: Adjacency matrix of the graph. Acts as a diffusion
                operator on the space of nodes. Is a square matrix with
                dimensions of (|V|, |V|).
            V [Tensor]: Node feature matrix. Must have dimensions of
                (batch, |V|, node_in).
        Returns:
            Tensor: Tensor representing the output node representations. The
                tensor has dimensions (batch, |V|, node_out).
        """
        # FIXME: use of torch.einsum is supposed to be really slow; try to
        # replace with matmul or tensordot if possible
        Vout = torch.einsum("ab,xai,ij->xbj", A, V, self.W) + self.b
        return Vout


class NodeToEdge(nn.Module):
    """Layer that transforms node weights to edge weights.
    """

    def __init__(self, node_in, edge_out):
        super(type(self), self).__init__()
        self.W = torch.randn((node_in, edge_out), requires_grad=True, dtype=float)
        self.b = torch.zeros(size=(1, 1, edge_out), requires_grad=True, dtype=float)
        nn.init.xavier_uniform_(self.W)

    def forward(self, B, V):
        """Computes forward pass of graph convolution

        Arguments:
            B [Tensor]: Incidence matrix of the graph. Acts as a diffusion
                operator between the node/edge space. Is a matrix with
                dimensions of (|E|, |V|).
            V [Tensor]: Node feature matrix. Must have dimensions of
                (batch, |V|, node_in).
        Returns:
            Tensor: Tensor representing the output node representations. The
                tensor has dimensions (batch, |E|, edge_out).
        """
        # FIXME: use of torch.einsum is supposed to be really slow; try to
        # replace with matmul or tensordot if possible
        Eout = torch.einsum("ab,xbi,ij->xaj", B, V, self.W) + self.b
        return Eout


class EdgeToNode(nn.Module):
    pass


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
        super(type(self), self).__init__()
        self.Wvv = torch.randn(  # Node to node
            (node_in, node_out), requires_grad=True, dtype=float
        )
        self.Wev = torch.randn(  # Edge to node
            (edge_in, node_out), requires_grad=True, dtype=float
        )
        self.Wve = torch.randn(  # Node to edge
            (node_in, edge_out), requires_grad=True, dtype=float
        )
        self.bv = torch.zeros(  # Node bias
            (1, 1, node_out), requires_grad=True, dtype=float
        )
        self.be = torch.zeros(  # Edge bias
            (1, 1, edge_out), requires_grad=True, dtype=float
        )

        self.weights = {"Wvv": self.Wvv, "Wev": self.Wev, "Wve": self.Wve}
        self.biases = {"bv": self.bv, "be": self.be}
        for W in self.weights.values():
            nn.init.xavier_uniform_(W)

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


class GraphEmbedding(nn.Module):
    """Graph embedding layer"""

    def __init__(self):
        super(type(self), self).__init__()

    def forward(self, V, E):
        """
        Embeds a graph with node feature tensor V into an embedding space
        with a linear transformation determined by E. This layer does not
        determine the embedding for you, it must be supplied.
        
        Arguments:
            V [tensor]: Node feature tensor of dimensions (batch, nodes, features)
            E [tensor]: Embedding matrix of dimensions (nodes, embedding_dim)
        Returns:
            tensor: New node tensor transformed into embedding space.
        """
        Vout = torch.einsum("xai,ab->xbi", V, E)
        return Vout
