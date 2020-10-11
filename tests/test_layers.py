import numpy as np
import torch
import networkx as nx
from sklearn.decomposition import non_negative_factorization
import pytest

from neuraltsp.nn.layers import (
    NodeToNode,
    NodeToEdge,
    EdgeToNode,
    GraphConvolution,
    GraphEmbedding,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_node_to_node():
    # Define convolution layer
    node_in, node_out = np.random.randint(1, 10, size=2)
    gconv = NodeToNode(node_in, node_out)
    gconv.to(device)

    # Define graph
    num_nodes = 10
    p = 0.5
    batch_size = 4
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)

    # Create torch tensors
    A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=torch.float).to(device)
    V = torch.randn((batch_size, G.number_of_nodes(), node_in), dtype=torch.float).to(
        device
    )

    # Forward pass
    Vout = gconv(A, V)

    # Assert output shapes are correct
    assert Vout.shape == (batch_size, G.number_of_nodes(), node_out)


def test_node_to_edge():
    # Define convolution layer
    node_in, edge_out = np.random.randint(1, 10, size=2)
    gconv = NodeToEdge(node_in, edge_out)
    gconv.to(device)

    # Define graph
    num_nodes = 10
    p = 0.5
    batch_size = 4
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)

    # Create torch tensors
    B = torch.tensor(
        nx.incidence_matrix(G, oriented=True).T.todense(), dtype=torch.float
    ).to(device)
    V = torch.randn((batch_size, G.number_of_nodes(), node_in), dtype=torch.float).to(
        device
    )

    # Forward pass
    Eout = gconv(B, V)

    # Assert output shapes are correct
    assert Eout.shape == (batch_size, G.number_of_edges(), edge_out)


def test_edge_to_node():
    # Define convolution layer
    edge_in, node_out = np.random.randint(1, 10, size=2)
    gconv = EdgeToNode(edge_in, node_out)
    gconv.to(device)

    # Define graph
    num_nodes = 10
    p = 0.5
    batch_size = 4
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)

    # Create torch tensors
    B = torch.tensor(
        nx.incidence_matrix(G, oriented=True).T.todense(), dtype=torch.float
    ).to(device)
    E = torch.randn((batch_size, G.number_of_edges(), edge_in), dtype=torch.float).to(
        device
    )

    # Forward pass
    Vout = gconv(B, E)

    # Assert output shapes are correct
    assert Vout.shape == (batch_size, G.number_of_nodes(), node_out)


def test_graph_convolution():
    # Define convolution layer
    node_in, node_out = np.random.randint(1, 10, size=2)
    edge_in, edge_out = np.random.randint(1, 10, size=2)
    gconv = GraphConvolution(node_in, node_out, edge_in, edge_out)
    gconv.to(device)

    # Define graph
    num_nodes = 10
    p = 0.5
    batch_size = 4
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)
    A = nx.adjacency_matrix(G).todense()
    B = nx.incidence_matrix(G, oriented=True).T.todense()
    V_size = (batch_size, G.number_of_nodes(), node_in)
    E_size = (batch_size, G.number_of_edges(), edge_in)

    # Create torch tensors
    A = torch.tensor(A, dtype=float)
    B = torch.tensor(B, dtype=float)
    V = torch.randn(V_size, dtype=float)
    E = torch.randn(E_size, dtype=float)

    # Forward pass
    Vout, Eout = gconv(A, B, V, E)

    # Assert output shapes are correct
    assert Vout.shape == (batch_size, G.number_of_nodes(), node_out)
    assert Eout.shape == (batch_size, G.number_of_edges(), edge_out)


def test_graph_embedding():
    # Define graph
    num_nodes = 10
    p = 0.25
    batch_size = 4
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)
    A = nx.adjacency_matrix(G).todense()

    # Define embedding
    emb = GraphEmbedding()
    nodes_in, nodes_out = G.number_of_nodes(), 5
    W, H, _ = non_negative_factorization(A, n_components=nodes_out, init=None)

    # Create torch tensors
    num_features = 5
    V = torch.randn((batch_size, num_nodes, num_features), dtype=float)
    W = torch.tensor(W, dtype=float)

    # Run through embedding layer
    Vout = emb(V, W)
