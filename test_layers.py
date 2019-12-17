import numpy as np
import torch
import networkx as nx

from layers import GraphConvolution

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    # Create torch tensors
    A = torch.tensor(nx.adjacency_matrix(G).todense(), dtype=float)
    B = torch.tensor(nx.incidence_matrix(G, oriented=True).T.todense(), dtype=float)
    V = torch.randn((batch_size, G.number_of_nodes(), node_in), dtype=float)
    E = torch.randn((batch_size, G.number_of_edges(), edge_in), dtype=float)

    # Forward pass
    Vout, Eout = gconv(A, B, V, E)

    # Assert output shapes are correct
    assert Vout.shape == (batch_size, G.number_of_nodes(), node_out)
    assert Eout.shape == (batch_size, G.number_of_edges(), edge_out)
