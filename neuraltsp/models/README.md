# One-shot solution
This algorithm will try to approximately solve the traveling salesman problem with a single forward pass of a graph convolutional neural network. The graph in question will be described by the distance matrix between locations.

The problem is formulated as a reinforcement learning task, where the reward is specified by the negative of the cost to traverse a given tour.
