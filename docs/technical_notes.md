# Technical notes

## Notation
Being a physicist, I perfer to use index notation and Einstein summation, so let's define that here.
Einstein summation basically means that if you see a repeated index, it is typically summed over, i.e.
$$ A_i B_i = \sum_i A_i B_i. $$
This is useful for complicated tensor contractions that you normally wouldn't be able to write as matrix multiplications, e.g.
$$ A_{ijk} B_{ii} C_{jk} $$
results in a scalar quantity that involves the trace over $B$, the transpose of $C$, and some other stuff.

In order to make dealing with tensors representing graph objects (or tensors defined on graph nodes/edges), I will introduce specific index notation to make things easier.
We consider a graph $G(\mathcal{V}, \mathcal{E})$, where $\mathcal{V}$ is the set of nodes $v$, and $\mathcal{E}$ is the set of edges $e$.
The cardinality of these sets will be $|\mathcal{V}| = n$ and $|\mathcal{E}| = m$, respectively, and we can refer to any node or edge by an integer index, i.e. $v_i$ or $e_j$.
Any graph-related indices will be denoted by Greek letters ($\alpha$, $\beta$, ...) and will be superscripted.
Indices into the "vertex space" will be denoted by regular Greek letters, and indices into the "edge space" will be noted by _dotted_ Greek letters ($\dot{\alpha}$, $\dot{\beta}$, ...).
Indices into the feature space, weight space, or internal representation space will be denoted with Roman letters ($i$, $j$, ...), and will always be subscripted.
We will suppress indices for the batch dimension, but do not forget that they exist and are attached to all feature "vectors" if using mini-batching.

Using the above notation will make it clear how certain tensors are structured.
We consider the following tensors: $V \in \mathbb{R}^{n \times f_v}$, $E \in \mathbb{R}^{m \times f_e}$, $A \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{m \times n}$, and $W \in \mathbb{R}^{f \times f^\prime}$, where $f_v$, $f_e$, $f$, and $f^\prime$ are the dimensions of the node/edge feature spaces (and can vary depending on the specifics of the neural network).
A node feature "vector" $V_i^\alpha$ has one index $\alpha$ into the node space and one index $i$ into the feature space.
An edge feature vector $E_i^{\dot{\beta}}$ has one index $\dot{\beta}$ into the edge space and one index $i$ into the feature space.
In the same way, we can denote the adjacency matrix (which maps nodes to their neigbors) by $A^{\alpha\beta}$ and the incidence matrix (which maps edges to their connected nodes) by $B^{\dot{\alpha}\beta}$.
Weight tensors/matrices $W_{ij}$ have two (or more) indices into the feature space.

## Graph feature propagation
Using the tensors defined above, it is simple to construct contractions between them with the appropriate free indices to create operations on $V$ and $E$ which can be repeatedly applied, as in the layers of a neural network.
The smallest number of tensors which can be used is two, but these do not contain any learnable parameters.
If we want to include learnable parameters, the smallest number of tensors involved is three.
There are three such possible contractions:
$$
A^{\alpha\beta} V^{\beta}_i W^{vv}_{ij}, \quad
E^{\dot{\alpha}}_i B^{\dot{\alpha}\beta} W^{ev}_{ij}, \quad
B^{\dot{\alpha}\beta} V^{\beta}_i W^{ve}_{ij}.
$$
The superscripts $vv$, $ev$ and $ve$ on $W$ are simply distinguishing labels, whose meaning will be made clear soon.
If we examine these contractions, we will see that the first two have two free indices - one in the node space and one in the feature space.
The third contraction has one free index in the edge space and one in the feature space.
This means that they have the same dimensions as the node and feature vectors, which suggests a layer-wise update rule
$$
(V^\alpha_i)^{(k)} = f^{(k)}\left( A^{\alpha\beta} (V^\beta_j)^{(k-1)} (W_{ji}^{vv})^{(k)} + B^{\dot{\alpha}\alpha} (E_j^{\dot{\alpha}})^{(k-1)} (W_{ji}^{ev})^{(k)} \right), \\
(E^{\dot{\alpha}}_i)^{(k)} = g^{(k)}\left( B^{\dot{\alpha}\beta} (V_j^\beta)^{(k-1)} (W_{ji}^{ve})^{(k)} \right),
$$
where $(k)$ denotes the neural network layer, and $f$ and $g$ are nonlinear activation functions like $tanh$ or $relu$.
Because the adjacency and incidence matrix act as local operators on the graph structure, these update rules lead to mixing of the features of adjacent nodes and edges.
In the analogy with regular convolutional networks (for images), this update rule is similar to a convolutional layer with a receptive field of one pixel.
If we want to generalize to greater "graph receptive fields", we will have to consider higher-order contractions.
For example, in order to take into account nodes that are two edges away would involve simply squaring the adjacency matrix: $A^{\alpha\beta} A^{\beta\gamma} V^{\gamma}_i W^{vv}_{ij}$.
The full update rule for a stride of two would be
$$
(V_i^\alpha)^{(k)} = f^{(k)}\left(\left(A^{\alpha\beta} A^{\beta\gamma} + B^{\dot{\alpha}\alpha} B^{\dot{\alpha}\gamma}\right) (V_j^\gamma)^{(k-1)} (W_{ji}^{vv})^{(k)}\right), \\
(E_i^{\dot{\alpha}})^{(k)} = g^{(k)}\left( A^{\alpha\beta} B^{\dot{\alpha}\alpha} (V_j^\beta)^{(k-1)} (W_{ji}^{ve})^{(k)} + B^{\dot{\alpha}\beta} B^{\dot{\beta}\beta} (E_j^{\dot{\beta}})^{(k-1)} (W_{ji}^{ee})^{(k)} \right)
$$
This can actually be simplified even further because the contraction of the incidence matrix with itself along the edge dimension is proportional to the adjacency matrix ($B^{\dot{\alpha}\alpha} B^{\dot{\alpha}\beta} \sim A^{\alpha\beta}$)