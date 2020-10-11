# References

## Traveling salesman problem

## Deep learning techniques
* Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift; S. Ioffe, C. Szegedy; 2015; [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

## Graph convolutional networks
* A Comprehensive Survey on Graph Neural Networks; Z. Wu et. al.; 2019; [arXiv:1901.00596](https://arxiv.org/abs/1901.00596)
	* Like the title says, a comprehensive survey on graph-NN
* Spectral Networks and Deep Locally Connected Networks on Graphs; J. Bruna, W. Zaremba, A. Szlam, Y. LeCun; 2014; [arXiv:1312.6203](https://arxiv.org/abs/1312.6203)
	* Original paper on graph-CNN?
* Graph Convolutional Networks; T. Kipf; 2016; [blog post](https://tkipf.github.io/graph-convolutional-networks/)
	* Useful blog post about how to design graph neural networks
* Neural Message Passing for Quantum Chemistry; J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, G. E. Dahl; 2017; [arXiv:1704.01212](https://arxiv.org/pdf/1704.01212.pdf)
    * Uses message-passing to update node representations in a graph, formalizes a very general framework
* Weighted Graph Cuts without Eigenvectors: A Multilevel Approach; I. S. Dhillon, Y. Guan, B. Kulis; [IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE, VOL. 29, NO. 1](http://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf)
	* Fast graph pooling technique that is similar to spectral pooling but uses weighted k-means (Graclus algorithm)
* `torch_geometric`; [website](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html)

## Reinforcement learning
* Continuous control with deep reinforcement learning; T. Lillicrap et. al.; ICLR 2016; [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
	* Deep Deterministic Policy Gradient (DDPG) algorithm
* Policy invariance under reward transformations: Theory and applications to reward shaping; A. Y. Ng, D. Harada, S. Russel; 1999; [link](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
	* Introduction of "reward potentials"
* Playing Atari with Deep Reinforcement Learning; V. Mnih et. al.; Nature 2015; [link](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
	* Introduction of Deep Q-Network (DQN), using memory replay and a target network
* Attention, Learn to Solve Routing Problems!; W. Kook, H. van Hoof, M. Welling; [link](https://arxiv.org/abs/1803.08475)
	* Solves many types of routing problems (TSP, VRP, others) using a multi-headed attention model with graph neural networks