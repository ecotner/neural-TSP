# neural-TSP
Use neural networks to approximately solve the traveling salesman problem

## Methods
* [Simplistic actor-critic](notebooks/training/01_actor_critic.ipynb): A naive actor critic model with a bandit-like environment.
* [Randomized environment](notebooks/training/02_actor_critic_random_env.ipynb): Same actor-critic model as before, but now the bandit environment is randomized between each episode.
* [2D action space and edge convolutions](notebooks/training/03_edge_features.ipynb): Trying a different representation for the action space where the visitation order is given by the angle around a reference point, and adding in convolutions over graph edges so that the distance matrix can be passed to the neural networks as edge features.