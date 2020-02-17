import numpy as np

from neural_tsp.libs.environment import TSPEnvironment


def test_environment():
    env = TSPEnvironment(num_locs_range=[5, 10])
    num_locs = env.num_locs
    action = np.random.choice(np.arange(num_locs), replace=False, size=num_locs)
    state, reward, done, info = env.step(action)
    assert state.locs.shape == (num_locs, 2)
    assert all(state.order == np.argsort(action))
    assert state.dist_matrix.shape == (num_locs, num_locs)
    assert state.dist == -reward
