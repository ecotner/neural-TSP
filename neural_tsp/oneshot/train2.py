import datetime as dt

import torch
import torch.optim as optim
import numpy as np

from neural_tsp.libs.environment import TSPEnvironment
from neural_tsp.oneshot.model import Actor, Critic


def train(max_episodes: int, lr: float):
    # Create environment
    env = TSPEnvironment(num_locs_range=[5, 10], oneshot=True)
    # Initialize policy/value networks
    actor = Actor()
    critic = Critic()
    critic.output_bias.data = torch.tensor(-20.0, dtype=torch.float)
    # Initialize optimizer
    val_optimizer = optim.SGD(params=critic.parameters(), lr=lr, momentum=0.99)
    actor_optimizer = optim.SGD(params=actor.parameters(), lr=lr, momentum=0.99)
    # Put on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor.to(device)
    critic.to(device)
    # Log training loss, reward, etc.
    with open("train.log", "w+") as fp:
        fp.write("loss,reward,q_value")

    # Iterate over episodes
    # max_episodes = 100
    # for episode in range(max_episodes):
    stop_time = dt.datetime.strptime("2020-03-02 08:00:00", "%Y-%m-%d %H:%M:%S")
    episode = 0
    while dt.datetime.now() < stop_time:
        s = env.reset()
        A = 1 / (
            torch.from_numpy(s.dist_matrix.astype(float))
            + torch.eye(len(s.locs), dtype=float)
        ) - torch.eye(len(s.locs), dtype=float)
        A = A / torch.max(torch.abs(A))
        V = torch.from_numpy(s.locs.reshape(1, -1, 2).astype(float))
        A = A.type(torch.float).to(device)
        V = V.type(torch.float).to(device)
        for _ in range(10):
            # Take action and update state
            action = actor(A, V)
            if np.random.rand() < 0.05:  # Add random noise occasionally
                action += torch.randn(action.shape, dtype=torch.float)
            Q = critic(A, V, action)
            s, r, _, _ = env.step(action.data.cpu().numpy())
            # Update actor
            actor_optimizer.zero_grad()
            loss = -Q
            loss.backward(retain_graph=True)
            actor_optimizer.step()
            q_val = -loss.data.cpu().numpy()
            # Update critic
            val_optimizer.zero_grad()
            loss = torch.pow(r - Q, 2)
            loss.backward()
            val_optimizer.step()
            loss_val = loss.data.cpu().numpy()
        with open("train.log", "a") as fp:
            fp.write(f"\n{loss_val},{r},{q_val}")
        if episode % 100 == 99:
            print(
                f"Episode {episode+1}; "
                f"loss: {loss_val}; Q-value: {q_val}; "
                f"reward: {r}"
            )
        episode += 1

        # print()
        # print("Input:\n", V)
        # print("Action:\n", action)
        # print("Adj. matrix:\n", A)
        # print("Q-value:\n", Q)

    # Save model
    torch.save(actor.state_dict(), "actor_model.torch")
    torch.save(critic.state_dict(), "critic_model.torch")
