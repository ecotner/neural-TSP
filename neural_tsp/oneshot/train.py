import torch
import torch.optim as optim
import numpy as np

from neural_tsp.libs.environment import TSPEnvironment
from neural_tsp.oneshot.model import PolicyNetwork, QNetwork


def train(max_episodes: int, lr: float):
    # Create environment
    env = TSPEnvironment(num_locs_range=[5, 10], oneshot=True)
    # Initialize policy/value networks
    policy = PolicyNetwork()
    value = QNetwork()
    value.output_bias.data = torch.tensor(-20.0, dtype=torch.float)
    # Initialize optimizer
    val_optimizer = optim.SGD(params=value.parameters(), lr=lr, momentum=0.99)
    policy_optimizer = optim.SGD(params=policy.parameters(), lr=lr, momentum=0.99)
    # Put on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)
    value.to(device)
    # Log training loss, reward, etc.
    with open("train.log", "w+") as fp:
        fp.write("loss,reward,q_value")

    # Iterate over episodes
    # max_episodes = 100
    for episode in range(max_episodes):
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
            action = policy(A, V)
            Q = value(A, V, action)
            s, r, _, _ = env.step(action.data.cpu().numpy())
            # Update critic
            val_optimizer.zero_grad()
            loss = torch.pow(r - Q, 2)
            loss.backward(retain_graph=True)
            val_optimizer.step()
            loss_val = loss.data.cpu().numpy()
            # Update actor
            policy_optimizer.zero_grad()
            loss = -Q
            loss.backward()
            policy_optimizer.step()
            q_val = -loss.data.cpu().numpy()
            with open("train.log", "a") as fp:
                fp.write(f"\n{loss_val},{r},{q_val}")
        if episode % 100 == 99:
            print(
                f"Episode {episode+1}/{max_episodes}; "
                f"loss: {loss_val}; Q-value: {q_val}; "
                f"reward: {r}"
            )

        # print()
        # print("Input:\n", V)
        # print("Action:\n", action)
        # print("Adj. matrix:\n", A)
        # print("Q-value:\n", Q)

    # Save model
    torch.save(policy.state_dict(), "actor_model.torch")
    torch.save(value.state_dict(), "critic_model.torch")
