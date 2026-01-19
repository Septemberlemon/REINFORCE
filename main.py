import torch
from torch.distributions import Categorical
import wandb
import gymnasium as gym
import numpy as np

from model import Policy, ValueNetwork


test_name = "test4"
episodes = 5000
gamma = 0.99
learning_rate = 0.001

env = gym.make("Acrobot-v1")

policy = Policy(6, 3).to("cuda")
# policy.load_state_dict(torch.load("checkpoints/test1.pth"))
policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
value_network = ValueNetwork(6).to("cuda")
value_optimizer = torch.optim.Adam(value_network.parameters(), lr=learning_rate)

wandb.init(project="Acrobot-v1", name=test_name, mode="online")

for episode in range(episodes):
    obs, info = env.reset()
    log_probs = []
    rewards = []
    steps = 0
    observations = [obs]
    while True:
        logits = policy(torch.tensor(obs).to("cuda"))
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        obs, reward, terminated, truncated, info = env.step(action.item())
        reward = 1 if terminated and not truncated else float(reward) * 0.01
        rewards.append(reward)
        observations.append(obs)

        steps += 1

        if terminated or truncated:
            break

    print(f"Episode: {episode}, steps: {steps}, rewards: {sum(rewards)}")
    G_t = 0
    returns = []
    for reward in rewards[::-1]:
        G_t *= gamma
        G_t += reward
        returns.insert(0, G_t)

    returns = torch.tensor(returns).to("cuda")
    baselines = value_network(torch.tensor(np.stack(observations[:-1])).to("cuda"))
    policy_loss = -((returns - baselines.detach()) * torch.stack(log_probs)).sum()
    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
    policy_optimizer.step()

    value_loss = torch.nn.functional.mse_loss(returns, baselines)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    wandb.log({
        "steps": steps,
        "rewards": sum(rewards),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    })

torch.save(policy.state_dict(), f"checkpoints/{test_name}_policy.pth")
torch.save(value_network.state_dict(), f"checkpoints/{test_name}_value.pth")
env.close()
wandb.finish()
