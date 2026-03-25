from sim.env import RANEnvironment
from utils.preprocess import preprocess
from agents.ppo_agent import PPOAgent
from utils.logger import Logger

import torch

def main():
    print("🔥 Week 3: PPO Training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = RANEnvironment(num_users=10)
    agent = PPOAgent().to(device)

    logger = Logger()

    epochs = 500
    avg_reward = 0

    for epoch in range(epochs):
        state = env.reset()
        state = preprocess(state)

        state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

        probs, value = agent(state_tensor)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        action_onehot = torch.zeros_like(probs)
        action_onehot[action] = 1.0

        reward = env._compute_reward(
            action_onehot.detach().cpu().numpy(), 
            env.reset()
        )

        advantage = reward - value.item()

        actor_loss = -dist.log_prob(action) * advantage
        critic_loss = advantage ** 2

        loss = actor_loss + 0.5 * critic_loss

        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # smoothing
        if epoch == 0:
            avg_reward = reward
        else:
            avg_reward = 0.9 * avg_reward + 0.1 * reward

        if epoch % 50 == 0:
            msg = f"Epoch {epoch} | Reward: {reward:.4f} | Avg: {avg_reward:.4f}"
            logger.log(msg)

    torch.save(agent.state_dict(), "checkpoints/ppo_model.pth")
    print("✅ Model saved")


if __name__ == "__main__":
    main()
