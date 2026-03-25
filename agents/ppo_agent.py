import torch
import torch.nn as nn
import torch.optim as optim

class PPOAgent(nn.Module):
    def __init__(self, input_dim=30, action_dim=10):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=3e-4)

    def forward(self, x):
        probs = self.actor(x)
        value = self.critic(x)
        return probs, value
