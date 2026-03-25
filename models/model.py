import torch
import torch.nn as nn

class SchedulerNet(nn.Module):
    def __init__(self, input_size=30, output_size=10):
        super(SchedulerNet, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

