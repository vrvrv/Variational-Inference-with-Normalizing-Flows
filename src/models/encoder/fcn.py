from typing import List
import torch.nn as nn


class FCNEncoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation=nn.ReLU()):
        super().__init__()

        hidden_sizes = [dim_input] + list(hidden_sizes)

        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.net.append(activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)
