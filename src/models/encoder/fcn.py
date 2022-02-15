from typing import List
import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.5)


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

    def init_weight(self):
        self.net.apply(init_weight)
