from typing import List
import torch
import torch.nn as nn


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.5)


class FCNDecoder(nn.Module):
    def __init__(self, hidden_sizes: List[int], dim_input: int, activation, last_activation):
        super().__init__()

        hidden_sizes = [dim_input] + list(hidden_sizes)
        self.net = []

        for i in range(len(hidden_sizes) - 1):
            self.net.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))

            if i < len(hidden_sizes) - 2:
                self.net.append(activation)
            else:
                self.net.append(last_activation)

        self.net = nn.Sequential(*self.net)

    def forward(self, z: torch.Tensor):
        return self.net(z)

    def init_weight(self):
        self.net.apply(init_weight)
