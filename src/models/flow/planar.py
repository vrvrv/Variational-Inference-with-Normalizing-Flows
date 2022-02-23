import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.flow import ACTIVATION_DERIVATIVES


class PlanarFlow(nn.Module):
    def __init__(self, D, activation=torch.tanh):
        super().__init__()
        self.D = D
        self.w = nn.Parameter(torch.Tensor(1, D))
        self.b = nn.Parameter(torch.Tensor(1))
        self.u = nn.Parameter(torch.Tensor(1, D))
        self.activation = activation
        self.activation_derivative = ACTIVATION_DERIVATIVES[activation]

        nn.init.xavier_uniform_(self.w, gain=0.5)
        nn.init.xavier_uniform_(self.u, gain=0.5)
        nn.init.normal_(self.b)

    def forward(self, z: torch.Tensor):
        lin = F.linear(z, self.w, self.b) # shape: (B, 1)
        f = z + self.u * self.activation(lin)  # shape: (B, D)
        phi = self.activation_derivative(lin) * self.w  # shape: (B, D)
        log_det = torch.log(torch.abs(torch.mm(phi, self.u.t()).add_(1.0)).clamp_(min=1e-8)) # shape: (B,)
        return f, log_det
