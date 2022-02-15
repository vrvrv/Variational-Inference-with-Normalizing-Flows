import torch
import torch.nn as nn


def recon_loss_fn(distribution: str, px=None, pz=None):
    if distribution == 'normal':
        def recon_loss(x, xhat):
            return torch.square(x - xhat).sum(list(range(x.dim()))[1:]) / 2

    elif distribution == 'bernoulli':
        def recon_loss(x, xhat):
            return (- x * torch.log(xhat + 1e-8) - (1 - x) * torch.log(1 - xhat + 1e-8)).sum(list(range(x.dim()))[1:])

    elif distribution == 'glmm':
        class recon_loss(nn.Module):
            def __init__(self):
                super().__init__()
                self.px = px
                self.pz = pz
                self.fe = nn.Linear(px, 1, bias=False)

            def forward(self, xzy, u):
                x = xzy[..., :self.px]
                z = xzy[..., self.px:self.px + self.pz]
                y = xzy[..., self.px + self.pz:]

                linear = torch.clip(self.fe(x) + torch.sum(z * u, dim=1, keepdim=True), min=-10, max=10)

                nll = - (y * linear - torch.exp(linear) - torch.lgamma(y + 1.0)).sum(-1)
                return nll

    else:
        raise NotImplementedError

    return recon_loss
