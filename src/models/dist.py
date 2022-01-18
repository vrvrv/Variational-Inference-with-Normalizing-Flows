import torch


def recon_loss_fn(distribution: str):
    if distribution == 'normal':
        def recon_loss(x, xhat):
            return torch.square(x - xhat).sum(list(range(x.dim()))[1:]) / 2

    elif distribution == 'bernoulli':
        def recon_loss(x, xhat):
            return (- x * torch.log(xhat+1e-8) - (1 - x) * torch.log(1-xhat+1e-8)).sum(list(range(x.dim()))[1:])
    else:
        raise NotImplementedError

    return recon_loss
