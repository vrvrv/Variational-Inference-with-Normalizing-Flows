import torch.nn as nn
from .fcn import FCNDecoder

ACTIVATION = {
    'relu': nn.ReLU(),
    'elu': nn.ELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'id': nn.Identity(),
    'softplus': nn.Softplus()
}


def init_decoder(architecture: str, **kwargs):
    if architecture == 'fcn':
        decoder = FCNDecoder(
            hidden_sizes=kwargs.get("hidden_sizes"),
            dim_input=kwargs.get("dim_input"),
            activation=ACTIVATION[kwargs.get('activation')],
            last_activation=ACTIVATION[kwargs.get('last_activation')]
        )
    else:
        raise NotImplementedError

    decoder.init_weight()

    return decoder
