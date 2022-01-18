import torch.nn as nn
from .fcn import FCNDecoder

ACTIVATION = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid()
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

    return decoder
