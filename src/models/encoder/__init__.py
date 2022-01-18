import torch.nn as nn
from .fcn import FCNEncoder

ACTIVATION = {
    'relu': nn.ReLU(),
    'lrelu': nn.LeakyReLU()
}


def init_encoder(architecture: str, **kwargs):
    if architecture == 'fcn':
        decoder = FCNEncoder(
            hidden_sizes=kwargs.get("hidden_sizes"),
            dim_input=kwargs.get("dim_input"),
            activation=ACTIVATION[kwargs.get('activation')]
        )

        hdim = kwargs.get("hidden_sizes")[-1]
    else:
        raise NotImplementedError

    return decoder, hdim
