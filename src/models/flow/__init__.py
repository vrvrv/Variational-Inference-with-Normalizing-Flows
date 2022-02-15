from typing import List
import torch
import torch.nn as nn


def dtanh(x):
    return 1 - torch.tanh(x) ** 2


ACTIVATION_DERIVATIVES = {
    torch.tanh: dtanh
}

from .planar import PlanarFlow
from .radial import RadialFlow


def init_flow(D: int, flow_seq: List[str], act_seq: List[str]):
    flowmodel = nn.ModuleList()

    assert len(flow_seq) == len(act_seq), f"length of flow_seq ({len(flow_seq)}) doesn't match with length of act_seq ({len(act_seq)})"
    for flow, act in zip(flow_seq, act_seq):

        if act == 'tanh':
            act_fn = torch.tanh
        else:
            raise NotImplementedError

        if flow == 'planar':
            f = PlanarFlow(D=D, activation=act_fn)
        elif flow == 'radial':
            f = RadialFlow(D=D, activation=act_fn)
        else:
            raise NotImplementedError

        flowmodel.append(f)

    return flowmodel
