import os
import sys
sys.path.append('/mnt/1c714c0e-2ec6-4b54-b86f-8ab77eaea676/study/code/HOI/My_Work/OmniHOI/GitHOI')


import torch
from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# import torch.utils.checkpoint as cp

from typing import Optional, Sequence, Union
from pyrutils.torch.general import pick_activation_function, MySequential
 
def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k

class InteractionAttention(nn.Module):
    def __init__(self, dims, activations: Optional[Sequence[Union[str, dict]]] = None,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        layers = []
        for dim_in, dim_out, activation in zip(dims[:-1], dims[1:], activations):
            layers.append(nn.Linear(dim_in,dim_out,bias=bias))
            layers.append(pick_activation_function(activation))
            layers.append(nn.Dropout(dropout))
        self.proj_mlp = nn.Sequential(*layers)

class IIM(nn.Module):
    def __init__(self, input_size: int, hidden_size, output_size: int, layer_num:int, activations: Optional[Sequence[Union[str, dict]]] = None,
                 num_heads: int =8, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.proj_mlp = nn.Linear(input_size, hidden_size, bias=bias)
        self.iim_layers = build_iim(hidden_size, layer_num, activations, num_heads, dropout, bias)
        self.out_proj = nn.Linear(layer_num*hidden_size, output_size, bias=bias)

    def forward(self, input):
        # def iim_fn(x):
        input = F.relu(self.proj_mlp(input))
        output_state = []
        for iim_layer in self.iim_layers:
            input = iim_layer(input)
            output_state.append(input)
        output_state = torch.cat(output_state, dim=-1)
        # output_state = cp.checkpoint(self.out_proj, output_state, use_reentrant=False)
        out_message = F.relu(self.out_proj(output_state))
        return out_message

class II_Layer(nn.Module):
    def __init__(self, dim, num_heads=8, activation = 'relu', bias = True, dropout: float = 0.0):
        super().__init__()
        self.activation = pick_activation_function(activation)
        self.att_r = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, bias=bias)
        self.att_mlp_r = nn.Linear(dim, dim, bias=bias)
        # self.ln_r = nn.LayerNorm((dim,))
    def forward(self, input):
        att_rs = self.att_r(query = input, key = input, value = input)[0]
        # att_rs = cp.checkpoint(att_fn, input, use_reentrant=False)
        rs = self.activation(self.att_mlp_r(att_rs)) + input
        # return self.ln_r(rs)
        return rs

def build_iim(dim: int, layer_num: int, activations: Optional[Sequence[Union[str, dict]]] = None,
              num_heads: int =8, dropout: float = 0.0, bias: bool = True):
    """
    Build a gcn.
    """
    if activations is None:
        activations = ['identity'] * layer_num
    if layer_num != len(activations):
        raise ValueError('Number of activations must be the same as the number of dimensions - 1.')
    layers = []
    for i in range(layer_num):
        layers.append(II_Layer(dim, num_heads, activations[i], bias=bias, dropout=dropout))
    return nn.ModuleList(layers)

