import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
from torch import Tensor
from itertools import product
from .base_uaq import UniformAffineQuantizer


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class PTQuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False):
        super(PTQuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
            
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
           

        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()

        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        # initialize quantizer

        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False
        self.is_last_first = False
        self.disable_act_quant = disable_act_quant

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

        # disable act quantization is designed for convolution before elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    @torch.jit.export
    def extra_repr(self):
        return 'wbit={}, abit={}, disable_act_quant={} is_first_or_last={}'.format(
            self.weight_quantizer.n_bits, self.act_quantizer.n_bits, self.disable_act_quant, self.is_last_first
        )



class PTQuantMatMul(nn.Module):
    def __init__(self, A_quant_params: dict = {},
                 B_quant_params: dict = {}, disable_act_quant: bool = False, quantizer: str = 'uaq'):
        super(PTQuantMatMul, self).__init__()
        self.A_quantizer = UniformAffineQuantizer(**A_quant_params)
        self.B_quantizer = UniformAffineQuantizer(**B_quant_params)
        self.use_matmul_quant = True

    def forward(self, A, B):
        if self.use_matmul_quant:
            A_sim = self.A_quantizer(A)
            B_sim = self.A_quantizer(B)
            out = A_sim @ B_sim
        else:
            out = A @ B
        return out

    def set_quant_state(self, use_matmul_quant: bool = False):
        self.use_matmul_quant = use_matmul_quant
