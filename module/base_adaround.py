import torch
from torch import nn
from typing import Union
from .base_uaq import UniformAffineQuantizer, round_ste


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(
        self,
        quntizer: UniformAffineQuantizer,
        weight_tensor: torch.Tensor,
        round_mode="learned_round_sigmoid",
        large_range=False,
    ):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = quntizer.n_bits
        self.sym = quntizer.sym
        self.large_range = large_range
        if isinstance(quntizer, UniformAffineQuantizer):
            self.quntizer = "uaq"
            self.leaf_param = quntizer.leaf_param
            self.delta_set = quntizer.delta_set
            self.zero_point_set = quntizer.zero_point_set
            self.n_levels = quntizer.n_levels
        else:
            raise NotImplementedError

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2 / 3
        self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.quntizer == "uaq":
            if self.round_mode == "nearest":
                x_int = torch.round(x / self.delta_set[self.n_bits - 2])  
            elif self.round_mode == "nearest_ste":
                x_int = round_ste(x / self.delta_set[self.n_bits - 2])
            elif self.round_mode == "stochastic":
                x_floor = torch.floor(x / self.delta_set[self.n_bits - 2])
                rest = (
                    x / self.delta_set[self.n_bits - 2]
                ) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print("Draw stochastic sample")
            elif self.round_mode == "learned_hard_sigmoid":
                x_floor = torch.floor(x / self.delta_set[self.n_bits - 2])
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    
                    x_int = x_floor + (self.alpha >= 0).float()
                   
            else:
                raise ValueError("Wrong rounding mode")

            x_quant = torch.clamp(
                x_int + self.zero_point_set[self.n_bits - 2], 0, self.n_levels - 1
            )
            x_float_q = (
                x_quant - self.zero_point_set[self.n_bits - 2]
            ) * self.delta_set[self.n_bits - 2]

        else:
            raise NotImplementedError
        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(
            torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1
        )  

    def init_alpha(self, x: torch.Tensor):  
        if self.quntizer == "lsq":
            raise NotImplementedError
        elif self.quntizer == "uaq":

            x_floor = torch.floor(x / self.delta_set[self.n_bits - 2])
            if self.round_mode == "learned_hard_sigmoid":
                print("Init alpha to be FP32")
                rest = (
                    x / self.delta_set[self.n_bits - 2]
                ) - x_floor  # rest of rounding [0, 1)
                alpha = -torch.log(
                    (self.zeta - self.gamma) / (rest - self.gamma) - 1
                )  # => sigmoid(alpha) = rest
                self.alpha = nn.Parameter(alpha)
            else:
                raise NotImplementedError

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, "bitwidth not supported"
        if self.quntizer == "uaq":
            self.n_bits = refactored_bit
            self.n_levels = 2**self.n_bits
        elif self.quntizer == "lsq":
            raise NotImplementedError

    @torch.jit.export
    def extra_repr(self):
        return "bit={}, soft_targets={},round_mode={}".format(
            self.n_bits, self.soft_targets, self.round_mode
        )
