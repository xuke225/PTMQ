import warnings
import torch
import torch.nn as nn


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    ::param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta_set = None
        self.zero_point_set = None
        self.inited = False

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise

        '''mse params'''
        self.scale_method = scale_method
        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited


    def forward(self, x: torch.Tensor):
        
        if self.inited is False:
            if self.leaf_param:
                self.delta_set, self.zero_point_set = self.init_quantization_scale(
                    x.clone().detach(), self.channel_wise)
            else:
                self.delta_set, self.zero_point_set = self.init_quantization_scale(
                    x.clone().detach(), self.channel_wise)
            self.inited = True

        # start quantization
        delta = self.delta_set[self.n_bits-2]
        x_int = round_ste(
            x / self.delta_set[self.n_bits-2]) + self.zero_point_set[self.n_bits-2]
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point_set[self.n_bits-2]) * delta

        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta_set = []
        zero_point_set = []
        for bit in range(2, 9):
            delta, zero_point = None, None
            if channel_wise:
                x_clone = x.clone().detach()
                n_channels = x_clone.shape[0]
                if len(x.shape) == 4:
                    x_max = x_clone.abs().max(
                        dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
                else:
                    x_max = x_clone.abs().max(dim=-1)[0]
                delta = x_max.clone()
                zero_point = x_max.clone()
                # determine the scale and zero point channel-by-channel
                for c in range(n_channels):
                    delta[c], zero_point[c] = self.init_quantization_scale_by_channel(
                        x_clone[c], bit)
                if len(x.shape) == 4:
                    delta = delta.view(-1, 1, 1, 1)
                    zero_point = zero_point.view(-1, 1, 1, 1)
                else:
                    delta = delta.view(-1, 1)
                    zero_point = zero_point.view(-1, 1)
            else:
                if 'max' in self.scale_method:
                    x_min = min(x.min().item(), 0)
                    x_max = max(x.max().item(), 0)
                    if 'scale' in self.scale_method:
                        x_min = x_min * (bit + 2) / 8
                        x_max = x_max * (bit + 2) / 8

                    x_absmax = max(abs(x_min), x_max)
                    if self.sym:
                        x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                    delta = float(x_max - x_min) / ((2 ** bit) - 1)
                    if delta < 1e-8:
                        warnings.warn(
                            'Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                        delta = 1e-8

                    zero_point = round(-x_min / delta)
                    delta = torch.tensor(delta).type_as(x)

                elif self.scale_method == 'mse':
                    # we always use symmetric quantization in mse mode
                    x_absmax = x.abs().max()
                    x_min = x.min().item()
                    best_score = 100000
                    for i in range(80):
                        new_max = x_absmax * (1.0 - (i * 0.01))
                        x_q = self.quantize(x, new_max, bit)
                        # L_p norm minimization as described in LAPQ
                        # https://arxiv.org/abs/1911.07190
                        score = lp_loss(x, x_q, p=2.4, reduction='all')
                        if score < best_score:
                            best_score = score
                            delta = (2 * new_max) / (2 ** bit - 1)
                            zero_point = float(
                                (new_max / delta).round()) if x_min < 0 else 0.0
                            # re-calculate the scale delta if zero-point is not 0,
                else:
                    raise NotImplementedError
            delta_set.append(delta)
            zero_point_set.append(zero_point)
        return delta_set, zero_point_set

    def init_quantization_scale_by_channel(self, x: torch.Tensor, bit: int):
        delta, zero_point = None, None
        if 'max' in self.scale_method:
            x_min = min(x.min().item(), 0)
            x_max = max(x.max().item(), 0)
            if 'scale' in self.scale_method:
                x_min = x_min * (bit + 2) / 8
                x_max = x_max * (bit + 2) / 8

            x_absmax = max(abs(x_min), x_max)
            if self.sym:
                x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

            delta = float(x_max - x_min) / ((2 ** bit) - 1)
            if delta < 1e-8:
                warnings.warn(
                    'Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                delta = 1e-8

            zero_point = round(-x_min / delta)
            delta = torch.tensor(delta).type_as(x)

        elif self.scale_method == 'mse':
            # we always use symmetric quantization in mse mode
            x_absmax = x.abs().max()
            x_min = x.min().item()
            best_score = 100000
            for i in range(80):
                new_max = x_absmax * (1.0 - (i * 0.01))
                x_q = self.quantize(x, new_max, bit)
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(x, x_q, p=2.4, reduction='all')
                if score < best_score:
                    best_score = score
                    delta = (2 * new_max) / (2 ** bit - 1)
                    zero_point = float(
                        (new_max / delta).round()) if x_min < 0 else 0.0
                    # re-calculate the scale delta if zero-point is not 0,
        else:
            raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, bit):
        delta = (2 * max) / (2 ** bit - 1)
        # we assume weight quantization is always signed
        zero_point = (max / delta).round()
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, (2 ** bit) - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={},channel_wise={},leaf_param={}'.format(
            self.n_bits, self.is_training, self.inited, self.channel_wise, self.leaf_param)
