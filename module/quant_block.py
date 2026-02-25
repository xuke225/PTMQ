import torch
import torch.nn as nn
from .base_uaq import UniformAffineQuantizer
from .quant_layer import PTQuantModule
import sys
from timm.models.vision_transformer import Attention, Block, Mlp
import torch.nn.functional as F
sys.path.append('../../../')
from models.imagenet.mnasnet import _InvertedResidual  # noqa:F401
from models.imagenet.mobilenetv2 import InvertedResidual  # noqa:F401
from models.imagenet.regnet import ResBottleneckBlock  # noqa:F401
from models.imagenet.resnet import BasicBlock, Bottleneck  # noqa:F401


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    Due to the branch architecture, we have to perform activation function
    and quantization after the elemental-wise add operation, therefore, we
    put this part in this class.
    """

    def __init__(self):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        self.is_last_first = False
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, PTQuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantBasicBlock(BaseQuantBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """

    def __init__(self, basic_block: BasicBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = PTQuantModule(
            basic_block.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = basic_block.relu1
        self.conv2 = PTQuantModule(
            basic_block.conv2, weight_quant_params, act_quant_params, disable_act_quant=True)

        if basic_block.downsample is None:
            self.downsample = None
        else:
            self.downsample = PTQuantModule(basic_block.downsample[0], weight_quant_params, act_quant_params,
                                            disable_act_quant=True)
        self.activation_function = basic_block.relu2
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantBottleneck(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """

    def __init__(self, bottleneck: Bottleneck, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = PTQuantModule(
            bottleneck.conv1, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.relu1
        self.conv2 = PTQuantModule(
            bottleneck.conv2, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.relu2
        self.conv3 = PTQuantModule(
            bottleneck.conv3, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.downsample is None:
            self.downsample = None
        else:
            self.downsample = PTQuantModule(bottleneck.downsample[0], weight_quant_params, act_quant_params,
                                            disable_act_quant=True)
        # modify the activation function to ReLU
        self.activation_function = bottleneck.relu3
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantResBottleneckBlock(BaseQuantBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """

    def __init__(self, bottleneck: ResBottleneckBlock, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.conv1 = PTQuantModule(
            bottleneck.f.a, weight_quant_params, act_quant_params)
        self.conv1.activation_function = bottleneck.f.a_relu
        self.conv2 = PTQuantModule(
            bottleneck.f.b, weight_quant_params, act_quant_params)
        self.conv2.activation_function = bottleneck.f.b_relu
        self.conv3 = PTQuantModule(
            bottleneck.f.c, weight_quant_params, act_quant_params, disable_act_quant=True)

        if bottleneck.proj_block:
            self.downsample = PTQuantModule(bottleneck.proj, weight_quant_params, act_quant_params,
                                            disable_act_quant=True)
        else:
            self.downsample = None
        # copying all attributes in original block
        self.proj_block = bottleneck.proj_block

        self.activation_function = bottleneck.relu
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        residual = x if not self.proj_block else self.downsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.activation_function(out)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out


class QuantInvertedResidual(BaseQuantBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """

    def __init__(self, inv_res: InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.use_res_connect = inv_res.use_res_connect
        self.expand_ratio = inv_res.expand_ratio
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                PTQuantModule(
                    inv_res.conv[0], weight_quant_params, act_quant_params),
                PTQuantModule(inv_res.conv[3], weight_quant_params,
                              act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
        else:
            self.conv = nn.Sequential(
                PTQuantModule(
                    inv_res.conv[0], weight_quant_params, act_quant_params),
                PTQuantModule(
                    inv_res.conv[3], weight_quant_params, act_quant_params),
                PTQuantModule(inv_res.conv[6], weight_quant_params,
                              act_quant_params, disable_act_quant=True),
            )
            self.conv[0].activation_function = nn.ReLU6()
            self.conv[1].activation_function = nn.ReLU6()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.use_res_connect:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    @torch.jit.export
    def extra_repr(self):
        return 'use_weight_quant={},use_act_quant={}'.format(
            self.use_weight_quant, self.use_act_quant)


class _QuantInvertedResidual(BaseQuantBlock):
    def __init__(self, _inv_res: _InvertedResidual, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()

        self.apply_residual = _inv_res.apply_residual
        self.conv = nn.Sequential(
            PTQuantModule(
                _inv_res.layers[0], weight_quant_params, act_quant_params),
            PTQuantModule(
                _inv_res.layers[3], weight_quant_params, act_quant_params),
            PTQuantModule(_inv_res.layers[6], weight_quant_params,
                          act_quant_params, disable_act_quant=True),
        )
        self.conv[0].activation_function = nn.ReLU()
        self.conv[1].activation_function = nn.ReLU()
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        if self.apply_residual:
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

# quant block for vit


class QuantBlockVit(nn.Module):
    def __init__(self, block: Block, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.norm1 = block.norm1
        self.attn = QuantAttention(
            block.attn, weight_quant_params, act_quant_params)
        self.ls1 = block.ls1
        self.drop_path1 = block.drop_path1
        self.norm2 = block.norm2
        self.mlp = QuantMlp(block.mlp, weight_quant_params, act_quant_params)
        self.ls2 = block.ls2
        self.drop_path2 = block.drop_path2
        # quant set
        self.use_weight_quant = False
        self.use_act_quant = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        self.attn.set_quant_state(self.use_weight_quant, self.use_act_quant)
        self.mlp.set_quant_state(self.use_weight_quant, self.use_act_quant)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class QuantAttention(nn.Module):
    def __init__(self, attentionblock: Attention, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.num_heads = attentionblock.num_heads
        self.head_dim = attentionblock.head_dim
        self.scale = attentionblock.scale
        self.fused_attn = attentionblock.fused_attn

        self.qkv = PTQuantModule(
            attentionblock.qkv, weight_quant_params, act_quant_params)
        self.q_norm = attentionblock.q_norm
        self.k_norm = attentionblock.k_norm
        self.attn_drop = attentionblock.attn_drop
        self.proj = PTQuantModule(
            attentionblock.proj, weight_quant_params, act_quant_params)
        self.proj_drop = attentionblock.proj_drop
        self.act_attn1 = UniformAffineQuantizer(**act_quant_params)
        self.act_act2 = UniformAffineQuantizer(**act_quant_params)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.use_act_quant:
            attn = self.act_attn1(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if self.use_act_quant:
            x = self.act_act2(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, PTQuantModule):
                m.set_quant_state(weight_quant, act_quant)


class QuantMlp(nn.Module):
    def __init__(self, mlpblock: Mlp, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        super().__init__()
        self.fc1 = PTQuantModule(
            mlpblock.fc1, weight_quant_params, act_quant_params)
        self.act = mlpblock.act
        self.drop1 = mlpblock.drop1
        self.norm = mlpblock.norm
        self.fc2 = PTQuantModule(
            mlpblock.fc2, weight_quant_params, act_quant_params)
        self.drop2 = mlpblock.drop2

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):

        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, PTQuantModule):
                m.set_quant_state(weight_quant, act_quant)


specials = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
    Block: QuantBlockVit
}
