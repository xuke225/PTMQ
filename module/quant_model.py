import random
import torch.nn as nn
from module.base_adaround import AdaRoundQuantizer
from module.base_uaq import UniformAffineQuantizer
from .quant_block import specials, BaseQuantBlock, QuantBlockVit
from .quant_layer import PTQuantModule, StraightThrough
from .merge_bn import search_fold_and_remove_bn


class PTQuantModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},

    ):
        super().__init__()
        search_fold_and_remove_bn(model)
        self.model = model
        self.quant_module_refactor(
            self.model, weight_quant_params, act_quant_params
        )

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        """
        Recursively replace the normal conv2d and Linear layer to QuantModule 
        :param module: nn.Module with nn.Conv2d or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer 
        :param act_quant_params: quantization parameters like n_bits for activation quantizer 
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child_module)](
                        child_module, weight_quant_params, act_quant_params
                    ),
                )
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    PTQuantModule(
                        child_module, weight_quant_params, act_quant_params
                    ),
                )
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6)):
                if prev_quantmodule is not None:
                    prev_quantmodule.activation_function = child_module
                    setattr(module, name, StraightThrough())
                else:
                    continue

            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(
                    child_module, weight_quant_params, act_quant_params
                )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (PTQuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)
        module_list = []
        for m in self.model.modules():
            if isinstance(m, PTQuantModule):
                module_list += [m]
        module_list[0].is_last_first = True
        module_list[-1].is_last_first = True

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, PTQuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True

    def set_other_layer_weight_to_nbits(self, nbit):
        w_list = []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    pass
                else:
                    module.bitwidth_refactor(nbit)
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)

    def set_mixed_bit_from_bit_conf_by_layer(self, wbit: list = [], abit: list = []):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        assert len(w_list) == len(wbit) and len(a_list) == len(abit)
        for i in range(len(w_list)):
            w_list[i].bitwidth_refactor(wbit[i])
        for i in range(len(a_list)):
            a_list[i].bitwidth_refactor(abit[i])

        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def set_mixed_bit_from_bit_conf_by_block(self, block_wbit: list = [], block_abit: list = []):
        block_list = list()

        for block in self.model.modules():
            if isinstance(block, BaseQuantBlock):
                block_list.append(block)
        assert len(block_wbit) == len(block_list) and len(
            block_abit) == len(block_list)
        for i in range(len(block_list)):
            for module in block_list[i].modules():
                """weight"""
                if isinstance(module, PTQuantModule):
                    module.weight_quantizer.bitwidth_refactor(block_wbit[i])
                """activation"""
                if isinstance(module, (PTQuantModule, BaseQuantBlock)):
                    module.act_quantizer.bitwidth_refactor(block_abit[i])

        ##
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)

        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def set_other_layer_act_to_nbits(self, nbit):
        a_list = []
        for module in self.model.modules():
            if isinstance(module, UniformAffineQuantizer):
                if module.leaf_param:
                    module.bitwidth_refactor(nbit)
                    a_list.append(module)
                else:
                    pass
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[0].bitwidth_refactor(8)
        a_list[-2].bitwidth_refactor(8)

    def set_other_layer_to_nbit(self, nbit):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                module.bitwidth_refactor(nbit)
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    # nbit_list=[3,4,5,6,7,8]
    def sample_other_layer_to_nbit(self, nbit_list):
        w_list, a_list = [], []
        module_list = []
        for module in self.model.modules():
            bits = random.choice(nbit_list)

            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                module.bitwidth_refactor(bits)
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)

        for block in self.model.modules():
            bits = random.choice(nbit_list)
            if isinstance(block, BaseQuantBlock):
                for module in block.modules():
                    """weight"""
                    if isinstance(module, PTQuantModule):
                        module.weight_quantizer.bitwidth_refactor(bits)
                    """activation"""
                    if isinstance(module, (PTQuantModule, BaseQuantBlock)):
                        module.act_quantizer.bitwidth_refactor(bits)

        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def set_block_quant(self, block_id: int = 0):
        block_list = list()
        for block in self.model.modules():
            if isinstance(block, BaseQuantBlock):
                block_list.append(block)
        assert block_id >= 0 and block_id <= len(block_list)
        # print(len(block_list))
        for b in block_list:
            b.set_quant_state(False, False)
        if block_id == 0:
            pass
        else:
            block_list[block_id-1].set_quant_state(True, True)

    def get_layers_bit_width(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    a_list.append(module.n_bits)
                else:
                    w_list.append(module.n_bits)
        return w_list, a_list


class PTQuant4Vit(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        # malq_p: dict = {}

    ):
        super().__init__()
        self.model = model
        self.quant_module_refactor(
            self.model, weight_quant_params, act_quant_params
        )

    def quant_module_refactor(
        self,
        module: nn.Module,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
    ):
        prev_quantmodule = None
        for name, child_module in module.named_children():
            if type(child_module) in specials:
                setattr(
                    module,
                    name,
                    specials[type(child_module)](
                        child_module, weight_quant_params, act_quant_params
                    ),
                )
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    PTQuantModule(
                        child_module, weight_quant_params, act_quant_params
                    ),
                )
                prev_quantmodule = getattr(module, name)
            elif isinstance(child_module, StraightThrough):
                continue

            else:
                self.quant_module_refactor(
                    child_module, weight_quant_params, act_quant_params
                )

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (PTQuantModule, QuantBlockVit)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, input):
        return self.model(input)

    def set_first_last_layer_to_8bit(self):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                if module.leaf_param:
                    a_list.append(module)
                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)
        module_list = []
        for m in self.model.modules():
            if isinstance(m, PTQuantModule):
                module_list += [m]
        module_list[0].is_last_first = True
        module_list[-1].is_last_first = True

    def set_other_layer_to_nbit(self, nbit, use_6bit_act=False):
        w_list, a_list = [], []
        for module in self.model.modules():
            if isinstance(
                module,
                (
                    UniformAffineQuantizer,
                    AdaRoundQuantizer,
                ),
            ):
                module.bitwidth_refactor(nbit)
                if module.leaf_param:
                    if not use_6bit_act:
                        pass
                    else:
                        if nbit < 6:
                            module.bitwidth_refactor(6)
                        else:
                            pass
                    a_list.append(module)

                else:
                    w_list.append(module)
        w_list[0].bitwidth_refactor(8)
        a_list[0].bitwidth_refactor(8)
        w_list[-1].bitwidth_refactor(8)
        "the image input has been in 0~255, set the last layer's input to 8-bit"
        a_list[-2].bitwidth_refactor(8)

    def disable_network_output_quantization(self):
        module_list = []
        for m in self.model.modules():
            if isinstance(m, PTQuantModule):
                module_list += [m]
        module_list[-1].disable_act_quant = True
