import torch
import sys
from typing import Union
sys.path.append('../../../')
from module.quant_model import PTQuantModel, PTQuant4Vit  # noqa:F401
from module.quant_layer import PTQuantModule  # noqa:F401
from module.quant_block import BaseQuantBlock, QuantBlockVit  # noqa:F401


def set_act_quantize_params(module: Union[PTQuantModel, PTQuant4Vit, PTQuantModule, BaseQuantBlock, QuantBlockVit],
                            cali_data, awq: bool = False, order: str = 'before', batch_size: int = 256):
    weight_quant, act_quant = act_get_quant_state(order, awq)
    module.set_quant_state(weight_quant, act_quant)

    for t in module.modules():
        if isinstance(t, (PTQuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(False)
        if isinstance(t, QuantBlockVit):
            t.attn.act_attn1.set_inited(False)
            t.attn.act_act2.set_inited(False)
    if isinstance(module, QuantBlockVit):
        module.attn.act_attn1.set_inited(False)
        module.attn.act_act2.set_inited(False)
    '''set or init step size and zero point in the activation quantizer'''
    batch_size = min(batch_size, cali_data.size(0))
    with torch.no_grad():

        for i in range(int(cali_data.size(0) / batch_size)):
            module(cali_data[i * batch_size:(i + 1) * batch_size])
    torch.cuda.empty_cache()

    for t in module.modules():
        if isinstance(t, (PTQuantModule, BaseQuantBlock)):
            t.act_quantizer.set_inited(True)
        if isinstance(t, QuantBlockVit):
            t.attn.act_attn1.set_inited(True)
            t.attn.act_act2.set_inited(True)
    if isinstance(module, QuantBlockVit):
        module.attn.act_attn1.set_inited(True)
        module.attn.act_act2.set_inited(True)


def act_get_quant_state(order, awq):
    if order == 'before':
        weight_quant, act_quant = False, True
    elif order == 'after':
        weight_quant, act_quant = awq, True
    elif order == 'together':
        weight_quant, act_quant = True, True
    else:
        raise NotImplementedError
    return weight_quant, act_quant
