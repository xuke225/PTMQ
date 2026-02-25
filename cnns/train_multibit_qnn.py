import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
import numpy as np
import random
import time
from importlib import import_module
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

sys.path.append('../')

from utils.common import AverageMeter, checkpoint, accuracy, get_train_samples
import utils.logger as logging
from module.layer_recon import layer_reconstruction
from module.block_recon import block_reconstruction
from module.set_act_quantize_params import set_act_quantize_params
from module.set_weight_quantize_params import set_weight_quantize_params
from module.quant_block import BaseQuantBlock
from module.quant_model import PTQuantModel
from module.quant_layer import PTQuantModule
from module.base_uaq import UniformAffineQuantizer
from data.imagenet.imagenet import get_imagenet_dataloader
import models.imagenet.hubconf as hubconf
from utils.configtool.core.config import cfg
from utils.configtool.core.builder import setup_env
import utils.configtool.core.config as config
import utils.common as utils


# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_func = nn.CrossEntropyLoss()
# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
ckpt = checkpoint(cfg.OUT_DIR)

def seed_all(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test(model, testLoader, topk=(1,)):
    model.eval()

    losses = AverageMeter()
    acc = AverageMeter()
    top5_accuracy = AverageMeter()

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = accuracy(outputs, targets, topk=topk)
            acc.update(predicted[0], inputs.size(0))
            if len(topk) == 2:
                top5_accuracy.update(predicted[1], inputs.size(0))

        current_time = time.time()
        if len(topk) == 1:
            logger.info(
                'Test Loss {:.4f}\tAccuracy {:.2f}%\t\tTime {:.2f}s'
                .format(float(losses.avg), float(acc.avg), (current_time - start_time))
            )
        else:
            logger.info(
                'Test Loss {:.4f}\tTop1 {:.2f}%\tTop5 {:.2f}%\tTime {:.2f}s'
                .format(float(losses.avg), float(acc.avg), float(top5_accuracy.avg), (current_time - start_time))
            )
    if len(topk) == 1:
        return acc.avg
    else:
        return acc.avg, top5_accuracy.avg


def set_drop_prob(qnn_model: PTQuantModel):
    for module in qnn_model.modules():
        if isinstance(module, UniformAffineQuantizer):
            module.prob = 1.0


def main():
    

    
    seed_all(cfg.SEED)
    setup_env()
    trainLoader, testLoader = get_imagenet_dataloader()
   
        # load cnn model
    cnn = eval('hubconf.{}(pretrained=True)'.format(cfg.ARCH.name))
    cnn.eval()
    cnn.to(device)
    # build quantization parameters
    wq_params = {'channel_wise': cfg.QUANT_PARAM.wq.channel_wise,
                    'scale_method': cfg.QUANT_PARAM.wq.scale_method, }
    aq_params = {'channel_wise': cfg.QUANT_PARAM.aq.channel_wise, 
                 'scale_method': cfg.QUANT_PARAM.aq. scale_method,
                 'leaf_param': True, 'prob': cfg.ORDER_PARAM.prob}
    qnn = PTQuantModel(model=copy.deepcopy(cnn), 
                       weight_quant_params=wq_params, act_quant_params=aq_params)

    qnn.to(device)

    qnn.eval()
    if not cfg.QUANT_PARAM.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    qnn.disable_network_output_quantization()
    print('check the model!')
    print(qnn)

    torch.cuda.empty_cache()
    cali_data = get_train_samples(
        train_loader=trainLoader, num_samples=cfg.W_CALIBRATION_PARAM.num_samples)
    assert cfg.W_CALIBRATION_PARAM.wwq is True

    '''init weight quantizer'''
    
    print('set_weight_quantize_params...')
    set_weight_quantize_params(qnn)
    print('set_weight_quantize_params done!')

    kwargs = dict(cali_data=cali_data, iters=cfg.W_CALIBRATION_PARAM.iters_w, 
                  weight=cfg.W_CALIBRATION_PARAM.weight,
                  b_range=(cfg.W_CALIBRATION_PARAM.b_start,
                           cfg.W_CALIBRATION_PARAM.b_end), warmup=cfg.W_CALIBRATION_PARAM.warmup, 
                    opt_mode='mse',
                    wwq=cfg.W_CALIBRATION_PARAM.wwq, waq=cfg.W_CALIBRATION_PARAM.waq, order=cfg.ORDER_PARAM.order, act_quant=cfg.QUANT_PARAM.act_quant,
                    lr=cfg.A_CALIBRATION_PARAM.lr, input_prob=cfg.ORDER_PARAM.input_prob, 
                    keep_gpu=cfg.W_CALIBRATION_PARAM.keep_gpu, batch_size=cfg.W_CALIBRATION_PARAM.mini_batch)

    def set_weight_act_quantize_params(module):
        if isinstance(module, PTQuantModule):
            layer_reconstruction(
                qnn, module, is_last_first=module.is_last_first, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(
                qnn, module, is_last_first=module.is_last_first, **kwargs)
        else:
            raise NotImplementedError

    def recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, PTQuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module)

            elif isinstance(module, BaseQuantBlock):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module)
            else:
                recon_model(module)
            torch.cuda.empty_cache()
    # Start calibration
    recon_model(qnn)

    qnn.to(device)

    qnn.set_quant_state(weight_quant=True, act_quant=cfg.QUANT_PARAM.act_quant)
    with torch.no_grad():
        for i in range(3, 9):
            qnn.set_other_layer_to_nbit(i)
            logger.info(qnn.get_layers_bit_width())
            qnn_acc = test(qnn, testLoader, topk=(1, 5))
            logger.info(
                'Full quantization (W{}A{}) after optim accuracy: {}'.format(i, i, qnn_acc))
        qnn.sample_other_layer_to_nbit([3, 4, 5, 6, 7, 8])
        logger.info('Mix precise quant')
        logger.info(qnn.get_layers_bit_width())
        qnn_acc = test(qnn, testLoader, topk=(1, 5))
        logger.info('Mix precise quantization accuracy: {}'.format(qnn_acc))


if __name__ == '__main__':
    
    main()
