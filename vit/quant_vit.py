import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import random
import multiprocessing
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
import sys
from timm.models.vision_transformer import Block
sys.path.append('../')
import utils.common as utils  # noqa:F401
import utils.configtool.core.config as config  # noqa:F401
import utils.logger as logging  # noqa:F401
from utils.configtool.core.builder import setup_env  # noqa:F401
from utils.configtool.core.config import cfg  # noqa:F401
from models.imagenet.hubconf import get_vit_net  # noqa:F401
from module.quant_model import PTQuant4Vit  # noqa:F401
from module.base_uaq import UniformAffineQuantizer  # noqa:F401
from module.quant_block import QuantBlockVit  # noqa:F401
from module.quant_layer import PTQuantModule  # noqa:F401
from module.block_recon import block_reconstruction  # noqa:F401
from module.layer_recon import layer_reconstruction  # noqa:F401
from module.set_weight_quantize_params import set_weight_quantize_params  # noqa:F401
from module.set_act_quantize_params import set_act_quantize_params  # noqa:F401
import data.vit_data.datasets as datasets  # noqa:F401
from utils.common import AverageMeter, checkpoint, accuracy, get_train_samples, pickle_dump_model, pickle_load_model  # noqa:F401

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
ckpt = checkpoint(cfg.OUT_DIR)

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def seed_all(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test_classification(net, test_loader, max_iteration=None, description=None):
    pos = 0
    tot = 0
    i = 0
    max_iteration = len(
        test_loader) if max_iteration is None else max_iteration
    with torch.no_grad():
        q = tqdm(test_loader, desc=description)
        for inp, target in q:
            i += 1
            inp = inp.cuda()
            target = target.cuda()
            out = net(inp)
            pos_num = torch.sum(out.argmax(1) == target).item()
            pos += pos_num
            tot += inp.size(0)
            q.set_postfix({"acc": pos/tot})
            if i >= max_iteration:
                break
    print(pos/tot)
    return pos/tot

def main():
    seed_all(cfg.SEED)
    setup_env()

    net = get_vit_net(cfg.net)
    print("="*20+">load data")
    g = datasets.ViTImageNetLoaderGenerator(
        '/data/ImageNet', 'imagenet', 64, 64, 4, kwargs={"model": net})

    test_loader = g.test_loader()
    calib_loader = g.calib_loader(num=1024)
    cali_data = None
    with torch.no_grad():
        for i, (images, target) in enumerate(calib_loader):
            cali_data = images
    del calib_loader

    wq_params = {'channel_wise': cfg.QUANT_PARAM.wq.channel_wise,
                'scale_method': cfg.QUANT_PARAM.wq.scale_method, }
    aq_params = {'channel_wise': cfg.QUANT_PARAM.aq.channel_wise, 
                 'scale_method': cfg.QUANT_PARAM.aq.scale_method,
                    'leaf_param': True, 'prob': 0.5}
    q_net = PTQuant4Vit(net, weight_quant_params=wq_params,
                        act_quant_params=aq_params)
        # print(q_net)
    set_weight_quantize_params(q_net)
    print("="*20+">check the model!")
    q_net.set_first_last_layer_to_8bit()
    q_net.disable_network_output_quantization()

    print(q_net)

    kwargs = dict(cali_data=cali_data, iters=cfg.W_CALIBRATION_PARAM.iters_w, weight=cfg.W_CALIBRATION_PARAM.weight,
                  b_range=(cfg.W_CALIBRATION_PARAM.b_start,
                           cfg.W_CALIBRATION_PARAM.b_end), warmup=cfg.W_CALIBRATION_PARAM.warmup, opt_mode='mse',
                  wwq=cfg.W_CALIBRATION_PARAM.wwq, waq=cfg.W_CALIBRATION_PARAM.waq, order=cfg.ORDER_PARAM.order, act_quant=cfg.QUANT_PARAM.act_quant,
                  lr=cfg.A_CALIBRATION_PARAM.lr, input_prob=cfg.ORDER_PARAM.input_prob, keep_gpu=cfg.W_CALIBRATION_PARAM.keep_gpu, batch_size=cfg.W_CALIBRATION_PARAM.mini_batch)

    def set_weight_act_quantize_params(module):
        if isinstance(module, PTQuantModule):
            layer_reconstruction(
                q_net, module, is_last_first=module.is_last_first, **kwargs)
        elif isinstance(module, QuantBlockVit):
            block_reconstruction(
                q_net, module, is_last_first=False, **kwargs)
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
            elif isinstance(module, QuantBlockVit):
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module)
            else:
                recon_model(module)
            torch.cuda.empty_cache()

    recon_model(q_net)
    q_net.set_quant_state(True, True)
    with torch.no_grad():
        for i in range(4, 9):
            q_net.set_other_layer_to_nbit(i, use_6bit_act=True)
            qnn_acc = test_classification(q_net, test_loader)
            if i < 6:
                logger.info(
                    'Full quantization (W{}A{}) for {} after optim accuracy: {}%'.format(i, 6, cfg.net, qnn_acc*100))
            else:
                logger.info(
                    'Full quantization (W{}A{}) for {} after optim accuracy: {}%'.format(i, i, cfg.net, qnn_acc*100))


if __name__ == '__main__':
    main()
