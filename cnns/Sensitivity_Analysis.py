import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
sys.path.append("../")
from utils.common import (  # noqa:F401
    AverageMeter,  # noqa:F401
    checkpoint,  # noqa:F401
    accuracy,
    get_train_samples,  # noqa:F401
    pickle_load_model,  # noqa:F401
)
from data.imagenet.imagenet import get_imagenet_dataloader  # noqa:F401
from module.quant_layer import PTQuantModule  # noqa:F401
from module.quant_block import BaseQuantBlock  # noqa:F401
from module.quant_model import PTQuantModel  # noqa:F401
from module.data_utils import save_grad_data, save_inp_oup_data  # noqa:F401
from utils.configtool.core.config import cfg  # noqa:F401
from utils.configtool.core.builder import setup_env  # noqa:F401
import utils.logger as logging  # noqa:F401
import utils.configtool.core.config as config  # noqa:F401
import utils.common as utils  # noqa:F401
from utils.common import pickle_dump_model  # noqa:F401

# Load config and check
config.load_configs()
logger = logging.get_logger(__name__)
ckpt = checkpoint(cfg.OUT_DIR)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def block_kl_on_cali_data(model: PTQuantModel, calidata, b_id):
    model.eval()
    losses = AverageMeter()
    start_time = time.time()
    bz = calidata.size(0) / 64
    with torch.no_grad():
        for i in range(int(bz)):
            inputs = calidata[i * 64: (i + 1) * 64].to(device)
            model.set_quant_state(weight_quant=False, act_quant=False)
            fp_outputs = model(inputs)
            model.set_block_quant(block_id=b_id)
            quant_outputs = model(inputs)
            loss = F.kl_div(F.log_softmax(quant_outputs, dim=1), F.softmax(
                fp_outputs, dim=1), reduction='batchmean')
            losses.update(loss.item(), inputs.size(0))
        current_time = time.time()
        logger.info(
            "KL Loss {:.4f}\ton {}th block\tTime {:.2f}s".format(
                float(losses.avg), b_id, (current_time - start_time)
            )
        )
    return losses.avg


def get_block_nums(nn):

    cnt = 0
    for t in nn.modules():
        if isinstance(t, BaseQuantBlock):
            cnt += 1
    return cnt


def main():
    seed_all()
    setup_env()
    trainLoader, testLoader = get_imagenet_dataloader()
    cali_data, cali_label = get_train_samples(
        train_loader=trainLoader, num_samples=1024, label_info=True
    )
    if cfg.qnn.baseline:
        print("load qnn model from baseline:{}".format(cfg.qnn.baselinepath))
        qnn = pickle_load_model(cfg.qnn.baselinepath)
    else:
        print("please use qnn model")
    qnn.disable_network_output_quantization()
    qnn.to(device)
    print(qnn)

    qnn.set_quant_state(weight_quant=True, act_quant=True)

    sen_table = list()
    block_nums = get_block_nums(qnn)
    print(block_nums)
    for i in range(3, 9):
        print("sensitivity for {}bit ".format(i))
        qnn.set_other_layer_to_nbit(i)

        temp_bit_list = list()
        for i in range(1, block_nums+1):
            temp_bit_list.append(block_kl_on_cali_data(qnn, cali_data, b_id=i))
        sen_table.append(temp_bit_list)
    save_path = cfg.OUT_DIR + '/checkpoint/Sensitivity_look_up_table_' + \
        str(cfg.ARCH.name)+'.pickle'
    pickle_dump_model(sen_table, save_path)


if __name__ == "__main__":
    main()
