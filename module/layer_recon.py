import torch  # NOQA: E402
import sys
import random
from typing import Union
sys.path.append("../../../")
from module.base_adaround import AdaRoundQuantizer  # noqa:F401
from module.quant_block import BaseQuantBlock  # noqa:F401
from module.quant_model import PTQuantModel, PTQuant4Vit  # noqa:F401
from module.quant_layer import PTQuantModule  # noqa:F401
from module.base_uaq import lp_loss  # noqa:F401
import utils.logger as logging  # noqa:F401
from utils.configtool.core.config import cfg  # noqa:F401
from .set_weight_quantize_params import weight_get_quant_state, get_init  # noqa:F401
from .set_act_quantize_params import set_act_quantize_params  # noqa:F401
from .block_recon import LinearTempDecay  # noqa:F401
logger = logging.get_logger(__name__)
# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def mfm(model, layer, input_data, bs, wwq, waq, keep_gpu, input_prob):
        cached_inps_quant_list1 = []
        cached_inps_quant_list2 = []
        cached_outs_fp = None
        cached_inps_fp = None

        groups1 = list()
        groups1.append(random.choice(cfg.group.H))
        groups1.append(random.choice(cfg.group.M))
        groups1.append(random.choice(cfg.group.L))
        bit_condinate = [3,4,5,6,7,8]
        groups2 = list(set(bit_condinate)-set(groups1))
        
        for i in groups1:
            # print("save input and output in {}-bit".format(i))
            model.set_other_layer_to_nbit(i)
            cached_inps_quant, cached_outs_fp = get_init(
                model,
                layer,
                input_data,
                wq=wwq,
                aq=waq,
                batch_size=bs,
                input_prob=True,
                keep_gpu=keep_gpu,
            )
            cached_inps_quant_list1.append(cached_inps_quant[0])
            cached_inps_fp = cached_inps_quant[1]

        MIXUP_Lambda = cfg.MIXUP_PARAM
        cached_inps_quant_mixed1 = (
                cached_inps_quant_list1[0] * MIXUP_Lambda[0]
                + cached_inps_quant_list1[1] * MIXUP_Lambda[1]
                + cached_inps_quant_list1[2] * MIXUP_Lambda[2]
        )

        for i in groups2:
            # print("save input and output in {}-bit".format(i))
            model.set_other_layer_to_nbit(i)
            cached_inps_quant, _ = get_init(
                model,
                layer,
                input_data,
                wq=wwq,
                aq=waq,
                batch_size=bs,
                input_prob=True,
                keep_gpu=keep_gpu,
            )
            cached_inps_quant_list2.append(cached_inps_quant[0])
        
        cached_inps_quant_mixed2 = (
                cached_inps_quant_list2[0] * MIXUP_Lambda[0]
                + cached_inps_quant_list2[1] * MIXUP_Lambda[1]
                + cached_inps_quant_list2[2] * MIXUP_Lambda[2]
        )
        
        return (cached_inps_quant_mixed1, cached_inps_quant_mixed2), cached_inps_fp, cached_outs_fp


def layer_reconstruction(
    model: Union[PTQuantModel, PTQuant4Vit],
    layer: PTQuantModule,
    cali_data: torch.Tensor,
    batch_size: int = 128,
    iters: int = 20000,
    weight: float = 0.001,
    opt_mode: str = "mse",
    act_quant: bool = False,
    b_range: tuple = (20, 2),
    warmup: float = 0.0,
    p: float = 2.0,
    lr: float = 4e-5,
    wwq: bool = True,
    waq: bool = True,
    order: str = "together",
    input_prob: float = 1.0,
    keep_gpu: bool = False,
    is_last_first=False,
):
    """
    Block reconstruction to optimize the output from each layer.
    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
        :param p: L_p norm minimization
    """

    """set scale"""
   
    groups = list()
    groups.append(random.choice(cfg.group.H))
    groups.append(random.choice(cfg.group.M))
    groups.append(random.choice(cfg.group.L))
    
    # get multi-bit inputs for act scale init
    inps_quant_list = []
    for i in groups:
        print("save input and output for act-scale in {}bits".format(i))
        model.set_other_layer_to_nbit(i)
        inps , outs = get_init(
                model,
                layer,
                cali_data[:256],
                wq=wwq,
                aq=waq,
                batch_size=batch_size,
                input_prob=True,
                keep_gpu=keep_gpu,
            )
        inps_quant_list.append(inps[0])
    
    MIXUP_Lambda = cfg.MIXUP_PARAM
    inps_quant_mixed = (
           inps_quant_list[0] * MIXUP_Lambda[0]
            + inps_quant_list[1] * MIXUP_Lambda[1]
            + inps_quant_list[2] * MIXUP_Lambda[2]
    )
    if act_quant and order == "together":
       
        set_act_quantize_params(
            layer,
            cali_data=inps_quant_mixed.to("cuda"),
            awq=True,
            order=order,
        )

       
    """set state"""
    cur_weight, cur_act = weight_get_quant_state(order, act_quant)
    layer.set_quant_state(cur_weight, cur_act)

    """set quantizer"""
    round_mode = "learned_hard_sigmoid"
    # Replace weight quantizer to AdaRoundQuantizer
    w_para, a_para = [], [],
    w_opt, a_opt = None, None
    scheduler, a_scheduler = None, None

    if is_last_first:
        logger.info("This layer is last or first block in qnn model")

    """weight round"""

    layer.weight_quantizer = AdaRoundQuantizer(
        quntizer=layer.weight_quantizer,
        round_mode=round_mode,
        weight_tensor=layer.org_weight.data,
    )
    layer.weight_quantizer.soft_targets = True

    
    w_para += [layer.weight_quantizer.alpha]

    """activation"""
    if act_quant and order == "together" and layer.act_quantizer.delta_set is not None:
        
        layer.act_quantizer.delta_set = torch.nn.Parameter(
            torch.tensor(layer.act_quantizer.delta_set)
        )
        a_para += [layer.act_quantizer.delta_set]
    layer.act_quantizer.is_training = True  

    if len(w_para) != 0:
        w_opt = torch.optim.Adam(w_para, lr=cfg.W_CALIBRATION_PARAM.wlr)
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt, T_max=iters, eta_min=0.0
        )

    loss_mode = "relaxation"  

    rec_loss = opt_mode

    loss_func = LossFunction(
        layer,
        round_loss=loss_mode,
        weight=weight,
        max_count=iters,
        rec_loss=rec_loss,
        b_range=b_range,
        decay_start=0,
        warmup=warmup,
        p=p,
    )
    torch.cuda.empty_cache()
    device = "cuda"
    sz = 1024
    quant_inps, fp_inp, fp_out = mfm(model, layer,cali_data,batch_size,wwq,waq,keep_gpu,input_prob)
    torch.set_num_threads(10)
    for i in range(iters):
        c = random.choice([0, 1])
        idx = torch.randint(0, sz, (batch_size,))
        cur_inp = quant_inps[c][idx].to(device)
        cur_fp_inps = fp_inp[idx].to(device)
        if input_prob < 1.0:
            cur_inp = torch.where(
                torch.rand_like(cur_inp) < input_prob,
                cur_inp,
                cur_fp_inps,
            )
        cur_out = fp_out[idx].to(device)
        
        w_opt.zero_grad()
        if a_opt:
            a_opt.zero_grad()
        err = 0
        if is_last_first:
            layer.weight_quantizer.bitwidth_refactor(8)
            layer.act_quantizer.bitwidth_refactor(8)
            output = layer(cur_inp)
            err = loss_func(output, cur_out)
        else:
            lambda_rec = cfg.MIXUP_PARAM
            
            rec_groups = list()
            rec_groups.append(random.choice(cfg.group.H))
            rec_groups.append(random.choice(cfg.group.M))
            rec_groups.append(random.choice(cfg.group.L))
            output, quant_pred = cal_output_layer(
                cur_inp,
                layer,
                tuple(cur_out.shape),
                rec_groups,
                lambda_rec,
                num_bit = 3 
            )
            if not cfg.GDLOSS:
                err = loss_func(output, cur_out)
            else:
                err = loss_func(output, cur_out, quant_pred=quant_pred)

        err.backward(retain_graph=True)

        w_opt.step()
        if a_opt:
            a_opt.step()
        if scheduler:
            scheduler.step()
        if a_scheduler:
            a_scheduler.step()

    torch.cuda.empty_cache()
    layer.weight_quantizer.soft_targets = False
    layer.act_quantizer.is_training = False


def cal_output_layer(
    inp,
    layer:  PTQuantModule,
    out_size,
    nbit_list,
    lambda_list,
    num_bit: int,

):
    output = torch.zeros(out_size).cuda()
    quant_pred = list()
    for i in range(num_bit):
        layer.weight_quantizer.bitwidth_refactor(nbit_list[i])  # GROUP h
        if nbit_list[i] < 6:
            layer.act_quantizer.bitwidth_refactor(6)
        else:
            layer.act_quantizer.bitwidth_refactor(nbit_list[i])
        quant_pred.append(layer(inp))
        # mfm for outs
        output += quant_pred[i] * lambda_list[i]

    return output, quant_pred


class LossFunction:
    def __init__(
        self,
        layer: PTQuantModule,
        round_loss: str = "relaxation",
        weight: float = 1.0,
        rec_loss: str = "mse",
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):
        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0

    def __call__(self, pred, tgt, quant_pred=None, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy
        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == "mse":
            if quant_pred is None:
                rec_loss = lp_loss(pred, tgt, p=self.p)
            else:
                rec_loss_fp = lp_loss(pred, tgt, p=self.p)
                rec_loss_h2l = lp_loss(quant_pred[2], quant_pred[0], p=self.p)
                rec_loss_h2m = lp_loss(quant_pred[1], quant_pred[0], p=self.p)
                #
                lambda_kd = cfg.GDLOSS_PARAM
                rec_loss = (
                    lambda_kd[0] * rec_loss_fp
                    + rec_loss_h2l * lambda_kd[1]
                    + rec_loss_h2m * lambda_kd[2]
                       
                )
               
        elif self.rec_loss == "fisher_diag":
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == "fisher_full":
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError(
                "Not supported reconstruction loss function: {}".format(
                    self.rec_loss)
            )

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == "none":
            b = round_loss = 0
        elif self.round_loss == "relaxation":
            round_loss = 0
            round_vals = self.layer.weight_quantizer.get_soft_targets()
            round_loss += (
                self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
            )
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 2000 == 0:
            print(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss), float(rec_loss), float(
                        round_loss), b, self.count
                )
            )
            logger.info(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss), float(rec_loss), float(
                        round_loss), b, self.count
                )
            )
        return total_loss
