# PTMQ: Post-Training Multi-Bit Quantization of Neural Networks

## Experiment Environment

### Software

- CUDA:11.7.3
- Python: 3.9
- PyTorch: 2.0
  
### Hardware

- GPU: GeForce RTX 3090

### Requirements

- configspace
- numpy
- simplejson
- yacs
  
## Pretrained models

CNN-based pretrained models can be downloaded from [BRECQ](https://github.com/yhhhli/BRECQ/releases/tag/v1.0). then move it to `~/.cache/torch/checkpoints`.

Transformer-based pretrained models are from [timm](https://github.com/huggingface/pytorch-image-models)

## Usage

### Data Preparation

setting the path for datasets by modifying the `IMAGENET_LOADER` attribute in the configuration file `\cnns\config\train_multibit_qnn.yaml`.

### Run example

Adjust the model in configuration file (and other parameters accordingly)

- Train Multi-bit Model\
    `python train_multibit_qnn.py --cfg ./config/train_multibit_qnn.yaml`

- Test Multi-bit Model\
    `python test_qnn.py --cfg ./config/test_qnn.yaml`

## Results

We performed the multi-bit quantized model using RTX 3090, the supernet training results are as follows:

| Model       | W3A3 | W4A4 |W5A5 |W6A6 |W7A7 |W8A8 |GPU Hours|
| ----------- | ---------------- | ---------------- |---------------- |---------------- |---------------- |---------------- |---------------- |
| ResNet-18   |64.92%|67.57%|69.00%|70.23%|70.66%|70.79%|1.4|
| ResNet-50   |69.67%|73.93%|75.18%|76.11%|76.45%|76.52%|7.8|
|RegNetX-600MF|59.29%|68.84%|70.50%|72.85%|73.20%|73.32%|3.5|
| MobileNet-V2|  --  |64.94%|68.13%|70.00%|71.07%|72.05%|3.1|

## Thanks

This implementation is mainly adapted from the source code of [QDrop](https://github.com/wimh966/QDrop)
