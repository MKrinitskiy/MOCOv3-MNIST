## MoCo v3 for Self-supervised ResNet and ViT

This is a fork of original [MoCo v3](https://github.com/MKrinitskiy/MOCOv3-MNIST.git) repository. The purpose of this repository is the hardcoded training of MoCoV3 using MNIST dataset.

In this fork, we also corrected some bugs popping up in case of non-distributed training.

### Introduction

This is a PyTorch implementation of [MoCo v3](https://arxiv.org/abs/2104.02057) for self-supervised ResNet and ViT.




### Usage: Preparation

Install PyTorch.

For ViT models, install [timm](https://github.com/rwightman/pytorch-image-models) (`timm==0.4.9`).

The code has been tested with CUDA 10.2/CuDNN 7.6.5, PyTorch 1.9.0 and timm 0.4.9.

### Usage: Self-supervised Pre-Training

Below is an example for MoCo v3 training. 

#### ResNet-50 with 1-node (1 GPU) training, batch 32

Run:
```
python main_moco.py \
  --arch=resnet50 \
  --workers=4 \
  --epochs=10 \
  --batch-size=32 \
  --learning-rate=1e-4 \
  --moco-dim=16 \
  --moco-mlp-dim=1024 \
  --crop-min=0.7 \
  --print-freq=10
```


#### Notes:
Using a smaller batch size has a more stable result (see paper), but has lower speed. Using a large batch size is critical for good speed in TPUs (as we did in the paper).

### Model Configs

See the commands listed in [CONFIG.md](https://github.com/MKrinitskiy/MOCOv3-MNIST/blob/master/CONFIG.md) for specific model configs, including our recommended hyper-parameters and pre-trained reference models.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### Citation (original paper)
```
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```
