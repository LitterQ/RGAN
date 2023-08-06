# RGAN (WGAN-GP-based)
This is the official Pytorch implementation of RGAN [Robust Generative Adversarial Network] based on WGAN-GP [Improved Training of Wasserstein GANs]([https://openreview.net/pdf?id=B1QRgziT-](https://arxiv.org/abs/1704.00028)). 

The codes are implemented based on the released codes from "Improved Training of Wasserstein GANs"([https://github.com/GongXinyuu/sngan.pytorch](https://github.com/igul222/improved_wgan_training))

## Prerequisites

- Python, NumPy, TensorFlow, SciPy, Matplotlib
- A recent NVIDIA GPU

## Training
- `python rgan_cifar_resnet.py`: Resnet-based CIFAR-10
- `python rgan_dc_cifar.py`: DCGAN CIFAR-10

## Acknowledgement

1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. The code of Spectral Norm GAN is inspired by [https://github.com/igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training).
