# RGAN (SNGAN-based)
This is the official Pytorch implementation of RGAN [Robust Generative Adversarial Network] based on SNGAN [Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-). 
The codes are implemented based on the released codes from "SNGAN.pytorch"(https://github.com/GongXinyuu/sngan.pytorch)

## Set-up

### install libraries:
```bash
pip install -r requirements.txt
```

### prepare fid statistic file
 ```bash
mkdir fid_stat
```
Download the pre-calculated statistics for CIFAR10, 
[fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz), to `./fid_stat`.

### train
```bash
sh exps/sngan_cifar10_ours.sh
```

## Acknowledgement

1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. The code of Spectral Norm GAN is inspired by [https://github.com/GongXinyuu/sngan.pytorch](https://github.com/GongXinyuu/sngan.pytorch).
