# RGAN (SNGAN-based)
The official Pytorch implementation of RGAN which is based on SNGAN[Spectral Normalization for Generative Adversarial Networks](https://openreview.net/pdf?id=B1QRgziT-). 
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
sh exps/sngan_cifar10.sh
```

### test
```bash
mkdir pre_trained
```
Download the pre-trained SNGAN model [sngan_cifar10.pth](https://drive.google.com/file/d/1koEJbx9anP2-BEMrqX6jgWXAvEUXG0AU/view?usp=sharing) to `./pre_trained`.
Run the following script:
```bash
sh exps/eval.sh
```

## Acknowledgement

1. Inception Score code from [OpenAI's Improved GAN](https://github.com/openai/improved-gan/tree/master/inception_score) (official).
2. FID code and statistics file from [https://github.com/bioinf-jku/TTUR](https://github.com/bioinf-jku/TTUR) (official).
3. The code of Spectral Norm GAN is inspired by [https://github.com/pfnet-research/sngan_projection](https://github.com/pfnet-research/sngan_projection) (official).
