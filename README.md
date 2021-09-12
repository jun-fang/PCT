# PCT
PyTorch implementation for [Positive-Congruent Training: Towards Regression-Free Model Updates](https://arxiv.org/abs/2011.09161) (CVPR 2021)

## Train & Test
See `run-cifar.sh` and `run-imagenet.sh`

## Results

- **CIFAR-10, ResNet-18**

|    PCT Approach    | Old Model <br/> Error Rate (%) | New Model <br/> Error Rate (%) | NFR (%) | Rel. NFR (%) |
| :----------------: | :-------: | :-------: | :-------: | :-------: |
| No Treatment  | 4.90 | 4.85 | 1.83 | 39.68 |
| Naive         | 4.90 | 5.01 | 1.97 | 41.35 |
| FD-KL         | 4.90 | 4.67 | **1.37** | **30.85** |
| FD-LM         | 4.90 | 4.58 | **1.34** | **30.77** |

- **CIFAR-100, ResNet-18**

|    PCT Approach    | Old Model <br/> Error Rate (%) | New Model <br/> Error Rate (%) | NFR (%) | Rel. NFR (%) |
| :----------------: | :-------: | :-------: | :-------: | :-------: |
| No Treatment  | 22.45 | 22.50 | 5.92 | 33.93 |
| Naive         | 22.45 | 23.41 | 6.65 | 36.63 |
| FD-KL         | 22.45 | 20.91 | **4.03** | **24.85** |
| FD-LM         | 22.45 | 20.78 | **3.68** | **22.84** |
