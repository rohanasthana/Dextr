# No Free Lunch in Neural Architectures? A Joint Analysis of Expressivity, Convergence, and Generalization [[PDF](https://openreview.net/pdf?id=EMys3eIDJ2)]

Wuyang Chen*, Wei Huang*, Zhangyang Wang

International Conference on Automated Machine Learning (AutoML-Conf), 2023

[code under development]

### Prerequisites
- Ubuntu 16.04
- Python 3.6.9
- CUDA 11.1 (lower versions may work but were not tested)
- NVIDIA GPU + CuDNN v7.3

This repository has been tested on V100 GPT. Configurations may need to be changed on different platforms.

### Installation
* Install dependencies:
```bash
pip install -r requirements.txt
```
* Download Tiny ImageNet (CIFAR-10 and CIFAR-100 will be automatically downloaded by Torchvision): https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4


### Command

```
# bulk train
python main_bulk.py --data /ssd1/dataset --dataset cifar10 --arch mlp --width 256 --lr 0.001 --epochs 3000 --gpu 0
python main_bulk.py --data /ssd1/dataset --dataset cifar100 --arch mlp --width 256 --lr 0.001 --epochs 3000 --gpu 0
python main_bulk.py --data /ssd1/dataset/tiny-imagenet-200 --dataset tinyimagenet --arch mlp --width 256 --lr 0.005 --epochs 3000 --gpu 0
```

**Prune DAG Ensemble**
```
CUDA_VISIBLE_DEVICES=0 python main.py --data /ssd1/dataset/tiny-imagenet-200 --dataset tinyimagenet --arch mlp --width 256 --lr 0.005 --epochs 3000
--dag 1_22_221-2_02_002 --rand_prune 0.2 0.5
--supernet
```


### Collect all NTK/NNGP/Length
```
python traversal_dags.py
CUDA_VISIBLE_DEVICES=2 python traversal_nngp_ntk.py --data /ssd1/dataset/tiny-imagenet-200 --dataset tinyimagenet --arch mlp --width 256 --repeat 3 --no_bias
CUDA_VISIBLE_DEVICES=2 python traversal_complexity.py --data /ssd1/dataset/tiny-imagenet-200 --dataset tinyimagenet --arch mlp --width 256 --repeat 3
--exp_name
```
