# Dextr: Zero-Shot Neural Architecture Search with Singular Value Decomposition and Extrinsic Curvature
Rohan Asthana, Joschua Conrad, Maurits Ortmanns, Vasileios Belagiannis

This repository contains the code for the paper titled "Dextr: Zero-Shot Neural Architecture Search with Singular Value Decomposition and Extrinsic Curvature" 
[\[link\]](https://openreview.net/forum?id=X0vPof5DVh).

## Setup
- Clone this repository using the command
``` bash
git clone https://github.com/rohanasthana/Dextr.git
```

## Correlation Experiments
### NASBench101, 301 and TransNASBench-micro
- To setup the environment run `source setup_naslib.sh` 
- To run the experiments for NASBench101, NASBench301 and TransNASBench-micro, run the following commands:
```bash
bash NASLib/scripts/cluster/benchmarks/run_tnb101.sh correlation dextr #TransNASBench101-micro
bash NASLib/scripts/cluster/benchmarks/run_nb101.sh correlation dextr #NASBench-101
bash NASLib/scripts/cluster/benchmarks/run_nb301.sh correlation dextr #NASBench-301
```

### NASBench-201
- To setup the environment for NASBench-201, run the following command `source setup_nasbench201.sh`
- Run the following commands to compute the correlation for the NASBench-201:
```bash
cd NASBench201/correlation
mkdir data
python NAS_Bench_201.py --start 0 --end 1000 --dataset cifar10
python NAS_Bench_201.py --start 0 --end 1000 --dataset cifar100
python NAS_Bench_201.py --start 0 --end 1000 --dataset ImageNet16-120
```

## NAS Experiments

### DARTS
- Experiments on DARTS search space use the same environment as NAS-Bench-201
- If the NAS-Bench-201 environment is not set up, set it up using the command `source setup_nasbench201.sh`. If it is set up already, then activate the environment using the command `conda activate nasbench201`
- Run the following commands to search in the DARTS space:

```bash
cd NASBench201
bash exp_scripts/zerocostpt_darts_pipeline.sh
```

- Train the searched network on ImageNet using this repository- (https://github.com/chenwydj/DARTS_evaluation)

### AutoFormer
- To setup the environment for AutoFormer experiments, run the following command `source setup_autoformer.sh`
    - Download imagenet in the folder `AutoFormer/imagenet
- Run the following commands for the AutoFormer experiments:

```bash
cd AutoFormer
bash search_autoformer.sh
```
- Train the searched architecture on ImageNet using the command

```bash
bash train_searched_result.sh
```

## Cite this Paper
```bash
@article{
    asthana2025dextr,
    title={Dextr: Zero-Shot Neural Architecture Search with Singular Value Decomposition and Extrinsic Curvature},
    author={Rohan Asthana and Joschua Conrad and Maurits Ortmanns and Vasileios Belagiannis},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2025},
    url={https://openreview.net/forum?id=X0vPof5DVh},
    note={}
}
