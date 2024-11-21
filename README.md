# Dextr

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
- To run the experiments for NASBench-201, run the following command `source setup_nasbench201.sh`
- Run the following commands to compute the correlation for the NASBench-201:
```bash
cd NASBench201/correlation
mkdir output
python NAS_Bench_201.py --start 0 --end 1000 --dataset cifar10
python NAS_Bench_201.py --start 0 --end 1000 --dataset cifar100
python NAS_Bench_201.py --start 0 --end 1000 --dataset ImageNet16-120
```

## NAS Experiments

### DARTS
- Run the following commands to search in the DARTS space:

```bash
cd NASBench201
bash exp_scripts/zerocostpt_darts_pipeline.sh
```

- Train the searched network on ImageNet using this repository- (https://github.com/chenwydj/DARTS_evaluation)

### AutoFormer
- To seutp the environment for AutoFormer experiments, run the following command `source setup_autoformer.sh`
    - Download imagenet in the folder `AutoFormer/imagenet
- Run the following commands for the AutoFormer experiments:

```bash
cd AutoFormer
bash search_autoformer.sh
```
