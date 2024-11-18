# Dextr

## Correlation Experiments

### NASBench101, 301 and TransNASBench-micro

- We use NASLib (https://github.com/automl/NASLib/tree/zerocost) for our correlation experiments on NASBench101, NASBench301, and TransNASBench-micro.

- To reproduce our results for NB101, NB301 and TNB101-micro, follow the instructions from NASLib (https://github.com/automl/NASLib/tree/zerocost) to set up the experiment.

- Copy all the subfolders of NASLib to the folder /NASLib/

- Our proxy Dextr is implemented in the file NASLib/naslib/predictors/pruners/measures/dextr.py

- Run the correlation experiments using scripts from the folder- NASLib/scripts/cluster/benchmarks

Eg- 
> bash NAS_suite/NASLib/scripts/cluster/benchmarks/run_tnb101.sh correlation dextr

### NASBench-201

- Follow the MeCo repository (https://github.com/HamsterMimi/MeCo) to set up the environment for our NASBench201 experiments.
- Run the following commands for NASBench201 experiments 
> cd NASBench201/correlation
> python NAS_Bench_201.py
