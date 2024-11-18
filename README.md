# Dextr

## Correlation Experiments

- We use NASLib (https://github.com/automl/NASLib/tree/zerocost) for our correlation experiments.

- To reproduce our results, follow the instructions from NASLib (https://github.com/automl/NASLib/tree/zerocost) to set up the experiment.

- Copy all the folders to the folder /NASLib/

- Our proxy Dextr is implemented in the file NASLib/naslib/predictors/pruners/measures/dextr.py

- Run the correlation experiments using scripts from the folder- NASLib/scripts/cluster/benchmarks

Eg- 
> bash NAS_suite/NASLib/scripts/cluster/benchmarks/run_tnb101.sh correlation dextr
