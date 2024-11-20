cd NASBench201/
conda create --name nasbench201 python=3.7
conda activate nasbench201
pip  install torch torchvision --index-url https://download.pytorch.org/whl/cu111
mkdir -p data
cd data
gdown --folder 1dZG-ygQYMajcu5yH6Bh4IdCnlIdUeAVs  #cifar100
gdown --folder 1V_c4J27RqHOkMiDxBfuJI7KJ9bwO9o0y #cifar10
gdown --folder 1Ui1Rw7MVhdLBaVpu1HYh_-5qEqzAu0nI #ImageNet16
cd ../zero-cost-nas/
pip install setuptools==59.5.0 statsmodels xautodl nats_bench
pip install .
cd ..
