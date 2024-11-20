cd NASLib/
sed -i 's/Cython==0.29.37/Cython==3.0.11/' requirements.txt
pip install --upgrade pip setuptools wheel
pip install -e .
pip install tornado
scripts/bash_scripts/download_data.sh all
gdown 1BV_BRMsCUVBtSVj4SN4QmA9Pjd35B2M4
mv transnas-bench_v10141024.pth naslib/data/
gdown 1oORtEmzyfG1GcnPHh0ijCs0gCHKEThNx
mv nasbench_only108.pkl naslib/data/
cd naslib/data/
gdown 1YJ80Twt9g8Gaf8mMgzK-f5hWaVFPlECF