#!/bin/bash
python3 train.py --data-path '/home/hu15nagy/Documents/ImageNet_data/ImageNet_data' --gp --change_qk --relative_position \
--mode retrain --model_type 'AUTOFORMER' --dist-eval --cfg './experiments/subnet_autoformer/TF_TAS-T.yaml' --output_dir './OUTPUT/sample'


