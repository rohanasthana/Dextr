#!/bin/bash
python3 search_autoformer.py --indicator-name dextr --data-path '/home/hu15nagy/Documents/ImageNet_data/ImageNet_data' --gp \
 --change_qk --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


