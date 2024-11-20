#!/bin/bash
python3 search_autoformer.py --indicator-name dextr  --data-path 'imagenet' --gp \
 --change_qk --relative_position --cfg './experiments/search_space/space-T.yaml' --output_dir './OUTPUT/search'


