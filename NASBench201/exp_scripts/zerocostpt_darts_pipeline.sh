#!/bin/bash
script_name=`basename "$0"`
id=${script_name%.*}
dataset=${dataset:-cifar10}
seed=${seed:-999}
gpu=${gpu:-"0"}
pool_size=${pool_size:-10}
space=${space:-s5}
metric=${metric:-'dextr'}
edge_decision=${edge_decision:-'random'}
validate_rounds=${validate_rounds:-10}
learning_rate=${learning_rate:-0.025}
while [ $# -gt 0 ]; do
    if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
        # echo $1 $2 // Optional to see the parameter:value result
    fi
    shift
done

echo 'id:' $id 'seed:' $seed 'dataset:' $dataset 'space:' $space
echo 'proj crit:' $metric
echo 'gpu:' $gpu

pwd
cd  sota/cnn
python3 networks_proposal.py \
    --search_space $space --dataset $dataset \
    --batch_size 16 \
    --seed $seed --save $id --gpu $gpu \
    --edge_decision $edge_decision \
    --proj_crit_normal $metric --proj_crit_reduce $metric --proj_crit_edge $metric \
    --pool_size $pool_size\

cd ../../zerocostnas/
python3 post_validate.py\
    --ckpt_path ../experiments/sota/$dataset-search-$id-$space-$seed-$pool_size-$metric\
    --save $id --seed $seed --gpu $gpu\
    --batch_size 64\
    --edge_decision $edge_decision --proj_crit $metric \
    --validate_rounds $validate_rounds\
