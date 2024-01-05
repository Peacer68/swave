export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_PATH=configs/reflow.json
VERBOSE="yes"

python train.py -c $CONFIG_PATH -v $VERBOSE
