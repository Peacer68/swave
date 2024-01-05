export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_PATH=configs/finetune.json
VERBOSE="yes"

python finetune.py -c $CONFIG_PATH -v $VERBOSE
