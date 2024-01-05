export CUDA_VISIBLE_DEVICES=0,1,2,3

CONFIG_PATH=configs/reflow-k.json
VERBOSE="yes"

python reflow-distill.py -c $CONFIG_PATH -v $VERBOSE
