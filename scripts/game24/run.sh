#!/bin/bash
SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PARENT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
echo $PARENT_DIR
export PYTHONPATH=$PARENT_DIR:$PYTHONPATH
MODEL_PATH='/home/ma-user/work/projects/Fot/ckpt/Meta-Llama-3-8B-Instruct'

python run.py --prompt_sample standard --method_generate propose --tree_num 2 --n_select_sample 5 --correction --model_path $MODEL_PATH
# python run.py --prompt_sample standard --method_generate propose --tree_num 4 --n_select_sample 5 --correction --model_path $MODEL_PATH
# python run.py --prompt_sample standard --method_generate propose --tree_num 8 --n_select_sample 5 --correction --model_path $MODEL_PATH
# python run.py --prompt_sample standard --method_generate propose --tree_num 16 --n_select_sample 5 --correction --model_path $MODEL_PATH
