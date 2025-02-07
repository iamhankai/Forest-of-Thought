#!/bin/bash
SCRIPT_PATH=$(realpath "$0")

SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PARENT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

echo $PARENT_DIR
cd $PARENT_DIR 
python run_with_mcf.py --dataset  gsm8k-new-mcts- --dataset_filepath datasets/gsm8k/test.parquet