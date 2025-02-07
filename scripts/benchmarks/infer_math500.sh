#!/bin/bash
SCRIPT_PATH=$(realpath "$0")

SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
PARENT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")

echo $PARENT_DIR
cd $PARENT_DIR 
python run_with_mcf.py --dataset  math500-new-mcts- --dataset_filepath datasets/math500/test.parquet