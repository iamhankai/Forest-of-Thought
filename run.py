import time
import os
import moxing as mox
import argparse
import shutil
import torch
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import numpy as np

os.system("pip install datasets")
os.system("pip install accelerate")
os.system("pip install pylatexenc")
os.system("pip install graphviz")
os.system("pip install transformers==4.41.2")
os.system("export PYTHONPATH={new_directory}:$PYTHONPATH")

os.system("nvidia-smi")
def parse_args():
    parser = argparse.ArgumentParser('run')
    parser.add_argument("--tree_nums", type=int, default=2, help="")
    parser.add_argument('--mode', type=str, help='run mode', default="game24")
    parser.add_argument("--dataset", type=str, default='gsm8k-llama3-8b-new-mcts-')
    parser.add_argument("--dataset_filepath", type=str, default='./datasets/gsm8k/test.parquet')
    parser.add_argument("--max_iter", type=int, default=1, help="Number of simulations per tree")
    parser.add_argument("--s3_model_path", type=str, default="s3://bucket-4031/bizhenni/projects/chain_of_thought/ckpt/ckpt/mistral-7b-instruct-v0.3")
    parser.add_argument("--model_type", type=str, default="llama")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--stop", choices=["cgdm", "random", "scaling", "majority", "score"], default="cgdm")
    parser.add_argument("--script", type=str, default="run_with_mcf_stop")
    parser.add_argument("--correct", action="store_true")
    parser.add_argument("--correct_threshold",  type=float, default=-1, help="")
    parser.add_argument('--base_mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    parser.add_argument("--start_id", type=int, default=0, help="")
    parser.add_argument("--end_id", type=int, default=-1, help="")
    parser.add_argument("--outputs", type=str, default='')

    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    new_directory = "/home/ma-user/modelarts/user-job-dir/forest-of-thought"
    os.chdir(new_directory)
    mox.file.copy_parallel(args.s3_model_path, f'{new_directory}/ckpt/model')
    print("torch.cuda.is_available()", torch.cuda.is_available())
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            cuda_devices = list(range(device_count))
            cuda_devices_list = list(map(str, cuda_devices))
            os.system(f"export CUDA_VISIBLE_DEVICES={','.join(cuda_devices_list)}")
        else:
            os.system(f"export CUDA_VISIBLE_DEVICES=0")

    cmd_dict = {
        'game24': f'cd {new_directory}/scripts/game24/;python run.py --prompt_sample standard --method_generate propose --tree_num {args.tree_nums} --n_select_sample 5 --correction --model_path {new_directory}/ckpt/model',
        'gsm8k': f'cd {new_directory};python run_with_mcf.py --dataset  gsm8k-new-mcts- --dataset_filepath datasets/gsm8k/test.parquet --tree_nums {args.tree_nums} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --stop scaling' ,
        'aime': f'cd {new_directory};python run_with_mcf.py --dataset  aime-new-mcts- --dataset_filepath datasets/aime2024/aime_2024_problems.parquet --tree_nums {args.tree_nums} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --stop scaling' ,
        'math500': f'cd {new_directory};python run_with_mcf.py --dataset  math-500-new-mcts- --dataset_filepath datasets/math500/test.parquet --tree_nums {args.tree_nums} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --stop scaling' ,
        'math500_sample': f'cd {new_directory};python run_with_mcf.py --dataset  math-500-sample-mcts- --dataset_filepath datasets/math500/test.parquet --tree_nums {args.tree_nums} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --stop scaling --start_id {args.start_id} --end_id {args.end_id}' ,

        'debug': f'cd {new_directory}',
    }

    print(cmd_dict[args.mode])
    # 构建完整的命令
    command = f'export PYTHONIOENCODING=utf8; python {args.script}.py --dataset {args.dataset} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --tree_nums {args.tree_nums} --model_type {args.model_type} --level {args.level} --dataset_filepath {args.dataset_filepath} --stop {args.stop} --base_mode {args.base_mode} --start_id {args.start_id} --end_id {args.end_id} 2>&1 | tee -a outputs/log-{args.dataset}{args.max_iter}-tree-{args.tree_nums}.txt'
    if args.correct:
        command = f'export PYTHONIOENCODING=utf8; python {args.script}.py --dataset {args.dataset} --max_iter {args.max_iter} --model_path {new_directory}/ckpt/model --tree_nums {args.tree_nums} --model_type {args.model_type} --level {args.level} --dataset_filepath {args.dataset_filepath} --correct --correct_threshold {args.correct_threshold}  --start_id {args.start_id} --end_id {args.end_id} 2>&1 | tee -a outputs/log-{args.dataset}{args.max_iter}-tree-{args.tree_nums}.txt'

    # 将命令写入 run.sh 脚本文件
    with open('run.sh', 'w') as file:
        file.write(command)
    os.system(cmd_dict[args.mode])
    # MathBlackBox
    if 'sample' in args.mode:
        outputs_file = f"s3://bucket-4031/bizhenni/projects/chain_of_thought/outputs/forest-of-thought/{args.s3_model_path}/{args.dataset.split('-')[0]}_{args.start_id}_{args.end_id}"
    else:
        outputs_file = f"s3://bucket-4031/bizhenni/projects/chain_of_thought/outputs/forest-of-thought/{args.s3_model_path}/{args.dataset.split('-')[0]}"
    mox.file.copy_parallel(f'/home/ma-user/modelarts/user-job-dir/forest-of-thought/outputs', outputs_file)
    print(f"results saved to {outputs_file}")
    while(1):
        time.sleep(10)
