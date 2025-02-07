import os
import json
import argparse
import time
from tasks import get_task
from methods.bfs import forest_solve, naive_solve
from models.load_local_model import Pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def load_local_model(args):
    llm = Pipeline(args.model_path, task=args.task)
    return llm

def run(args):
    model = load_local_model(args)
    correct_num = 0
    task = get_task(args.task)
    logs = []
    file = f'./logs/game24/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_start{args.task_start_index}_end{args.task_end_index}_tree{args.tree_num}.json'
    os.makedirs(os.path.dirname(file), exist_ok=True)
    print("results saved to", file)
    total_time = 0
    start_time = time.time()
    for i in range(args.task_start_index, args.task_end_index):
        print('index:', i+1)
        if args.naive_run:
            ys, info, decode_counts = naive_solve(args, task, i, model, to_print=True)
        else:
            ys, info, decode_counts = forest_solve(args, task, i, model) 
        if ys:
            correct_num += 1
            print('ys:', ys)
        print('index:', i + 1, 'correct_num:', correct_num, 'correct_avg:', correct_num / (args.task_end_index - args.task_start_index) * 100)
        
        total_time = time.time() - start_time
        info.update({'idx': i, 'correct_num': correct_num, 'decode_counts': decode_counts, 'total_time': total_time})
        logs.append(info)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)
        
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str, default='llama3_8b')
    args.add_argument('--task', type=str, default='game24')
    args.add_argument('--model_path', type=str, default='/home/ma-user/work/projects/Fot/ckpt/Meta-Llama-3-8B-Instruct')
    args.add_argument('--temperature', type=float, default=0.7)

    args.add_argument('--task_start_index', type=int, default=0)
    args.add_argument('--task_end_index', type=int, default=96)

    args.add_argument('--naive_run', action='store_true') # if True base method is CoT, else ToT
    args.add_argument('--correction', action='store_true', help="correction") 
    args.add_argument('--prompt_sample', type=str, choices=['standard', 'cot'], default="standard") 

    args.add_argument('--method_select', type=str, choices=['sample', 'greedy'], default='greedy')
    args.add_argument('--n_evaluate_sample', type=int, default=5)  # only thing needed if naive_run
    args.add_argument('--n_select_sample', type=int, default=5)  # 5

    args.add_argument('--tree_num', type=int, default=2)  # 5
    args = args.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)