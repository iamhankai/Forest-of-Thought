from collections import Counter
import copy
import math
import os
import re
import hashlib
import json
import random
import numpy as np
import random
import argparse

from models.load_local_model import Pipeline
from utils.examples import get_examples, get_similarity_question
from utils.utils import *


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--tree_nums", type=int, default=2, help="")
    args.add_argument("--max_iter", type=int, default=1, help="Number of simulations per tree")
    args.add_argument("--model_path", type=str, default="/home/ma-user/work/projects/Fot/ckpt/Qwen2.5-Math-7B-Instruct")
    args.add_argument("--model_type", type=str, default="qwen")
    args.add_argument("--dataset", type=str, default='aime-sample-new-mcts-')
    args.add_argument("--level", type=int, default=1)
    args.add_argument("--dataset_filepath", type=str, default='datasets/aime2024/aime_2024_problems.jsonl')
    args.add_argument("--output_file", type=str, default="results.json")
    args.add_argument("--output_dir", type=str, default="outputs")
    args.add_argument("--start_id", type=int, default=0, help="")
    args.add_argument("--end_id", type=int, default=10, help="")
    args.add_argument("--dynamic_self_correction", action="store_true", help="if true, use dynamic_self_correction")
    args.add_argument("--correct_threshold",  type=float, default=-1, help="")
    args.add_argument("--stop", choices=["cgdm", "random", "scaling", "majortiy", "score"], default="scaling")
    args.add_argument('--base_mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    args = args.parse_args()
    return args

def generate(prompt,history=[], truncate=True):
    # global client
    history_ = [{"role": "user" if i % 2 ==0 else 'assistant', "content": h} for i,h in enumerate(history)]
    if truncate:
        history_ = history_[-2:]
    
    messages = history_ + [
            {"role": "user", "content": prompt}
        ]

    content = client.get_respond(messages)[0]
    return content, list(history)+[prompt, content]

def cal_reward(question,ans):
    query = f'Question: {question}\nAnswer:{ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [0,100], ig. from 0 to 100. \nResponse format:\n[Analyst]...[Score]...'
    ret = generate(query)
    score = ret[0].split('Score')[-1]
    scores = pattern.findall(score)
    if not scores:
        return 0
    else:
        ret = float(scores[-1])
        return ret 

def get_weak_answer(question, new_len=0, ans_format=''):
    query = f'Question: {question}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet\'s think step by step.'
    ans = generate(query)[0]
    prompt = question + ans + "\nTherefore, the answer is"
    return generate(prompt)
    # return generate(query)

def get_weak_hints(question,weak_answer,ground_truth_label=None,new_len=0,history=[],alreadygood=False,ans_format=''):
    query = f'Question: {question}\nSince we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return generate(query, history)

def get_better_answer(question,weak_answer,hint,new_len=0,history=[],ans_format=''):
    query = f'Question: {question}\nPlease refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet\'s think step by step.'
    ans = generate(query, history)[0]
    prompt = question + ans + "\nTherefore, the answer is"
    return generate(prompt)

def filter_mature_node(childs, to_explore, to_explore_reward, max_expand=3):
    filterd_to_explore = []
    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    for node in to_explore:
        if len(childs.get(node,[])) < max_expand or max([avg_reward.get(child,-999) for child in childs.get(node,[])]) < avg_reward.get(node,-999):
            filterd_to_explore.append(node)
    
    return filterd_to_explore

def get_best_explore_from_ucb(to_explore, ucb_bank):
    # 初始化最佳节点和最高UCB值
    best_node = None
    highest_ucb = float('-inf')
    
    # 遍历所有待探索的节点
    for node in to_explore:
        ucb_value = ucb_bank.get(node, float('-inf'))
        if ucb_value > highest_ucb:
            highest_ucb = ucb_value
            best_node = node      
    return best_node

def compute_ucb(r_c, N_n, N_c, C):
    return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))

def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4, gamma=0.85):
    # 计算所有节点的访问次数
    visit_count = {node: len(to_explore_reward[node]) for node in to_explore}

    # 计算所有节点的平均奖励
    # avg_reward = {node: sum(to_explore_reward[node]) / len(to_explore_reward[node]) for node in to_explore}
    avg_reward = {node: (min(to_explore_reward[node]) + np.mean(to_explore_reward[node])) / 2 for node in to_explore}

    # 获取所有叶子节点
    leaves = set(to_explore) - set(fathers.values())
    
    # 更新所有叶子节点的UCB值
    for leaf in leaves:
        # ucb_bank[leaf] = avg_reward[leaf]
        ucb_bank[leaf] = compute_ucb(avg_reward[leaf],len(to_explore_reward.get(fathers.get(leaf,None),[])),len(to_explore_reward.get(leaf,[])),C)
    
    # 从叶子节点向上更新父节点的UCB值
    nodes_to_update = list(leaves)
    while nodes_to_update:
        new_nodes_to_update = set()
        for node in nodes_to_update:
            father = fathers.get(node)
            if father is not None:
                if father not in ucb_bank:
                    new_nodes_to_update.add(father)
                if father in ucb_bank:
                    # 计算父节点的UCB值
                    ucb_values = []
                    child_reward = []
                    for child in childs[father]:
                        ucb_values.append(ucb_bank[child])
                        child_reward.append(avg_reward[child])
                    if child_reward:
                        father_reward = (avg_reward[father] + max(child_reward))/2
                    else:
                        father_reward = avg_reward[father]
                    ucb_bank[father] = compute_ucb(father_reward,len(to_explore_reward.get(fathers.get(father,None),[])),len(to_explore_reward.get(father,[])),C)
        nodes_to_update = list(new_nodes_to_update)

def step(query,weak_answer,ground_truth_label=None,history=[],alreadygood=False,ans_format=''):
    hints, history = get_weak_hints(query,weak_answer,ground_truth_label=ground_truth_label,history=history,alreadygood=alreadygood,ans_format=ans_format)
    answer, history = get_better_answer(query,weak_answer,hints,history=history,ans_format=ans_format)
    return hints,answer,history

def monte_carlo_tree(DATA_NAME, query, ground_truth, max_iter=2, ans_format=''):
    is_correct = False

    to_explore = []
    to_explore_reward = {}
    history_bank = {}
    hints_bank = {}
    ucb_bank = {}
    fathers = {}
    childs = {}
    def sampling_reward(answer):
        if answer not in to_explore_reward:
            to_explore_reward[answer] = []
        reward = cal_reward(query,answer)
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(hints,weak_answer):
        if weak_answer not in hints_bank:
            hints_bank[weak_answer] = []
        hints_bank[weak_answer].append(hints)

    def add_to_childs(father,child):
        if father not in childs:
            childs[father] = []
        childs[father].append(child)

    hints_reward_imp_bank = {}
    def add_to_hints_reward_imp_bank(hints,weak_answer,reward,answer):
        if weak_answer not in hints_reward_imp_bank:
            hints_reward_imp_bank[weak_answer] = []
        hints_reward_imp_bank[weak_answer].append((hints, reward, answer))

    weak_answer, history = get_weak_answer(query, ans_format=ans_format)

    history_bank[weak_answer] = tuple(history)
    answers_list = [weak_answer,]
    to_explore = [weak_answer,]
    childs[weak_answer] = []
    fathers[weak_answer] = None

    ##add total-bad answer###
    hints_list = []
    if check(ground_truth, weak_answer, DATA_NAME):
        is_correct = True
        return is_correct, hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank 

    sampling_reward(weak_answer)

    if True: # not check(ground_truth,weak_answer):
        total_bad = random.choice(["I Don't Know","I can't understand this question.","I can't help with this question.","I don't know how to solve this question.","I don't know the answer to this question.","I don't know the answer to this question, sorry."])
        total_bad_history = copy.deepcopy(history)
        total_bad_history[-1] = total_bad
        history_bank[total_bad] = tuple(total_bad_history)
        answers_list += [total_bad,]
        to_explore += [total_bad,]
        childs[total_bad] = []
        fathers[total_bad] = None
        sampling_reward(total_bad)
    
    update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
    for i in range(max_iter):
        print('iteration:', i)
        filterd_to_explore = filter_mature_node(childs, to_explore, to_explore_reward)
        weak_answer = get_best_explore_from_ucb(filterd_to_explore, ucb_bank)
        sampling_reward(weak_answer)
        hints,answer,history = step(query,weak_answer,history=history_bank[weak_answer],ans_format=ans_format)
        add_to_hints_bank(hints,weak_answer)
        history_bank[answer] = tuple(history)
        to_explore.append(answer)
        sampling_reward(answer)
        fathers[answer] = weak_answer
        childs[answer] = []
        add_to_childs(weak_answer,answer)
        answers_list.append(answer)
        hints_list.append(hints)
        if check(ground_truth,answer,DATA_NAME):
            is_correct = True
            return is_correct, hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank

        update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
        add_to_hints_reward_imp_bank(hints,weak_answer,min(to_explore_reward.get(answer)) - min(to_explore_reward.get(weak_answer)),answer)#ucb_bank[answer] - ucb_bank[weak_answer]

    return is_correct, hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank

def get_ans_format(DATA_NAME, ground_truth):
    ans_format = r'"[Final Answer] The answer is [answer] \n#### [answer]"'
    return ans_format

class Monte_Carlo_Forest():
    def __init__(self, args) -> None:
        self.max_iter = args.max_iter
        self.dataset = args.dataset
        self.data_name = f"{args.dataset}{args.max_iter}-{os.path.basename(args.model_path)}-tree-{args.tree_nums}"
        self.outputs = f'{args.output_dir}/{self.data_name}/jsons/'
        self.tree_nums = args.tree_nums
        self.correct_num = 0
        self.stop = args.stop
        self.trees_ans = []
        self.learning_cases = get_examples().get(args.dataset.split('-')[0])

    def run(self, example):
        query_list, ground_truth_list = get_query_gt_list(example, self.data_name)
        ans_format = get_ans_format(self.data_name, ground_truth_list)
        data_list = [] 
        index = 0 
        for query, ground_truth in zip(query_list, ground_truth_list):
            index += 1
            print(f"index:{index}")
            tree_list = []
            for t in range(self.tree_nums):
                if t > 0: # slow thinking
                    cases_list = np.array(self.learning_cases)[:,0].tolist()
                    query_case = get_similarity_question(query, cases_list)
                    query_index = np.array(self.learning_cases)[:,0].tolist().index(query_case)
                    case = self.learning_cases[query_index]
                    query = 'Question:' + case[0] + 'Answer:' + case[1] + '\n' + 'Question:' + query
                else: # quick thinking/IO
                    pass
                
                max_iter = min(t+1, self.max_iter)
                max_iter = self.max_iter
                is_correct, hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank = monte_carlo_tree(self.dataset, query, str(ground_truth), max_iter=max_iter, ans_format=ans_format)
                
                data = {
                    'query':query,
                    'ground_truth':ground_truth,
                    'hints_list':hints_list,
                    'answers_list':answers_list,
                    'to_explore':to_explore,
                    'to_explore_reward':to_explore_reward,
                    'hints_bank':hints_bank,
                    'history_bank':history_bank,
                    'hints_reward_imp_bank':hints_reward_imp_bank,
                    'fathers':fathers,
                    'childs':childs,
                    'ucb_bank':ucb_bank,
                    'correct_num':self.correct_num,
                    'current_num':index,
                    'is_correct':is_correct,
                    'tree_num': t,
                }
                tree_list.append({'tree_num': t, 'data':data})
                self.trees_ans += answers_list
                if is_correct and self.stop == 'scaling': 
                    self.correct_num += 1
                    break
                
                if t >= 1:
                    fot_ans_lst = [x for tree_id, x in enumerate(self.trees_ans) if x is not None] 
                    counter = Counter(fot_ans_lst)
                    max_count = max(counter.values())  
                    if max_count > self.tree_nums / 2:
                        self.fot_early_stop_signal = True 
                        break
      
            data_list.append({'current_num':index, 'outputs': tree_list})
            print(f'correct_num={self.correct_num}, current_num={index}')
        os.makedirs(self.outputs, exist_ok=True)
        with open(f'{self.outputs}/{hashlib.md5(str(example).encode()).hexdigest()}.json','w+', encoding="utf-8") as f:
            json.dump(data_list, f, indent=4, ensure_ascii=False)

        return data_list

if __name__ == '__main__':

    args = parse_args()
    model_path = args.model_path

    global client
    client = Pipeline(model_path, args.model_type, args.dynamic_self_correction, args.correct_threshold)

    dataset = mcts_load_data(args)
    monte_carlo_forest = Monte_Carlo_Forest(args)
    datas = monte_carlo_forest.run(dataset)
