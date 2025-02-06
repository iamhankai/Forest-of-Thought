import copy
# from curses.ascii import isalpha, isdigit
import math
# import multiprocessing as mp
# import torch
import os
import re
import hashlib
import json
import random
from functools import lru_cache
import numpy as np
import random
import argparse

from models.load_local_model import Pipeline
from utils.examples import get_examples, get_similarity_question
from utils.early_stop import *
from utils.utils import *


def generate(prompt, history=[], truncate=True):
    # global client
    history_ = [{"role": "user" if i % 2 ==0 else 'assistant', "content": h} for i, h in enumerate(history)]
    if truncate:
        history_ = history_[-2:]
    
    messages = history_ + [
            {"role": "user", "content": prompt}
        ]

    content, scores = client.get_respond(messages)
    return content, list(history)+[prompt, content], scores

def cal_reward(question,ans):
    # query = f'Question: {question}\nAnswer:{ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [-100,+100], ig. from -100 to +100. \nResponse format:\n[Analyst]...[Score]...'
    query = f'Question: {question}\nAnswer:{ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score! You need to be very harsh and mean in calculating grades, and never give full marks to ensure that the marks are authoritative. \nOutput a score between [0,100], ig. from 0 to 100. \nResponse format:\n[Analyst]...[Score]...'
    ret = generate(query)
    score = ret[0].split('Score')[-1]
    scores = pattern.findall(score)
    if not scores:
        return 0
    else:
        ret = float(scores[-1])
        # if abs(ret - 100.0) < 1e-5:
        #     ret = 50.0
        # if ret >= 95:
        #     ret = 50
        return ret 

def get_weak_answer(question,new_len=0,ans_format=''):
    query = f'Question: {question}\nThe response should begin with [reasoning process]...[Verification]... and end with {ans_format}\nLet\'s think step by step.'
    return generate(query)[:2]

def get_weak_hints(question,weak_answer,ground_truth_label=None,new_len=0,history=[],alreadygood=False,ans_format=''):
    query = f'Question: {question}\nSince we have a weak Answer, could you provide me with a relection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for ervery possible imperfect to minus every possible score!\nLet\'s think step by step.'
    return generate(query, history)[:2]

def get_better_answer(question,weak_answer,hint,new_len=0,history=[],ans_format=''):
    query = f'Question: {question}\nPlease refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet\'s think step by step.'
    return generate(query, history)[:2]

def get_best_answer(question, all_answer_str):
    ans_format = 'The best answer is [answer]\n'
    query = f"You are a highly specialized mathematics expert, proficient in solving mathematical problems, and always capable of selecting the most correct answer from the given options. \
        \Question: {question}\nAnswers:{all_answer_str}\nWhich of the following answers is the most correct? The response should begin with {ans_format}"
    return generate(query)

def get_cot_answer(question, ans_format=''):
    query = f'Question: {question}\nLet\'s think step by step.'
    return generate(query)[:1]

def get_final_answer(query):
    return generate(query)[:2]

datas = []
pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
extractor_0 = Extractor()
@lru_cache(1024)
def extract_label(DATA_NAME, text: str, type='') -> str:
    if 'gsm' not in DATA_NAME and type != 'digit':
        if '####' in text:
            text = text.split('####')[-1]
        elif 'The answer is' in text:
            text = text.split('The answer is')[-1]
            if '####' in text:
                text = text.split('####')[-1]
        if 'box' in text:
            return extract_boxed_answer(text)
        else:
            return text
    if '\n####' in text:
        text = text.split('\n####')[-1].replace(',','')
    elif 'The answer is' in text:
        text = text.split('The answer is')[-1].replace(',','')
    if 'box' in text:
        return extract_boxed_answer(text)
    numbers = pattern.findall(text)
    if not numbers:
        return None
    if '\n####' in text or 'The answer is' in text:
        return numbers[0]
    else :
        return numbers[-1]

@lru_cache(1024)
def check(gt, ans, DATA_NAME):
    gt = str(gt)
    gt_label = extract_label(DATA_NAME, gt)
    if gt_label.isdigit():
        type = 'digit'
    elif gt_label.isupper() and gt_label.isalpha():
        type = 'option'
    elif gt_label.lower() in ['yes','no']:
        gt_label = gt_label.lower()
        type = 'yesorno'
    else :
        type = 'formula'
    ans_label = extract_label(DATA_NAME, ans)
    if ans_label:
        if type == 'option':
            ans_label = ans_label.strip()[0]
        elif type == 'yesorno':
            ans_label = ans_label.lower()
        elif type == 'formula':
            ans_label = ans_label.replace('$','')
    print(gt_label,ans_label)
    if 'gsm' not in DATA_NAME and type != 'digit':
        return is_equiv(gt_label,ans_label)
    print(gt_label,ans_label)
    if gt_label is None or ans_label is None:
        return False
    try:
        if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
            return True
        else:
            return False
    except:
        return False


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

def update_ucb(fathers, childs, to_explore, to_explore_reward, ucb_bank, C=1.4,gamma=0.85):
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
                    father_reward = (avg_reward[father] + max(child_reward))/2
                    ucb_bank[father] = compute_ucb(father_reward,len(to_explore_reward.get(fathers.get(father,None),[])),len(to_explore_reward.get(father,[])),C)
        nodes_to_update = list(new_nodes_to_update)

def step(query,weak_answer,ground_truth_label=None,history=[],alreadygood=False,ans_format=''):
    hints,history = get_weak_hints(query,weak_answer,ground_truth_label=ground_truth_label,history=history,alreadygood=alreadygood,ans_format=ans_format)
    answer,history = get_better_answer(query,weak_answer,hints,history=history,ans_format=ans_format)
    return hints,answer,history

def get_tree_ans(to_explore_reward, ucb_bank, answers_list):
    def weighted_score(answer):
        # 假设你根据奖励、访问次数、UCB等计算综合得分
        reward_score = min(to_explore_reward[answer]) * 0.5  # 奖励占50%
        visit_count = len(to_explore_reward[answer]) * 0.3  # 访问次数占30%
        ucb_score = ucb_bank.get(answer, 0) * 0.2  # UCB值占20%
        return reward_score + visit_count + ucb_score

    best_answer = max(answers_list, key=weighted_score)
    return best_answer

def monte_carlo_tree(DATA_NAME, query, max_iter=16, ans_format=''):
    is_activate = True

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
        reward = cal_reward(query, answer)
        to_explore_reward[answer].append(reward)

    def add_to_hints_bank(hints, weak_answer):
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
        hints_reward_imp_bank[weak_answer].append((hints,reward,answer))

    ###get weak answer###
    weak_answer, history = get_weak_answer(query, ans_format=ans_format)
    
    history_bank[weak_answer] = tuple(history)
    activated_answers_list = []
    activated_answer_scores = []

    answers_list = [weak_answer,]
    to_explore = [weak_answer,]
    childs[weak_answer] = []
    fathers[weak_answer] = None

    sampling_reward(weak_answer)
    activated_answer_scores.append(to_explore_reward[weak_answer])
    if to_explore_reward[weak_answer] > 95:
        is_correct = True
        # return is_correct, hints_list,answers_list,to_explore,to_explore_reward,hints_bank,history_bank,hints_reward_imp_bank,fathers,childs,ucb_bank 
        return weak_answer, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank


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
        weak_answer = get_best_explore_from_ucb(filterd_to_explore, ucb_bank)  # selection
        sampling_reward(weak_answer) # similation
        # extend
        hints, answer, history = step(query,weak_answer,history=history_bank[weak_answer],ans_format=ans_format)
        add_to_hints_bank(hints, weak_answer)
        history_bank[answer] = tuple(history)
        to_explore.append(answer)
        sampling_reward(answer)
        fathers[answer] = weak_answer
        childs[answer] = []
        add_to_childs(weak_answer, answer)
        
        extract_ans = extract_label(DATA_NAME, answer)
        if extract_ans:
            activated_answers_list.append(extract_ans)
            activated_answer_scores.append(to_explore_reward[answer])
        answers_list.append(answer)

        update_ucb(fathers=fathers,childs=childs,to_explore=to_explore,to_explore_reward=to_explore_reward,ucb_bank=ucb_bank)
        add_to_hints_reward_imp_bank(hints,weak_answer,min(to_explore_reward.get(answer)) - min(to_explore_reward.get(weak_answer)),answer)#ucb_bank[answer] - ucb_bank[weak_answer]

    tree_ans = get_tree_ans(to_explore_reward, ucb_bank, answers_list)
    return tree_ans, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank

class Monte_Carlo_Forest():
    def __init__(self, args) -> None:
        self.max_iter = args.max_iter
        self.dataset = args.dataset
        self.data_name = f"{args.dataset}{args.max_iter}-{os.path.basename(args.model_path)}-tree-{args.tree_nums}"
        self.outputs = f'{args.output_dir}/{self.data_name}/jsons/'
        self.tree_nums = args.tree_nums
        self.correct_num = 0
        self.learning_cases = get_examples()[args.dataset.split('-')[0]]
        self.activate_signal=[True] * self.tree_nums
        self.stop_stategy = args.stop
        self.trees_score = [None] * self.tree_nums
        self.trees_ans = [None] * self.tree_nums
        self.fot_early_stop_signal = False
        self.expert_memory = dict()
        self.base_mode = args.base_mode

    def run(self, example):
        os.makedirs(self.outputs, exist_ok=True)
        print(f"results save to {self.outputs}/{hashlib.md5(str(example).encode()).hexdigest()}.json")

        if self.base_mode == 'mcts':
            return self.mctsr_run(example)
        elif self.base_mode == 'cot':
            return self.cot_run(example)
        elif self.base_mode == 'tot':
            return self.tot_run(example)
        else:
            raise 'bad base mode!'
        
    def mctsr_run(self, example):
        fw = open(f'{self.outputs}/{hashlib.md5(str(example).encode()).hexdigest()}.json','w+', encoding="utf-8")

        query_list, ground_truth_list = get_query_gt_list(example, self.data_name)
        ans_format = get_ans_format(self.data_name, ground_truth_list)
        data_list = [] 
        index = 0 
        for query, ground_truth in zip(query_list, ground_truth_list):
            self.fot_early_stop_signal = False
            index += 1
            print(f"index:{index}")
            
            total_answers_list = []
            # best_ans = ''
            for t in range(self.tree_nums):
                if t > 0: # slow thinking input diversity
                    cases_list = np.array(self.learning_cases)[:,0].tolist()
                    query_case = get_similarity_question(query, cases_list)
                    query_index = np.array(self.learning_cases)[:,0].tolist().index(query_case)
                    case = self.learning_cases[query_index]
                    query = 'Question:' + case[0] + 'Answer:' + case[1] + '\n' + 'Question:' + query
                else: # quick thinking = IO
                    pass
                max_iter = min(t+1, self.max_iter)
                tree_ans, is_activate, activated_answers_list, activated_answer_scores, answers_list, to_explore, to_explore_reward, ucb_bank = monte_carlo_tree(self.dataset, query, max_iter=max_iter, ans_format=ans_format)
                self.trees_ans[t] = tree_ans
                total_answers_list.append(answers_list)
                if t >= 1:
                    fot_ans_lst = [x for tree_id, x in enumerate(self.trees_ans) if x is not None and self.activate_signal[tree_id]]
                    try:
                        counter = Counter(fot_ans_lst)
                        max_count = max(counter.values())  
                        if max_count > self.tree_nums / 2:
                            self.fot_early_stop_signal = True
                            fot_ans = max(counter, key=counter.get)
                            break
                    except:
                        pass

            if not self.fot_early_stop_signal: # 提前终止得到答案
                fot_ans = self.get_fot_final_answer(query, self.trees_ans, total_answers_list, self.trees_score, t, fot=True)
            is_correct = False
            if fot_ans is not None and check(ground_truth, fot_ans, self.dataset):
                self.correct_num += 1
                is_correct = True

            print(f'trees_ans={self.trees_ans}')
            print(f'correct_num={self.correct_num}, current_num={index}')
            # save results
            data = {
                'query':query,
                'ground_truth':ground_truth,
                'fot_ans': fot_ans,
                'activated_answers_list':activated_answers_list,
                'activated_answer_scores': activated_answer_scores,
                'answers_list':total_answers_list,
                'trees_ans':self.trees_ans,
                'to_explore':to_explore,
                'to_explore_reward':to_explore_reward,
                'ucb_bank':ucb_bank,
                "is_correct": is_correct,
                'correct_num': self.correct_num,
                'current_num': index,
                'is_activate': is_activate,
                'tree_nums': self.tree_nums,
            }

            data_list.append({'index':index, 'data':data})
            json.dump({'index':index, 'data':data}, fw, indent=4, ensure_ascii=False)
            fw.write('\n')
            
        fw.close()
        return data_list

    def tot_run(self, example):
        fw = open(f'{self.outputs}/{hashlib.md5(str(example).encode()).hexdigest()}.json','w+', encoding="utf-8")

        query_list, ground_truth_list = get_query_gt_list(example, self.data_name)
        from methods.tot.task import ToT_Task
        from utils.visualize import visualize
        
        data_list = [] 
        index = 0 
        for query, ground_truth in zip(query_list, ground_truth_list):
            self.fot_early_stop_signal = False
            index += 1
            print(f"index:{index}")
            
            total_answers_list = []
            # best_ans = ''
            for t in range(self.tree_nums):
                
                Task = ToT_Task(query, propose_method=client, value_method=client)
                output, root = Task.run()
                visualize(root, Task, 'tot', f'tot_{t}', index + 1)
                
                self.trees_ans[t] = extract_label(self.dataset,  output['summary'])
                total_answers_list.append(output['content'])
                
                if t >= 1:
                    fot_ans_lst = [x for tree_id, x in enumerate(self.trees_ans) if x is not None and self.activate_signal[tree_id]]
                    try:
                        counter = Counter(fot_ans_lst)
                        max_count = max(counter.values())  # 找到最高的出现次数
                        if max_count > self.tree_nums / 2:
                            self.fot_early_stop_signal = True # 节省资源
                            fot_ans = max(counter, key=counter.get)
                            break
                    except:
                        pass

            if not self.fot_early_stop_signal: # 提前终止得到答案 
                fot_ans = self.get_fot_final_answer(query, self.trees_ans, total_answers_list, self.trees_score, t, fot=True)
            is_correct = False
            if fot_ans is not None and check(ground_truth, fot_ans, self.dataset):
                self.correct_num += 1
                is_correct = True

            print(f'trees_ans={self.trees_ans}')
            print(f'correct_num={self.correct_num}, current_num={index}')
            # save results
            data = {
                'query':query,
                'ground_truth':ground_truth,
                'fot_ans': fot_ans,
                'answers_list':total_answers_list,
                'trees_ans':self.trees_ans,
                "is_correct": is_correct,
                'correct_num': self.correct_num,
                'current_num': index,
                'tree_nums': self.tree_nums,
            }

            data_list.append({'index':index, 'data':data})
            json.dump({'index':index, 'data':data}, fw, indent=4, ensure_ascii=False)
            fw.write('\n')
            
        fw.close()
        return data_list
    
    def cot_run(self, example):
        fw = open(f'{self.outputs}/{hashlib.md5(str(example).encode()).hexdigest()}.json','w+', encoding="utf-8")

        query_list, ground_truth_list = get_query_gt_list(example, self.data_name)
        ans_format = get_ans_format(self.data_name, ground_truth_list)
        data_list = [] 
        index = 0 
        for query, ground_truth in zip(query_list, ground_truth_list):
            self.fot_early_stop_signal = False
            index += 1
            print(f"index:{index}")
            
            total_answers_list = []
            total_scores_list = []
            # best_ans = ''
            for t in range(self.tree_nums):
                if t > 0: # slow thinking input diversity
                    cases_list = np.array(self.learning_cases)[:,0].tolist()
                    query_case = get_similarity_question(query, cases_list)
                    query_index = np.array(self.learning_cases)[:,0].tolist().index(query_case)
                    case = self.learning_cases[query_index]
                    query = 'Question:' + case[0] + 'Answer:' + case[1] + '\n' + 'Question:' + query
                else: # quick thinking = IO
                    pass
                tree_ans = get_cot_answer(query)
                self.trees_ans[t] = extract_label(self.dataset, tree_ans[0])
                total_answers_list.append(tree_ans[0])
                total_scores_list.append(tree_ans[-1])
                # early stop
                if t >= 1:
                    fot_ans_lst = [x for tree_id, x in enumerate(self.trees_ans) if x is not None and self.activate_signal[tree_id]]
                    try:
                        counter = Counter(fot_ans_lst)
                        max_count = max(counter.values())  # 找到最高的出现次数
                        if max_count > self.tree_nums / 2:
                            self.fot_early_stop_signal = True # 节省资源
                            fot_ans = max(counter, key=counter.get)
                            break
                    except:
                        pass

            if not self.fot_early_stop_signal: # 提前终止得到答案  
                fot_ans = self.get_fot_final_answer(query, self.trees_ans, total_answers_list, [], t, fot=True)
            is_correct = False
            if fot_ans is not None and check(ground_truth, fot_ans, self.dataset):
                self.correct_num += 1
                is_correct = True

            print(f'trees_ans={self.trees_ans}')
            print(f'correct_num={self.correct_num}, current_num={index}')
            # save results
            data = {
                'query':query,
                'ground_truth':ground_truth,
                'fot_ans': fot_ans,
                'answers_list': total_answers_list,
                'scores_list': total_scores_list,
                'trees_ans':self.trees_ans,
                "is_correct": is_correct,
                'correct_num': self.correct_num,
                'current_num': index,
                'base_mode': self.base_mode,
                'tree_nums': self.tree_nums,
            }

            data_list.append({'index':index, 'data':data})
            json.dump({'index':index, 'data':data}, fw, indent=4, ensure_ascii=False)
            fw.write('\n')
            
        fw.close()
        return data_list
    
    def get_fot_final_answer(self, query, activated_answers_list, total_answers_list, activated_answer_scores=[], t=-1, fot=False):
        if self.stop_stategy == 'cgdm':
            most_common_elements = most_frequent_elements(activated_answers_list)
            if len(most_common_elements) == 1:
                best_ans = most_common_elements[0]
            else:
                origin_question = query.split('Question:')[-1]
                if origin_question in self.expert_memory.keys():
                    best_ans = self.expert_memory[origin_question]
                else:
                    answers_list = ''
                    for id, ans in enumerate(total_answers_list):
                        answers_list += f'Answer {id}: {ans}\n'
                    expert_ans, _, scores = get_best_answer(origin_question, answers_list)
                    try:
                        best_ans = expert_ans.split('The best answer is ')[-1]
                        match = re.search(r'\d+', best_ans)
                        if match:
                            number = int(match.group())
                            if number < len(total_answers_list):
                                best_ans = activated_answers_list[number]
                    except:
                        ans = activated_answers_list[0]
                        prompt = origin_question + ans + "\nTherefore, the answer is"
                        best_ans = get_final_answer(prompt)
                        best_ans = extract_label(self.dataset, best_ans)
                    self.expert_memory[origin_question] = best_ans
            if best_ans in ["I Don't Know", "I can't understand this question.", "I can't help with this question.", "I don't know how to solve this question.", "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]:
                best_ans = get_cot_answer(query)
                best_ans = extract_label(self.dataset, best_ans)
                
        elif self.stop_stategy == 'random':
            best_ans = random.choice(activated_answers_list)
        elif self.stop_stategy == 'score':
            index_list = highest_score_elements(activated_answer_scores)
            if len(index_list) == 1:
                best_score_index = index_list[0]
                best_ans = activated_answers_list[best_score_index]
            else:
                best_score_index = index_list[-1]
                best_ans = activated_answers_list[best_score_index]
            if not fot:
                self.trees_score[t] = activated_answer_scores[best_score_index]
        return best_ans


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--tree_nums", type=int, default=2, help="")
    args.add_argument("--max_iter", type=int, default=1, help="Number of simulations per tree")
    args.add_argument("--model_path", type=str, default="/home/ma-user/work/projects/Fot/ckpt/HK-O1aw")
    args.add_argument("--model_type", type=str, default="qwen")
    args.add_argument("--dataset", type=str, default='aime-new-mcts-')
    args.add_argument("--level", type=int, default=1)
    args.add_argument("--dataset_filepath", type=str, default='./datasets/hkO1aw/test.jsonl')
    args.add_argument("--output_dir", type=str, default="outputs")
    args.add_argument("--stop", choices=["cgdm", "random", "majority", "score"], default="cgdm")
    args.add_argument("--dynamic_self_correction", action="store_true")
    args.add_argument("--correct_threshold",  type=float, default=-0.5, help="")
    args.add_argument('--base_mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    args.add_argument("--start_id", type=int, default=0, help="")
    args.add_argument("--end_id", type=int, default=10, help="")
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    global client
    client = Pipeline(args.model_path, args.model_type, args.dynamic_self_correction)

    dataset = mcts_load_data(args)
    monte_carlo_forest = Monte_Carlo_Forest(args)

    datas = monte_carlo_forest.run(dataset)

    print("infer_times=", client.infer_times)
    print("tokens_per_second=", client.tokens_per_second_sum/client.infer_times)
