import itertools
import random
import re
import numpy as np
from functools import partial
from collections import Counter


def check_numbers(num_list_str, x):
    """
    检查等号左边的数字和 left 后面的数字是否在给定的数字列表中。

    :param num_list: 输入字符串，格式为 ["-6 + 1 = -5 (left: 1 24)",]
    :param input_str: 数字列表字符串，格式为 "4 5 6 10"
    :return: 返回一个布尔值，表示是否所有数字都在数字列表中
    """
    for input_str in num_list_str:
        # 解析等号左边的数字
        equation_part = input_str.split('(')[0].strip()
        left_side = equation_part.split('=')[0].strip()
        left_numbers = [int(num) for num in left_side.replace('+', ' ').replace('-', ' -').split() if num.lstrip('-').isdigit()]

        # 解析 left 后面的数字
        left_info = input_str.split('left:')[1].split(')')[0].strip()
        for num in left_info.split():
            try:
                left_numbers.extend([int(num) * -1])
            except:
                break

        # 解析数字列表
        num_list = [int(num) for num in x.split()]

        # 使用 Counter 统计每个数字的出现次数
        left_counter = Counter(left_numbers)
        num_list_counter = Counter(num_list)

        # 比较两个 Counter 是否相等
        if left_counter == num_list_counter:
            return True, input_str
    return False, ""


def check_expression(input_string, expression, target_result=24):
    # 提取输入字符串中的数字
    input_numbers = set(map(int, input_string.split()))
    
    # 移除表达式中的空格和等号右边的部分
    expression_left_side = ''.join(expression.split())[:-len(f"={target_result}")]  # 假设等号右边没有空格
    
    # 提取表达式左边的所有数字
    expr_numbers = re.findall(r'\d+', expression_left_side)
    if len(expr_numbers) != 4:
        return False
    # 检查表达式中的所有数字是否都在输入字符串提供的数字集合中
    all_digits_from_input = all(number in input_numbers for number in expr_numbers)
    
    # 评估表达式的值
    try:
        expr_result = eval(expression_left_side)
    except Exception as e:
        print(f"Error evaluating expression: {e}")
        return False
    
    # 检查表达式的值是否等于目标结果
    is_correct = expr_result == target_result
    
    # 返回两个条件都满足的结果
    return all_digits_from_input and is_correct

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    
    value_outputs = llm.generate(value_prompt)
    value_outputs = value_outputs.split('\n')[-1]
    print(f"y={y.strip()}, value_outputs={value_outputs}")
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}  # 去重
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value  # 去重
        values.append(value)
    return values

def get_proposals_task1(args, task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)  # get few shot
    if 'gpt' in args.backend:
        response = llm(propose_prompt, n=1, stop=None)[0].split('\n')
    else:
        response = llm.generate(propose_prompt)
    # print("Before correction:\n", response)
    if not args.correction:
        proposals = response.split('\n')
    else:
        proposals = task.result_correction_3num(x, response) 
        print(f"After correction:\n{proposals}\n\n")
    # while not proposals:  
    #     response = llm.generate(propose_prompt)
    #     if not args.correction:
    #         proposals = response.split('\n')
    #     else:
    #         proposals = task.result_correction_3num(y, response)  
    #     if proposals:
    #         break
    
    return [y.strip() + _.strip() + '\n' for _ in proposals]

def get_proposals_task2(args, task, x, y): 
    propose_prompt = task.propose_prompt_wrap_3_num(x, y)  # get few shot
    if 'gpt' in args.backend:
        proposals = llm(propose_prompt, n=1, stop=None)[0].split('\n')
    else:
        response = llm.generate(propose_prompt)
    if not args.correction:
        proposals = response.split('\n')
        flag = True
    else:
        proposals, flag = task.result_correction_2num(y, response)  
    if flag:
        return proposals  
    else:
        return []

def get_proposals_task3(task, x, y): 
    try:
        propose_prompt, value = task.propose_prompt_wrap_2_num(x, y)  # get few shot
    except:
        propose_prompt, value = y, 0.001
    return propose_prompt, value

def get_samples(task, x, y, prompt_sample, stop):
    if prompt_sample == 'standard': # IO
        prompt = task.standard_prompt_wrap(x, y) 
    elif prompt_sample == 'cot': # COT
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')

    samples = llm.generate(prompt)
    results = samples.split('Answer:')[-1]
    return results

# cot
def naive_solve(args, task, idx, model='None', to_print=True):
    global llm
    llm = model
    if 'gpt' in args.backend:
        llm = partial(llm, model=args.backend, temperature=args.temperature)

    original_string = task.get_input(idx)  # input
    number_list = init_numbers(original_string, args.tree_num)
    for t in range(0, args.tree_num):
        x = number_list[t]
        print(f"input:{x}")

        ys = get_samples(task, x, '', args.prompt_sample, stop=None)
        result = check_expression(x, ys)
        print(f"The expression:{ys}, is valid:", result)
        if result:
            return ys, {}, llm.infer_times

    return '', {}, llm.infer_times

def init_numbers(x, count):
    number_list = [x]
    numbers = x.split()
    # 生成随机顺序的数字
    for i in range(count - 1):
        random.shuffle(numbers)  # 随机打乱顺序
        shuffled_string = ' '.join(numbers)
        # print(f"第 {i+1} 组: {shuffled_string}")
        number_list.append(shuffled_string)
    return number_list
    

def forest_solve(args, task, idx, model='None', to_print=True):
    global llm
    llm = model
    if 'gpt' in args.backend:
        llm = partial(llm, model=args.backend, temperature=args.temperature)

    original_string = task.get_input(idx)  # input
    number_list = init_numbers(original_string, args.tree_num)
    for t in range(0, args.tree_num):
        x = number_list[t]
        print(f"input:{x}")
        ys = ['']  # current output candidates
        infos = []
        for step in range(task.steps):  # 3
            # generation
            if step == 0:
                new_ys = [get_proposals_task1(args, task, x, y) for y in ys]  
            elif step == 1:
                new_ys = [get_proposals_task2(args, task, x, y) for y in ys]  
            
            if step == 0:
                new_ys = list(itertools.chain(*new_ys))
                ids = list(range(len(new_ys))) 
                # evaluation
                values = get_values(task, x, new_ys, args.n_evaluate_sample)  
            
                # selection
                if args.method_select == 'sample':
                    ps = np.array(values) / sum(values)
                    select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
                elif args.method_select == 'greedy':
                    select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]  # 选择值最大的结果 n_select_sample=5
                select_new_ys = [new_ys[select_id] for select_id in select_ids]

                # log
                try: 
                    sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                    print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
                except:
                    continue
                infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
                ys = select_new_ys
            else:  
                for new_res in new_ys:
                    if len(new_res) > 0:
                        print('--final new_ys --:', new_res)
                        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': 20, 'select_new_ys': new_res})
                        return new_res, {'steps': infos}, llm.infer_times
    
    return '', {'steps': infos}, llm.infer_times