import json
import re
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from utils.utils import check, extract_label, is_equiv
from utils.early_stop import most_frequent_elements, highest_score_elements

def process_answer(ans):
    """处理并提取答案标签"""
    if not ans:
        return ans
    ans_label = extract_label('math', ans)
    if ans_label and '$' in str(ans_label):
        ans_label = ans_label.replace('$','').replace('\\','')
    return ans_label

def determine_best_answer(tree_ans, to_explore_reward):
    """确定最佳答案"""
    most_common_elements = most_frequent_elements(tree_ans)
    if len(most_common_elements) == 1:
        return most_common_elements[0]
    else:
        return None # tree_ans[-1]
    
def process_label(label, fw):
    raw_answers_list = []
    to_explore_reward = dict()
    """处理每个标签"""
    for label_per_tree in label:
        query = label_per_tree['query']
        ground_truth = label_per_tree['ground_truth']
        raw_answers_list += label_per_tree['answers_list']
        to_explore_reward.update(label_per_tree['to_explore_reward']) 
        gt_label = process_answer(ground_truth)
        
    answers_list = [extract_label('math', ans) for ans in raw_answers_list]
    tree_ans = [process_answer(ans) for ans in answers_list if ans]

    best_ans = determine_best_answer(tree_ans, to_explore_reward)
    if best_ans in ["I Don't Know", "I can't understand this question.", "I can't help with this question.", "I don't know how to solve this question.", "I don't know the answer to this question.", "I don't know the answer to this question, sorry."]:
        best_ans = None
    if best_ans is None:
        handle_need_model_judge(query, gt_label, tree_ans, label_per_tree, fw)
        return None
    
    gt_label = clean_label(gt_label)
    final_ans = clean_label(best_ans)

    final_ans, gt_label = adjust_gt_label_if_needed(final_ans, gt_label)
    return final_ans, gt_label

def classify_gt_label(gt_label):
    """分类标签类型"""
    if gt_label.isdigit():
        return 'digit'
    elif gt_label.isupper() and gt_label.isalpha():
        return 'option'
    elif gt_label.lower() in ['yes','no']:
        return 'yesorno'
    else:
        return 'formula'

def clean_label(label):
    """清理标签中的特殊字符"""
    return re.sub(r'\\text\{([^}]*)\}', r'\1', label)# .replace('$', '') # .replace('\\', '')

def adjust_gt_label_if_needed(final_ans, gt_label):
    if final_ans:
        final_ans = re.sub(r'\\text\{([^}]*)\}', r'\1', final_ans).replace(' ', '')
        gt_label = re.sub(r'\\text\{([^}]*)\}', r'\1', gt_label).replace(' ', '')
        
    """根据需要调整gt_label"""
    if final_ans.isdigit() and not gt_label.isdigit():
        numbers = re.findall(r'\d+', gt_label)
        if numbers:
            gt_label = str(numbers[0])
    return final_ans, gt_label

def handle_need_model_judge(query, gt_label, judge_ans, raw_label, fw):
    """处理需要模型判断的情况"""
    need_model_judge = {'query': query, 'gt_label': gt_label, 'pred_ans': judge_ans, 'raw_label': raw_label}
    json.dump(need_model_judge, fw, ensure_ascii=False)
    fw.write('\n')

def main(input_file):
    gt_ans = []
    dataset = '/home/ma-user/work/projects/Fot/code/MathBlackBox-main/datasets/math/test.jsonl'
    with open(dataset, 'r') as f:
        labels = f.readlines()[:1000]
        for label in labels:
            label = eval(label)
            gt_ans.append(label['answer'])
            
    correct_num = 0
    badcase_list = []
    with open("need_model_judge.json", 'w') as fw, open(input_file, 'r') as f:
        labels = json.load(f)
        for idx, label in enumerate(labels):
            is_equal = False
            result = process_label(label, fw)
            if result:
                final_ans, gt_label = result
                gt_label = gt_ans[idx]
                if '\\mbox' in gt_label:
                    gt_label = gt_label.split('\\mbox')[0]
                else:
                    gt_label = process_answer(gt_label) # here! attention
                print(f"final_ans={final_ans}")
                print(f"gt_label={gt_label}")
                try:
                    final_ans, gt_label = adjust_gt_label_if_needed(final_ans, gt_label)
                    is_equal = check(gt_label, final_ans, 'math')
                except:
                    is_equal = check(gt_label, final_ans, 'math')
                print(f"is_equal={is_equal}")
                
                if is_equal:
                    correct_num += 1
                else:
                    badcase_list.append({'query':label[0]['query'], 'gt_label':gt_label, 'pred_ans':final_ans})
    print('badcase length=', len(badcase_list))
    print('labels length=', len(labels))
    print(f'correct_num={correct_num}')
    with open('badcase_list.json', 'w') as f:
        json.dump(badcase_list, f, ensure_ascii=False)
        

if __name__ == '__main__':
    input_file = "73ec6992a00cc9f20a2ae46780e9b06b.json"
    main(input_file) 

