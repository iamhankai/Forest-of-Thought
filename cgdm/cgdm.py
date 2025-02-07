import json
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
from models.load_local_model import Pipeline
from utils.utils import is_equiv, extract_label


MODEL_PATH = "/home/ma-user/work/projects/Fot/ckpt/QwQ-32B-Preview"
client = Pipeline(MODEL_PATH, 'qwen')

def generate(prompt, history=None, truncate=True):
    """生成回复"""
    if history is None:
        history = []
    history_ = [{"role": "user" if i % 2 == 0 else 'assistant', "content": h} for i, h in enumerate(history)]
    messages = (history_[-2:] if truncate else history_) + [{"role": "user", "content": prompt}]
    content = client.get_respond(messages, max_length=4096)
    return content, history + [prompt, content]

def get_best_answer(question, all_answer_str):
    """获取最佳答案"""
    query = f"You are a highly specialized mathematics expert. Question: {question}\nAnswers:{all_answer_str}\nThe best answer is [answer]\n"
    return generate(query)

def get_new_answer(question):
    """获取新答案"""
    return generate(question)

def main(input_file="need_model_judge.json", output_file="model_choose_results.json"):
    """主函数"""
    correct_num = 0
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line = line.replace(': null', ": 'null'")
            line = line.replace(': false', ": 'false'")
            line = line.replace(': true', ": 'true'")
            label = eval(line.strip())
            question, gt_label, answers_list = label['query'], label['gt_label'], label['pred_ans']
            if answers_list:
                formatted_answers = '\n'.join([f'Answer {id}: {ans}' for id, ans in enumerate(answers_list)])
                expert_ans, _ = get_best_answer(question, formatted_answers)
            else:
                expert_ans, _ = get_new_answer(question)
            
            best_ans = extract_label('math', expert_ans)
            
            is_equal = is_equiv(gt_label, best_ans)
            print(f'gt_label={gt_label}, best_ans={best_ans}, is_equal={is_equal}')
            result = {'query': question, 'gt': gt_label, 'model_choice': expert_ans, 'ex_pred': best_ans, 'is_correct': str(is_equal)}
            json.dump(result, f_out, ensure_ascii=False)
            f_out.write('\n')
            correct_num += is_equal
        print(f'correct_num={correct_num}')

if __name__ == '__main__':
    main()