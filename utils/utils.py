from functools import lru_cache
import os
import re
import time
from datasets import load_dataset
import pandas as pd
from sympy import symbols, simplify, Rational


def parse_latex_vector(latex_str):
    # 去除左括号和右括号以及反斜杠
    clean_str = latex_str.replace("\\left(", "").replace("\\right)", "")
    
    # 替换LaTeX的\frac{}{}为可以计算的形式
    frac_pattern = r'\\frac\{(\d+)\}\{(\d+)\}'
    clean_str = re.sub(frac_pattern, r'\1/\2', clean_str)
    
    # 分割字符串得到各个元素
    elements = clean_str.split(',')
    
    # 解析每个元素
    parsed_elements = []
    for el in elements:
        el = el.strip()  # 移除可能的额外空格
        try:
            # 检查是否为整数或小数
            num = float(el)
        except ValueError:
            # 如果不是，则尝试作为Rational对象解析
            num = Rational(el).evalf()
        parsed_elements.append(num)
    
    return tuple(parsed_elements)

def mcts_load_data(args):
    data_name = args.dataset
    filename = args.dataset_filepath

    if not filename.endswith('.parquet'):
        df = pd.read_json(filename, lines=True)
        filename = filename.replace('.jsonl', '.parquet')
        # 转换为 Parquet 文件
        df.to_parquet(filename)

    st = time.time()
    dataset = load_dataset('parquet', data_files=filename, split='train')
    end = time.time()
    print(f'load dataset cost {end - st}s')

    if 'math' in data_name and 'level' in data_name:
        dataset = dataset.filter(lambda example: example["level"].endswith(str(args.level)))
    if 'sample' in data_name:
        dataset = dataset.select(range(args.start_id, args.end_id))
    return dataset

def get_query_gt_list(example, data_name):
    if 'gsm8k' in data_name:
        query_list = example['question']
        ground_truth_list = example['answer']
    elif 'math' in data_name:
        query_list = example['problem']
        ground_truth_list = example['answer']
    elif 'aime' in data_name:
        query_list = example['Problem']
        ground_truth_list = example['Answer']
    elif 'hkO1aw' in data_name:
        query_list = example['prompt']
        ground_truth_list = example['answer']
    return query_list, ground_truth_list

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        boxed_pattern = r"\\boxed{([^}]+)}"
        match = re.search(boxed_pattern, pred_str)
        if match:
            boxed_str = str(match.group(1)).replace('{', '').replace('}', '')
        if boxed_str:
            return boxed_str
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

class Extractor:
    def extract_matching_bracket(cls, target_str: str):
        if not target_str:
            return target_str
        current_nest_level = 1
        for i, ch in enumerate(target_str):
            if ch == '{':
                current_nest_level += 1
            elif ch == '}':
                current_nest_level -= 1
            if current_nest_level == 0:
                break
        return target_str[:i]

    def clean(cls, target_str: str):
        opt = target_str.strip().replace('{{', '{').replace('}}', '}')
        if not opt:
            return opt
        if opt[-1] == '.' or opt[-1] == '。':
            return opt[:-1]
        return opt

    def extract_answer(cls, pred: str, extract_last_num=False):
        if pred.find('The final answer is ') >= 0:
            x = pred[pred.find('The final answer is ') +
                     len('The final answer is '):]
            x = x[1:x.find('$.')]
            # print(x)
            return cls.clean(x)
        if pred.find('\n\nQuestion:') >= 0:
            pred = pred.split('\n\nQuestion:')[0]
            if pred.find('The answer is'):
                pred = pred[pred.find('The answer is') + len('The answer is'):]
                return cls.clean(pred)
        if pred.find('# Answer') >= 0:
            return cls.clean(pred[pred.find('# Answer') + len('# Answer'):])
        if pred.find('The answer is:') >= 0:
            return cls.clean(pred[pred.find('The answer is:') +
                                  len('The answer is:'):])
        if pred.find('####') >= 0:
            return cls.clean(pred[pred.find('####') + 4:])
        left = '\\boxed{'
        if pred.find(left) >= 0:
            pred = pred[pred.find(left) + len(left):]
            return cls.clean(cls.extract_matching_bracket(pred))

        if extract_last_num:
            nums = []
            opt = ''

            def contain_digit(opt):
                for ch in opt:
                    if ch.isdigit():
                        return True
                return False

            for ch in pred:
                if ch.isdigit() or ch in ' ,.':
                    opt = opt + ch
                else:
                    if contain_digit(opt):
                        nums.append(opt)
                    opt = ''
            if contain_digit(opt):
                return cls.clean(opt)
            if nums:
                return cls.clean(nums[-1])
        return None
    
pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
extractor_0 = Extractor()
# @lru_cache(1024)
def extract_label(DATA_NAME, text: str, type='') -> str:
    # import pdb; pdb.set_trace()
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
    elif '\\text{ pm}' in text or 'pm' in text:
        text = re.sub(r'\\boxed\{|\\text\{|\}', '', text).split(':')[0]
        return text.replace(' ', '')
    numbers = pattern.findall(text)
    if not numbers:
        return None
    if '\n####' in text or 'The answer is' in text:
        return numbers[0]
    else :
        return numbers[-1]

def get_ans_format(data_name, ground_truth):
    ans_format = r'"[Final Answer] The answer is [answer] \n#### [answer]"'
    return ans_format

def is_digit_followed_by_alpha(s):
    match = re.match(r'^([-+]?\d+)', s)
    if match:
        return str(match.group(1))
    return s

# @lru_cache(1024)
def check(gt, ans, DATA_NAME):
    if DATA_NAME == 'math':
        if str(gt) == str(ans):
            return True
        try:
            gt_tuple = parse_latex_vector(gt)
            pred_tuple = eval(ans)
            if gt_tuple == pred_tuple:
                return True
        except:
            pass
        try:
            if 'x' in gt and 'x' in ans:
                x = symbols('x')
                gt = gt.replace('x', '*x').replace(' ', '')
                ans = ans.replace('x', '*x').replace(' ', '')
                if simplify(gt) == simplify(ans):
                    return True
            elif simplify(gt) == simplify(ans):
                return True
        except:
            pass
        try:
            if float(gt) == float(ans):
                return True
        except:
            pass
    
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

    # gt_label = is_digit_followed_by_alpha(gt_label)
    ans_label = extract_label(DATA_NAME, ans)
    # ans_label = is_digit_followed_by_alpha(ans_label)
    if ans_label:
        if type == 'option':
            ans_label = ans_label.strip()[0]
        elif type == 'yesorno':
            ans_label = ans_label.lower()
        elif type == 'formula':
            ans_label = ans_label.replace('$','')
    print(gt_label, ans_label)
    if 'gsm' not in DATA_NAME and type != 'digit': # math
        return is_equiv(gt_label,ans_label)
    print(gt_label, ans_label)
    if gt_label is None or ans_label is None:
        return False
    try:
        if ans_label == gt_label or abs(float(ans_label) - float(gt_label)) < 1e-5:
            return True
        else:
            return False
    except:
        return False

def fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string

def fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set)
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string
    
def fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set)
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()  # noqa: W605

    # a_b == a, a_{b} == a_b for bit conversion
    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    # 10800 == 10,800; we only deal with single number
    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

if __name__ == '__main__':

    gt = "106^\circ"
    ans = '106'
    import pdb; pdb.set_trace()
    aa = check(gt, ans, 'gsm8k')
    print(aa)