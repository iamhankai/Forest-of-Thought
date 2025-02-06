import re
import os
import sympy
import pandas as pd
from tasks.base import Task, DATA_PATH
from utils.prompts.game24 import * 
from collections import Counter


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24_test.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 2 
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        import pdb; pdb.set_trace()
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def propose_prompt_wrap_3_num(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt_3_num.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def propose_prompt_wrap_2_num(x: str, y: str='') -> str:
        last_num = y.split('left:')[-1].strip()[:-1]
        # num1, num2 = map(int, last_num.split())
        if len(last_num.split(' ')) == 2:
            num1, num2 = map(float, last_num.split())
            if num1 + num2 == 24:
                return y.strip() + f'\n{num1} + {num2} = 24\n', 20
            elif abs(num1 - num2) == 24:
                if num1 > num2:
                    return y.strip() + f'\n{num1} - {num2} = 24\n', 20
                else:
                    return y.strip() + f'\n{num2} - {num1} = 24\n', 20
            elif num1 * num2 == 24:
                return y.strip() + f'\n{num1} * {num2} = 24\n', 20
            elif num2 != 0 and num1 / num2 == 24:
                return y.strip() + f'\n{num1} / {num2} = 24\n', 20
            elif num1 != 0 and num2 / num1 == 24:
                return y.strip() + f'\n{num2} / {num1} = 24\n', 20
            else:
                return y, 0.001
        elif len(last_num.split(' ')) == 1:
                result = int(last_num)
                if result == 24:
                    return y, 20
                else:
                    return y, 0.001
        else:
            return y, 0.001
            

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if isinstance(value_outputs, str):
            value_names = [value_outputs]

        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value

    @staticmethod
    def result_correction_3num(input: str, x: str) -> str:
        input_list = input.split(' ')
        multi_instance = x.split('\n')
        result = []
        for ins in multi_instance:
            # ins = ins.split('. **')[-1]
            # ins = ins.replace('*', '').replace(',', '')
            all_numbers = re.findall(r'-?\d+\.?\d*', ins)
            if len(all_numbers) < 6:
                continue
            first_two_numbers = all_numbers[:2]
            first_two_flag = True

            real_left_num = input_list.copy()
            for item in first_two_numbers:
                if item in real_left_num:
                    real_left_num.remove(item)
                else:
                    first_two_flag=False
                    break
            if not first_two_flag:
                continue

            model_input_left_three_numbers = all_numbers[3:]
            correct_left_three_numbers = [all_numbers[2]] + real_left_num
            has_decimal = any(isinstance(num, str) and '.' in num for num in correct_left_three_numbers)
            if has_decimal:
                continue

            check_numbers = re.findall(r'-?\d+\.?\d*', ' '.join(map(str, correct_left_three_numbers)))
            if len(check_numbers) != 3:
                import pdb; pdb.set_trace()
            
            if Counter(model_input_left_three_numbers) == Counter(correct_left_three_numbers):
                result.append(ins)
            else:
                ori = ' '.join(all_numbers[3:])
                tmp = ' '.join(correct_left_three_numbers)
                new_express = ins.replace(ori, tmp)
                result.append(new_express)
 
        return result

    @staticmethod
    def result_correction_2num(input: str, x: str) -> str:

        input_list = re.findall(r'-?\d+\.?\d*', input)[3:]
        multi_instance = x.split('\n')
        result = []
        for ins in multi_instance:
            # ins = ins.split('. **')[-1]
            # ins = ins.replace('*', '').replace(',', '')
            all_numbers = re.findall(r'-?\d+\.?\d*', ins)
            if len(all_numbers) < 5:
                continue
            first_two_numbers = all_numbers[:2]
            first_two_flag = True
            real_left_num = input_list.copy()
            for item in first_two_numbers:
                if item in real_left_num:
                    real_left_num.remove(item)
                else:
                    first_two_flag=False
                    break
            if not first_two_flag:
                continue  
            
            model_input_left_numbers = all_numbers[3:]

            correct_left_numbers = [all_numbers[2]] + real_left_num
            has_decimal = any(isinstance(num, str) and '.' in num for num in correct_left_numbers)
            if has_decimal:
                continue
            check_numbers = re.findall(r'-?\d+\.?\d*', ' '.join(map(str, correct_left_numbers)))
            if len(check_numbers) != 2:
                continue
                
            ori = ' '.join(all_numbers[3:])
            tmp = ' '.join(correct_left_numbers)
            new_express = ins.replace(',', ' ').replace(ori, tmp).replace('...', '')
            result.append(new_express)
            try:
                y, value = Game24Task.propose_prompt_wrap_2_num(input, new_express)
                if value == 20:
                    print('final results:', input+y)
                    return [input+y], True
            except:
                continue
        return [], False