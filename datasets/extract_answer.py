import json

input_file_list = [
    # "aime2024/aime_2024_problems.jsonl",
    # "gsm8k/test.jsonl",
    "math500/test.jsonl",
    # "math/test.jsonl",
    # "",
]
for input_file in input_file_list:
    answers_and_type = []
    with open(input_file, 'r') as f:
        labels = f.readlines()
        for label in labels:
            label = eval(label)
            if 'aime' in input_file:
                answer = label['Answer']
            else:
                answer = label['answer']
            if 'gsm8k' in input_file:
                answer = answer.split('\n####')[-1].strip()
            # import pdb; pdb.set_trace()
            try:
                answer = str(answer)
                answer_is_digit = 1 if answer.isdigit() else 0
                answers_and_type.append(answer + '\t' + str(answer_is_digit))
            except:
                print(answer, input_file)
            
    open(f'{input_file}_answer.jsonl', 'w').write('\n'.join(answers_and_type))
