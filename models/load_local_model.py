import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

system_prompt = {"mistral": {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
                 "qwen":{"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."}}

class Pipeline:
    def __init__(self, model_id="", model_type='llama', correction=False, correct_threshold=-1, dataname='gsm8k', task='benchmark'):
        self.model_id = model_id # "./ckpt/Meta-Llama-3-8B-Instruct"
        print(f'load model weight {self.model_id}...')
        self.device = "cuda"
        self.model_name = os.path.basename(self.model_id).lower()
        self.correction = correction
        self.correct_threshold = correct_threshold
        self.task = task
        if 'qwen' in self.model_name.lower():
            self.model_type = 'qwen'
        elif 'llama' in self.model_name.lower():
            self.model_type = 'llama'
        elif 'glm' in self.model_name.lower():
            self.model_type = 'glm'
        elif 'deepseek' in self.model_name.lower():
            self.model_type = 'deepseek'
        else:
            self.model_type = model_type
        if self.model_type == 'mistral' or self.task== 'game24':
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                model_kwargs={"torch_dtype": torch.float16},
                device_map='auto',
                trust_remote_code=True,
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map='auto',
            ).eval()

        self.infer_times = 0
        self.tokens_per_second_sum = 0
        self.dataname = dataname

    def get_respond(self, messages, max_length=1024):
        self.infer_times += 1
        if self.model_type == 'mistral' or self.task== 'game24':
            cur_system_prompt = system_prompt['mistral']
            if isinstance(messages, list):
                messages = [cur_system_prompt] + messages
            terminators = [
                self.pipeline.tokenizer.eos_token_id,
                self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            try:
                max_length = 1024 
                outputs = self.pipeline(
                    messages,
                    max_new_tokens=max_length,
                    eos_token_id=terminators,
                    temperature=0.95,
                    do_sample=True,
                    pad_token_id = self.pipeline.tokenizer.eos_token_id,
                )
            except ValueError as e:
                if "Input length of input_ids is" in str(e) and "but `max_length` is set to" in str(e):
                    max_length = int(str(e).split("Input length of input_ids is ")[1].split(",")[0])
                    max_length = max_length + 100  # Add some buffer
                    outputs = self.pipeline(
                        messages,
                        max_new_tokens=max_length,
                        eos_token_id=terminators,
                        temperature=0.95,
                        do_sample=True,
                        pad_token_id = self.pipeline.tokenizer.eos_token_id,
                    )
                else:
                    raise
            except Exception as e:
                # Handle other exceptions if necessary
                print(f"An unexpected error occurred: {e}")
                return "", -10

            response = outputs[0]["generated_text"][len(messages):].split('Input:')[0].strip()
            return response, 0
        else:
            if 'glm' == self.model_type:
                return self.get_respond_glm(messages, max_length)
            elif 'llama' == self.model_type:
                return self.get_respond_llama(messages, max_length)
            elif 'qwen' == self.model_type:
                return self.get_respond_qwen(messages, max_length)
            elif 'deepseek' in self.model_type:
                return self.get_respond_deepseek(messages, max_length)
            elif 'HK-O1aw' == self.model_type:
                return self.get_respond_HK_O1aw(messages, max_length)

    def get_respond_llama(self, messages, max_new_tokens=1024):
        inputs = self.tokenizer.apply_chat_template(messages,
                                                add_generation_prompt=True,
                                                tokenize=True,
                                                return_tensors="pt",
                                                return_dict=True
                                                )
        inputs = inputs.to(self.device)
        start_time = time.time()
        with torch.no_grad():
            try:
                gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            except:
                return "", -10

            end_time = time.time()
            inference_time = end_time - start_time
            num_tokens = generated_ids.shape[-1]
            self.tokens_per_second_sum += num_tokens / inference_time

            generated_ids, average_confidence = self.self_correction(messages, generated_ids)
            outputs = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return response, average_confidence
       
    def get_respond_glm(self, messages, max_length=1024):
        inputs = self.tokenizer.apply_chat_template(messages,
                                            add_generation_prompt=True,
                                            tokenize=True,
                                            return_tensors="pt",
                                            return_dict=True
                                            )

        inputs = inputs.to(self.device)

        with torch.no_grad():
            try:
                gen_kwargs = {"max_length": max_length, "do_sample": True, "top_k": 1}
                generated_ids = self.model.generate(**inputs, **gen_kwargs)
            except ValueError as e:
                if "Input length of input_ids is" in str(e) and "but `max_length` is set to" in str(e):
                    input_length = int(str(e).split("Input length of input_ids is ")[1].split(",")[0])
                    gen_kwargs["max_length"] = input_length + 100  # Add some buffer
                    generated_ids = self.model.generate(**inputs, **gen_kwargs)
                else:
                    raise
            except Exception as e:
                # Handle other exceptions if necessary
                print(f"An unexpected error occurred: {e}")
                return ""

            if self.correction and 'Please refine the your answer according to your Reflection or Feedback.' not in messages[0]['content']:
                generated_ids = self.self_correction(messages, generated_ids)
            outputs = generated_ids[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        return response
    
    def get_respond_qwen(self, messages, max_length):

        messages[0]['content'] = messages[0]['content'].split('The response should begin with ')[0]
        cur_system_prompt = system_prompt['qwen']
        messages = [cur_system_prompt] + messages
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        start_time = time.time()
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.8,
            temperature=0.7,
        )
        end_time = time.time()
        inference_time = end_time - start_time
        num_tokens = generated_ids.shape[-1]
        self.tokens_per_second_sum += num_tokens / inference_time
    
        generated_ids, average_confidence = self.self_correction(messages, generated_ids)
        outputs = generated_ids[:, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response, average_confidence

    def get_respond_deepseek(self, messages, max_length):
        
        if 'The response should begin with' in messages[0]['content']:
            messages[0]['content'] = messages[0]['content'].split('The response should begin with ')[0]
            messages[0]['content'] += '\nPlease reason step by step, and put your final answer within \\boxed{}."'
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1024
        )

        if self.correction:
            generated_ids = self.self_correction(messages, generated_ids)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def self_correction(self, messages, generated_ids):
        input_ids = generated_ids[:, :-1]  

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # 获取每个 token 的 log probabilities
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=-1)

        # 获取生成的 token 的 log probabilities
        generated_log_probs = torch.gather(log_probabilities, 2, generated_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        average_confidence = generated_log_probs.mean().item()

        # print(f"Average confidence: {average_confidence}")
        if self.correction and average_confidence < self.correct_threshold:
            ans_format = r'"[Final Answer] The answer is [answer] \n#### [answer]"'
            query = f'Please refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with {ans_format}\nLet\'s think step by step.'
            messages[0]['content'] = messages[0]['content'].split('Please reason step by step')[0].strip() + query
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            new_generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024
            )

            new_input_ids = new_generated_ids[:, :-1] 

            with torch.no_grad():
                new_outputs = self.model(new_input_ids)
                new_logits = new_outputs.logits

            new_log_probabilities = torch.nn.functional.log_softmax(new_logits, dim=-1)
            new_generated_log_probs = torch.gather(new_log_probabilities, 2, new_generated_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
            new_average_confidence = new_generated_log_probs.mean().item()
            
            if new_average_confidence > average_confidence:
                # print(f"New average confidence: {new_average_confidence}")
                average_confidence = new_average_confidence
                generated_ids = new_generated_ids

        return generated_ids, average_confidence
    
    def generate(self, prompt, history=[], truncate=True, max_new_tokens=1024):
        # history_ = [{"role": "user" if i % 2 ==0 else 'assistant', "content": h} for i,h in enumerate(history)]
        # if truncate:
        #     history_ = history_[-2:]
        
        # messages = history_ + [
        #         {"role": "user", "content": prompt}
        #     ]

        # content, _ = self.get_respond(messages, max_new_tokens)
        content, _ = self.get_respond(prompt, max_new_tokens)
        return content 