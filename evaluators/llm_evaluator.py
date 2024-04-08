import os
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from modelscope import snapshot_download
from evaluators.evaluator import Evaluator
MODEL_CLASSES = {
    "llama": (LlamaForCausalLM, LlamaTokenizer),
    "auto": (AutoModelForCausalLM, AutoTokenizer)
}

class LLM_Evaluator(Evaluator):
    def __init__(self, choices, model_type, model_path, k=0, temperature=0.01, constrained_decoding=True, cot=False):
        super().__init__(choices, k)
        self.model_path = model_path
        self.model_type = model_type
        self.temperature = temperature
        self.constrained_decoding = constrained_decoding
        self.cot = cot
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        if "Tongyi-Finance" in self.model_path:
            self.model_path = snapshot_download(self.model_path)
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path, 
            use_fast=False,
            trust_remote_code=True
        )
        self.model = model_class.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        self.generation_config = dict(
            temperature=temperature,
            top_k=10,
            top_p=0.8,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=10
        )
        
        if self.constrained_decoding is True:
            self.generation_config['output_scores'] = True
            self.generation_config['return_dict_in_generate'] = True
            self.generation_config['max_new_tokens'] = 1
            self.generation_config['top_p'] = 1.0
            self.generation_config['top_k'] = 0
        
        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        
    def eval_subject(self, 
            subject_name, 
            test_df,
            dev_df=None,
            save_result_dir=None,
            do_test=False,
            multiple=False,
            dynamic_fs=False,
            language="zh"
        ):
        all_answers = {}
        correct_num = 0
        result = []
        score = []
        if dynamic_fs == False:
            history = self.generate_few_shot_prompt(subject=subject_name, dev_df=dev_df, cot=self.cot, multiple=multiple, language=language)
        answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            if dynamic_fs == True:
                history = self.generate_dynamic_few_shot_prompt(subject=subject_name, row=row, multiple=multiple, language=language)
            question = self.format_example(row, include_answer=False, cot=self.cot, language=language)
            instruction = history + question
            inputs = self.tokenizer(instruction, return_tensors="pt")
            generation_output = self.model.generate(
                input_ids = inputs["input_ids"].to(self.device),
                attention_mask = inputs['attention_mask'].to(self.device),
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_config
            )
            _, length = inputs.input_ids.shape
            if multiple == False:
                if self.constrained_decoding is True:
                    logits = generation_output.scores[0][0]

                    logits = logits.float().cpu().detach()
                    choicesAll_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]].numpy()
                    assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
                    ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
                    response = self.tokenizer.decode([logits.argmax(-1).item()])
                else:
                    response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                    ans = self.extract_answer(row, response)
                if ans == answers[row_index]:
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
                ground_truth = answers[row_index]
            else:
                response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                ans = self.extract_multiple_answer(response)
                ground_truth = {char: int(char in answers[row_index]) for char in 'ABCD'}
                if ans == ground_truth:
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
            print(f"\n========================begin {str(row_index)}========================")
            print("prompt:")
            print(instruction)
            print("response:     ", response.strip())
            print("ans:          ", ans)
            print("ground truth: ", ground_truth, "\n")
            result.append(response)
            score.append(correct)
            print(f"\n========================end {str(row_index)}========================")
            
            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)
        test_df['model_output'] = result
        test_df['correctness'] = score
        test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio, all_answers