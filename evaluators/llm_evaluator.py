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
          
        model_class, tokenizer_class = MODEL_CLASSES[model_type]
        self.tokenizer = tokenizer_class.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        self.model = model_class.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        self.generation_config = dict(
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.5,
            max_new_tokens=100
        )
        
        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        self.A_id = self.tokenizer.encode("：A")[-1]
        self.B_id = self.tokenizer.encode("：B")[-1]
        self.C_id = self.tokenizer.encode("：C")[-1]
        self.D_id = self.tokenizer.encode("：D")[-1]


    def eval_subject(self, 
            subject_name, 
            test_df,
            dev_df=None,
            save_result_dir=None,
            do_test=False,
            multiple=False,
            rag=False,
            language="zh"
        ):
        all_answers = {}
        if self.constrained_decoding is True:
            self.generation_config['output_scores'] = True
            self.generation_config['return_dict_in_generate'] = True
            self.generation_config['do_sample'] = False
            self.generation_config['top_k'] = 1
            self.generation_config['max_new_tokens'] = 1
        else:
            self.generation_config['top_p'] = 0.9
            self.generation_config['top_k'] = 40
            self.generation_config['temperature'] = self.temperature
        correct_num = 0
        if save_result_dir:
            result = []
            score = []
        history = self.generate_few_shot_prompt(subject=subject_name, dev_df=dev_df, cot=self.cot, multiple=multiple, language=language)
        if rag == True:
            history = self.generate_rag_few_shot_prompt(subject=subject_name, row=row, multiple=multiple, language=language)
        answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=self.cot, language=language)
            instruction = history + question

            inputs = self.tokenizer(instruction, return_tensors="pt")
            generation_output = self.model.generate(
                inputs.input_ids.to("cuda"),
                **self.generation_config
            )
            batch_size, length = inputs.input_ids.shape
            if multiple == False:
                if self.constrained_decoding is True:
                    logits = generation_output.scores[0][0]

                    logits = logits.float().cpu().detach()
                    choices1_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]]
                    choices2_logits = logits[[self.A_id,self.B_id,self.C_id,self.D_id]]
                    choicesAll_logits = (choices1_logits + choices2_logits).numpy()
                    assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
                    ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
                    response = self.tokenizer.decode([logits.argmax(-1).item()])
                else:
                    response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                    ans, direct_extract = self.extract_answer(row, response)
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
            print("prompt:\n", instruction)
            print("response:     ", response.strip())
            print("ans:          ", ans)
            print("ground truth: ", ground_truth, "\n")
            if save_result_dir:
                result.append(response)
                score.append(correct)
            print(f"\n========================end {str(row_index)}========================")
            
            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output'] = result
            test_df['correctness'] = score
            test_df.to_csv(os.path.join(save_result_dir, f'{subject_name}_test.csv'))

        return correct_ratio, all_answers