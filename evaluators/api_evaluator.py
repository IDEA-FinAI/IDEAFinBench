import os
from tqdm import tqdm
from openai import OpenAI
from evaluators.evaluator import Evaluator
from time import sleep
import re


class API_Evaluator(Evaluator):
    def __init__(self, choices, k, model_name, use_openai=False, cot=False, api_key="EMPTY"):
        super().__init__(choices, k)
        self.choices = choices
        self.k = k
        self.api_key = api_key
        self.cot = cot
        self.model_name=model_name
        if use_openai == False:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
        else:
            self.client = OpenAI(
                api_key=api_key,
            )
    
    def run_llm(self, messages, temperature=0.3):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                temperature=temperature,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            return None
    
    def format_example(self, line, include_answer, language):
        example=line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'

        example += '\n答案：' if language == "zh" else '\nAnswer: '
        if include_answer:
            return [
                {"role":"user", "content": example if not self.cot else example+"\n让我们一步一步思考，\n"},
                {"role":"assistant","content":line["answer"] if not self.cot else line["explanation"]+f"\n所以答案是{line['answer']}。"}
            ] if language=="zh" else [
                {"role":"user", "content": example if not self.cot else example+"\nLet's think step by step,\n"},
                {"role":"assistant","content":line["answer"] if not self.cot else line["explanation"]+f"\nSo the answer is {line['answer']}."}
            ]
        else:
            return [
                {"role": "user", "content": example if not self.cot else example+"\n让我们一步一步思考，\n"}
            ] if language=="zh" else [
                {"role": "user", "content": example if not self.cot else example+"\nLet's think step by step,\n"}
            ]
                
    def generate_few_shot_prompt(self, subject, dev_df, multiple, language):
        if multiple==False:
            prompt=[{
                    "role":"system",
                    "content":self.one_ans_instruct_zh.format(subject=subject) if language=="zh" else self.one_ans_instruct_en.format(subject=subject)
                }]
        else:
            prompt=[{
                    "role":"system",
                    "content":self.multi_ans_instruct_zh.format(subject=subject) if language=="zh" else self.multi_ans_instruct_en.format(subject=subject)
                }]
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True, language=language)
            prompt+=tmp
        return prompt
    
    # RAG-fewshot暂不支持cot，cpa_one_rag和cpa_multi_rag里的Fewshot目前不包含解析用于cot。
    # 暂时仅支持cpa_one_rag和cpa_multi_rag这两份数据集
    def generate_dynamic_few_shot_prompt(self, subject, row, multiple=False, language="zh"):
        if multiple == False:
            prompt=[{"role":"system","content":self.one_ans_instruct_zh.format(subject=subject) if language=="zh" else self.one_ans_instruct_en.format(subject=subject)}]
        else:
            prompt=[{"role":"system","content":self.multi_ans_instruct_zh.format(subject=subject) if language=="zh" else self.multi_ans_instruct_en.format(subject=subject)}]
        for i in range(self.k):
            shot = row[f"shot{i+1}"]
            index = shot.find("答案：")
            if index != -1:
                shot_message = [
                    {"role": "user","content": shot[:index]},
                    {"role": "assistant","content": shot[index:]}
                ]
                prompt += shot_message
            else:
                continue
        return prompt

    def eval_subject(
        self, 
        subject, 
        test_df, 
        dev_df, 
        save_result_dir,
        do_test,
        multiple,
        dynamic_fs,
        language
    ):
        correct_num = 0
        all_answers = {}
        result = []
        score=[]
        prefix_prompt = self.generate_few_shot_prompt(subject=subject, dev_df=dev_df, multiple=multiple, language=language)
        answers = ['NA'] * len(test_df) if do_test is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(),total=len(test_df)):
            if dynamic_fs == True:
                prefix_prompt = self.generate_dynamic_few_shot_prompt(subject, row, multiple=multiple, language=language)
            question = self.format_example(row, include_answer=False, language=language)
            full_prompt = prefix_prompt + question
            response=None
            timeout_counter=0
            while response is None and timeout_counter<=30:
                try:
                    sleep(0.5)
                    response = self.run_llm(
                        messages=full_prompt,
                        temperature=0.3
                    )
                except Exception as msg:
                    if "timeout=600" in str(msg):
                        timeout_counter+=1
                    print(msg)
                    sleep(5)
                    continue
            if response is None:
                correct=0
            if multiple == False:
                ans = self.extract_answer(row, response)
                if ans == answers[row_index]:
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
                ground_truth = answers[row_index]
            else:
                ans = self.extract_multiple_answer(response)
                ground_truth = {char: int(char in answers[row_index]) for char in 'ABCD'}
                if ans == ground_truth:
                    correct_num += 1
                    correct = 1
                else:
                    correct = 0
            print(f"\n========================begin {str(row_index)}========================")
            print("prompt:")
            for item in full_prompt:
                print(item)
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
        test_df.to_csv(os.path.join(save_result_dir, f'{subject}_test.csv'))

        return correct_ratio, all_answers
