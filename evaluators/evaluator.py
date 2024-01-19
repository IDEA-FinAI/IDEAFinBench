import re
import random
import os
import json
import time
import pandas as pd

class Evaluator:
    def __init__(self, choices, k=-1):
        self.choices = choices
        self.k = k
        self.one_ans_instruct_zh = "以下是关于{subject}考试的单项选择题，请选出其中的一个正确答案。"
        self.multi_ans_instruct_zh = "以下是关于{subject}考试的多项选择题，请选出其中的多个正确答案。"
        self.one_ans_instruct_en = "The following is a question about {subject} with only one answer, please select the correct choice."
        self.multi_ans_instruct_en = "The following is a question about {subject} with multiple answers, please select all the correct choices."

    def eval_subject(self):
        pass
    
    def main_process(self, args):
        assert os.path.exists(args.data_dir + "/subject_mapping.json"), "subject_mapping.json not found!"
        with open(args.data_dir + "/subject_mapping.json") as f:
            subject_mapping = json.load(f)
        subject_list = list(subject_mapping.keys())
        accuracy, summary = {}, {}

        run_date=time.strftime('%Y-%m-%d_%H-%M-%S',time.localtime(time.time()))
        output_dir = args.output_dir
        save_result_dir=os.path.join(output_dir,f"take1")
        if not os.path.exists(save_result_dir):
            os.makedirs(save_result_dir,exist_ok=True)

        all_answers = {}
        for index,subject_name in enumerate(subject_list):
            print(f"{index/len(subject_list)} Inference starts at {run_date} on {args.model_path} with subject of {subject_name}!")
            val_file_path=os.path.join(args.data_dir + '/val', f'{subject_name}_val.csv')
            dev_file_path=os.path.join(args.data_dir + '/dev', f'{subject_name}_dev.csv')
            test_file_path=os.path.join(args.data_dir + '/test', f'{subject_name}_test.csv')

            # 如果do_test是False，则使用val集，否则使用test集
            val_df=pd.read_csv(val_file_path) if args.do_test is False else pd.read_csv(test_file_path)
            dev_df=pd.read_csv(dev_file_path) if args.shots > 0 else None

            correct_ratio, answers = self.eval_subject(
                subject_name, 
                val_df, 
                dev_df,
                save_result_dir=save_result_dir if args.do_save_csv else None,
                do_test=args.do_test,
                multiple=args.multiple,
                rag=args.rag,
                language=args.language
            )
            print(f"Subject: {subject_name}")
            print(f"Acc: {correct_ratio}")
            accuracy[subject_name] = correct_ratio
            summary[subject_name] = {"score":correct_ratio,
                                    "num":len(val_df),
                                    "correct":correct_ratio*len(val_df)/100}
            all_answers[subject_name] = answers

        total_num = 0
        total_correct = 0
        category_list = list(value[2] for value in subject_mapping.values())
        summary['grouped'] = { key :{"correct":0.0,"num":0} for key in category_list}
        for subj, info in subject_mapping.items():
            group = info[2]
            summary['grouped'][group]["num"] += summary[subj]['num']
            summary['grouped'][group]["correct"] += summary[subj]['correct']
        for group, info in summary['grouped'].items():
            info['score'] = 100*info["correct"] / info["num"]
            total_num += info["num"]
            total_correct += info["correct"]
        summary['All'] = {"score": 100 * total_correct / total_num, "num": total_num, "correct": total_correct}

        json.dump(all_answers,open(save_result_dir+'/submission.json','w'),ensure_ascii=False,indent=4)

        print('-' * 80)
        print("Accuracy_subject:")
        for k, v in accuracy.items():
            formatted_v = "{:.2f}".format(v)
            print(k, ": ", formatted_v)
        print('-' * 80)
        print("Accuracy_grouped:")
        for k, v in summary['grouped'].items():
            formatted_score = "{:.2f}".format(v['score']) 
            print(k, ": ", formatted_score)

        print("Avg: ")
        formatted_avg = "{:.2f}".format(summary['All']['score'])
        print(formatted_avg)
        
        json.dump(summary,open(save_result_dir+'/summary.json','w'),ensure_ascii=False,indent=2)    
    
    def format_example(self, line, include_answer=True, cot=False, language="zh"):
        example = line['question']
        for choice in self.choices:
            example = str(example) + f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += ("\n答案：让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。\n\n") if language == "zh" \
                    else ("\nAnswer：Let's think step by step,\n" + line["explanation"] + f"\nSo the answer is {line['answer']}.\n\n")
            else:
                example += f'\n答案：{line["answer"]}\n\n' if language == "zh" else f'\nAnswer：{line["answer"]}\n\n'
        else:
            if cot:
                example += "\n答案：让我们一步一步思考，\n1." if language == "zh" else "\nAnswer：Let's think step by step,\n1."
            else:
                example += '\n答案：' if language == "zh" else '\nAnswer：'
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False, multiple=False, language="zh"):
        if multiple == False:
            prompt = self.one_ans_instruct_zh.format(subject) if language == "zh" else self.one_ans_instruct_en.format(subject)
            prompt += "\n\n"
        else:
            prompt = self.multi_ans_instruct_zh.format(subject) if language == "zh" else self.multi_ans_instruct_en.format(subject)
            prompt += "\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot,
                language=language
            )
        return prompt

    def generate_rag_few_shot_prompt(self, subject, row, multiple=False, language="zh"):
        if multiple == False:
            prompt = self.one_ans_instruct_zh.format(subject) if language == "zh" else self.one_ans_instruct_en.format(subject)
            prompt += "\n\n"
        else:
            prompt = self.multi_ans_instruct_zh.format(subject) if language == "zh" else self.multi_ans_instruct_en.format(subject)
            prompt += "\n\n"
        k = self.k
        for i in range(k):
            prompt += row[f"shot{i+1}"] + "\n"
        return prompt

    def extract_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'答案:([ABCD])',
            r'选择答案([ABCD])',
            r'选([ABCD])',
            r'选择选项([ABCD])',
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) >= 1:
            answer = m[0]
            return answer, False
        # only containing one choice-context
        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(line[f'{c}'])] = c
            pattern += re.escape(str(line[f'{c}']))+"|"
        pattern = pattern[:-1]
        m = re.findall(pattern, gen_ans, re.M)
        print("w/ escape:",repr(pattern),gen_ans,(len(m)>=1))
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False
        return  random.choice('ABCD'), False

    def extract_multiple_answer(self, gen_ans):
        letter_dict = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        patterns = {
            'comma_separated': r'([A-D](?:,[A-D])+)',   # 逗号分隔模式
            'dunhao_separated': r'([A-D](?:、[A-D])+)', # 顿号分隔模式
            'continuous': r'([A-D]{2,})'                # 连续字母模式
        }
        gen_ans = gen_ans.replace(" ", "")
        matches = {pat: re.search(pattern, gen_ans) for pat, pattern in patterns.items()}
        
        # 确定最先出现的匹配项，即多个字母靠近一起出现的组合
        first_match = None
        for _, match in matches.items():
            if match:
                if not first_match or match.start() < first_match.start():
                    first_match = match

        # 根据最先出现的匹配项更新字典
        if first_match:
            letters = first_match.group(0).replace(',', '').replace('、', '')
            for letter in letters:
                letter_dict[letter] = 1

        return letter_dict
