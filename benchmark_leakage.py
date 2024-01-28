import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def cal_leakage_eval_dir(dir_path, model, tokenizer):
    total_folder_score = 0
    csv_nums = 0
    for filename in tqdm(os.listdir(dir_path), desc=f"Processing files in {dir_path}"):
        if filename.endswith(".csv"):
            csv_nums += 1
            file_path = os.path.join(dir_path, filename)
            df = pd.read_csv(file_path)
            file_score = 0

            for _, row in df.iterrows():
                tuple_list = [
                    [row['question'] + "\nA. " + row['A'][:len(row['A']) // 2], row['A'][len(row['A']) // 2:]],
                    [row['question'] + "\nA. " + row['A'] + "\nB. " + row['B'][:len(row['B']) // 2], row['B'][len(row['B']) // 2:]],
                    [row['question'] + "\nA. " + row['A'] + "\nB. " + row['B'] + "\nC. " + row['C'][:len(row['C']) // 2], row['C'][len(row['C']) // 2:]],
                    [row['question'] + "\nA. " + row['A'] + "\nB. " + row['B'] + "\nC. " + row['C'] + "\nD. " + row['D'][:len(row['D']) // 2], row['D'][len(row['D']) // 2:]],
                ]

                for text, last_option_half in tuple_list:
                    print("用于检测数据泄露的前缀：")
                    print(text)
                    print("GroundTruth：")
                    print(last_option_half)
                    inputs = tokenizer(text, return_tensors='pt')
                    inputs = inputs.to('cuda:0')
                    pred = model.generate(
                        **inputs, 
                        do_sample=False, 
                        max_new_tokens=1, 
                        repetition_penalty=1.05,
                        output_scores=True, 
                        return_dict_in_generate=True
                    )
                    predicted_token = tokenizer.decode(pred.sequences[0][-1]).strip()
                    print("模型预测的下一个token：")
                    print(predicted_token)
                    
                    if last_option_half.startswith(predicted_token):
                        file_score += 1

            total_folder_score += file_score / len(df) if df.shape[0] > 0 else 0

    leakage_score = total_folder_score / csv_nums if csv_nums > 0 else 0
    return leakage_score

if __name__ == "__main__":
    LLMS_PATH = "/data/FinAi_Mapping_Knowledge/LLMs/"
    tokenizer = AutoTokenizer.from_pretrained(LLMS_PATH + "Baichuan2-7B-Base", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(LLMS_PATH + "Baichuan2-7B-Base", device_map="auto", torch_dtype='auto', trust_remote_code=True)

    dir_paths = ["datasets/cpa_one_old/val", "datasets/cpa_one/val"]
    # 使用字典存储每个测试集的数据泄露分数
    leakage_score_dict = {}
    for dir_path in dir_paths:
        leakage_score_dict[dir_path] = cal_leakage_eval_dir(dir_path, model, tokenizer)
    for dir_path, score in leakage_score_dict.items():
        print(f"测试集{dir_path}的数据泄露分数为：{score}")