# 参考了FinEval框架的实现：https://github.com/SUFE-AIFLM-Lab/FinEval
# Eval主入口，支持同时进行多个模型的评测

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PROJ_HOME = os.getcwd()
DATASET_HOME = os.path.join(PROJ_HOME, "datasets")

def evaluate_model(model_type, model_path, exp_name):
    exp_date = datetime.now().strftime("%Y%m%d%H%M%S")
    print(f"exp_date: {exp_date}")
    output_path = os.path.join(PROJ_HOME, "output_dir", exp_name, exp_date)
    print(f"output_path: {output_path}")

    dataset = "cpa_one"
    dataset_dir = os.path.join(DATASET_HOME, dataset)
    command_dict = {  
        "--model_type": model_type,        # 需要加载模型的类型，llama或auto
        "--model_path": model_path,        # 模型路径
        "--data_dir": dataset_dir,         # 测试集目录
        "--output_dir": output_path,       # 输出目录
        "--cot": "False",                  # 是否使用cot，大部分13B以下模型基本不具备cot能力，反而会对做题造成干扰，准确率严重下降，建议为False
        "--multiple": "False",             # 当前测试集是否为多选题，例如使用cpa_multi测试集
        "--shots": "0",                    # fewshot的shot数，0表示不使用fewshot
        "--constrained_decoding": "True",  # 受限解码仅支持单选题&&answer-only模式为True，其他情况必须设置为False
        "--temperature": "0.01",           # 大部分情况下都默认为0.01，模型会倾向于直接输出答案，如果使用cot则需要调高温度
        "--do_test": "False",              # FinExamEval公布了答案，默认do_test为False，选择验证集val计算准确率，如果接入其他测试集并且需要过一遍test，这时候才选择为True
        "--rag": "False",                  # FinExamEval提供了rag-fewshot数据集示例，检索相似例题作为当前题目的fewshot，使用cpa_one_rag或cpa_multi_rag测试集就需要启用，常规测试集默认为False
        "--language": "zh",                # 根据语言选择不同的prompt，例如CPA为zh，CFA为en
    }

    command = [
        "python", "main.py",
    ]
    for key, value in command_dict.items():
        command.extend([key, value])

    if not os.path.exists(f"eval_logs/{dataset}"):
        os.mkdir(f"eval_logs/{dataset}")
 
    log_name = f"cot_{command_dict['--cot']}_shots_{command_dict['--shots']}_rag_{command_dict['--rag']}"
    log_dir = os.path.join(PROJ_HOME, f"eval_logs/{dataset}", log_name)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file_path = log_dir + f"/{exp_name}_{exp_date}.log"
    
    # 注意，每次运行eval_parralel.py时，小概率某个model的log文件没有创建，需要手动核查下，如果没有就重新运行一次
    
    with open(log_file_path, 'w') as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"Command for model {model_path} exited with error code {result.returncode}")

if __name__ == "__main__":
    LLMS_PATH = "/data/FinAi_Mapping_Knowledge/LLMs" # LLMs本地模型路径，也可以在下面的model_path直接指定huggingface的模型路径
    # 不需要测试的模型直接注释掉即可
    model_data = [
        {
            "model_type": "auto",
            "model_path": LLMS_PATH + "/Yi-6B",
            "exp_name": "Yi-6B",
        },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-6B-instruct",
        #     "exp_name": "Yi-6B-instruct",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-34B",
        #     "exp_name": "Yi-34B",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-6B-Chat",
        #     "exp_name": "Yi-6B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-6B-Chat-sft-v0",
        #     "exp_name": "Yi-6B-Chat-sft-v0",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-6B-Chat-sft-v1",
        #     "exp_name": "Yi-6B-Chat-sft-v1",
        # # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-34B-Chat",
        #     "exp_name": "Yi-34B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-34B-Chat-sft-v1",
        #     "exp_name": "Yi-34B-Chat-sft-v1",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Yi-34B-Chat-sft-v2",
        #     "exp_name": "Yi-34B-Chat-sft-v2",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/chatglm3-6b",
        #     "exp_name": "chatglm3-6b",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/chatglm3-6b-base",
        #     "exp_name": "chatglm3-6b-base",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Baichuan2-7B-Base",
        #     "exp_name": "Baichuan2-7B-Base",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Baichuan2-13B-Base",
        #     "exp_name": "Baichuan2-13B-Base",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Baichuan2-7B-Chat",
        #     "exp_name": "Baichuan2-7B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Baichuan2-13B-Chat",
        #     "exp_name": "Baichuan2-13B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/DISC-FinLLM",
        #     "exp_name": "DISC-FinLLM",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Qwen-7B",
        #     "exp_name": "Qwen-7B",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Qwen-14B",
        #     "exp_name": "Qwen-14B",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Qwen-7B-Chat",
        #     "exp_name": "Qwen-7B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Qwen-14B-Chat",
        #     "exp_name": "Qwen-14B-Chat",
        # },
        # {
        #     "model_type": "llama",
        #     "model_path": LLMS_PATH + "/Llama-2-7b-chat-hf",
        #     "exp_name": "Llama-2-7b-chat-hf",
        # },
        # {
        #     "model_type": "llama",
        #     "model_path": LLMS_PATH + "/Llama-2-13b-chat-hf",
        #     "exp_name": "Llama-2-13b-chat-hf",
        # },
        # {
        #     "model_type": "llama",
        #     "model_path": LLMS_PATH + "/chinese-alpaca-2-7b",
        #     "exp_name": "chinese-alpaca-2-7b",
        # },
        # {
        #     "model_type": "llama",
        #     "model_path": LLMS_PATH + "/chinese-alpaca-2-13b",
        #     "exp_name": "chinese-alpaca-2-13b",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "Tongyi-Finance-14B-Chat",
        #     "exp_name": "Tongyi-Finance-14B-Chat",
        # },
        # {
        #     "model_type": "auto",
        #     "model_path": LLMS_PATH + "/Qwen-72B-Chat",
        #     "exp_name": "Qwen-72B-Chat",
        # },
    ]

    output_base_dir = os.path.join(PROJ_HOME, "output_dir")
    eval_logs_dir = os.path.join(PROJ_HOME, "eval_logs")
    for directory in [output_base_dir, eval_logs_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)

    with ThreadPoolExecutor(max_workers=len(model_data)) as executor:
        futures = [
            executor.submit(
                evaluate_model,
                model_type=model["model_type"],
                model_path=model["model_path"],
                exp_name=model["exp_name"]
            )
            for model in model_data
        ]

        for future in futures:
            future.result()
