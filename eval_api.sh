#!/bin/bash
# 测试本地LLM需要自己部署好本地 http://0.0.0.0:8000 的LLM API
set -e

PYTHON_SCRIPT="main.py"

export PROJ_HOME=$PWD

USE_API="True"                    # 执行eval_api.sh则默认为True，不需要修改
USE_OPENAI="False"                # 是否使用openai，调用本地LLM的API则为False
MODEL_NAME="chatglm3-6b"          # 模型名称，openai需指定gpt-3.5-turbo,gpt-4等，本地LLM不需要指定，但需要LLM名称用于日志文件命名
OPENAI_KEY="sk-**************"    # 填入自己的openai key，如果USE_OPENAI="False"并使用本地LLM会自动忽略

DATA_DIR="cpa_one"                # 指定数据集名称
COT="False"                       # 是否使用cot，大部分13B以下模型基本不具备cot能力，反而会对做题造成干扰，准确率严重下降，建议为False
MULTIPLE="False"                  # 当前测试集是否为多选题，例如使用cpa_multi测试集需要设置为True，其他情况默认为False
SHOTS=4                           # fewshot的示例个数，0表示不使用fewshot
TEMPERATURE=0.01                  # 大部分情况下都默认为0.01，模型会倾向于直接输出答案，如果使用cot则需要调高温度
DO_TEST="False"                   # FinKBenchmark公布了答案，默认do_test为False，可以直接选择验证集val评测模型准确率，如果使用人员接入其他测试集并且需要过一遍test，这时候才选择为True 
DYNAMIC_FS="False"                # FinKBenchmark提供了dynamic few-shot数据集示例，检索相似例题作为当前题目的fewshot，使用cpa_one_rag或cpa_multi_rag测试集就需要启用，常规测试集默认为False   
LANGUAGE="zh"                     # 根据中文或英文选择不同的prompt，例如CPA为zh，CFA为en     

exp_date=$(date +"%Y%m%d%H%M%S")
echo "exp_date": $exp_date
output_dir=$PROJ_HOME/output_dir/${MODEL_NAME}/$exp_date
echo "output_dir": $output_dir
datasets_dir=$PROJ_HOME/datasets/$DATA_DIR
echo "datasets_dir": $datasets_dir

# 添加日志目录和文件的创建
log_name="cot_${COT}_shots_${SHOTS}_rag_${RAG}"
log_dir=$PROJ_HOME/eval_logs/$DATA_DIR/$log_name
mkdir -p $log_dir
log_file_path=$log_dir/${MODEL_NAME}_${exp_date}.log

# 执行 Python 脚本并重定向输出到日志文件
python $PYTHON_SCRIPT \
    --model_path $MODEL_NAME \
    --data_dir $datasets_dir \
    --cot $COT \
    --shots $SHOTS \
    --temperature $TEMPERATURE \
    --output_dir $output_dir \
    --do_test $DO_TEST \
    --multiple $MULTIPLE \
    --dynamic_fs $DYNAMIC_FS \
    --language $LANGUAGE \
    --use_api $USE_API \
    --openai_key $OPENAI_KEY \
    --use_openai $USE_OPENAI \
    > $log_file_path 2>&1
