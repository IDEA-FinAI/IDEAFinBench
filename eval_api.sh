#!/bin/bash
#测试本地LLM需要自己部署好本地http://0.0.0.0:8000的LLM API
set -e

PYTHON_SCRIPT="main.py"

export PROJ_HOME=$PWD

USE_API="True"
MODEL_NAME="gpt-3.5-turbo"        # 模型名称，openai需指定gpt-3.5-turbo,gpt-4等，本地LLM不需要指定，但需要LLM名称用于日志文件命名
USE_OPENAI="True"                 # 是否使用openai，调用本地LLM的API则为False
OPENAI_KEY="sk-**************"    # 填入自己的openai key，本地LLM自动忽略

DATA_DIR="cpa_one"
COT="False"
SHOTS=4
TEMPERATURE=0.2
DO_SAVE_CSV="True"
DO_TEST="False"
MULTIPLE="False"
RAG="False"
LANGUAGE="zh"

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
    --do_save_csv $DO_SAVE_CSV \
    --output_dir $output_dir \
    --do_test $DO_TEST \
    --multiple $MULTIPLE \
    --rag $RAG \
    --language $LANGUAGE \
    --use_api $USE_API \
    --openai_key $OPENAI_KEY \
    --use_openai $USE_OPENAI \
    > $log_file_path 2>&1
