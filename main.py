import os
import argparse
from evaluators.llm_evaluator import LLM_Evaluator
from evaluators.api_evaluator import API_Evaluator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
choices = ["A", "B", "C", "D"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default=None, type=str)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cot",choices=["False","True"], default="False")
    parser.add_argument("--shots", "-k", type=int, default=5)
    parser.add_argument("--constrained_decoding", choices=["False","True"], default="True")
    parser.add_argument("--temperature",type=float,default=0.2)
    parser.add_argument("--do_save_csv", choices=["False","True"], default="False")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--do_test", choices=["False","True"], default="False")
    parser.add_argument('--multiple', choices=["False","True"], default="False")
    parser.add_argument('--rag', choices=["False","True"], default="False")
    parser.add_argument("--language", default="zh", type=str)
    parser.add_argument("--use_api", choices=["False", "True"], default="False")
    parser.add_argument("--openai_key", default="sk-*****************", type=str)
    parser.add_argument("--use_openai", choices=["False", "True"], default="False")
    args = parser.parse_args()

    args.cot = args.cot == "True"
    args.constrained_decoding = args.constrained_decoding == "True"
    args.do_save_csv = args.do_save_csv == "True"
    args.do_test = args.do_test == "True"
    args.multiple = args.multiple == "True"
    args.rag = args.rag == "True"
    args.use_api = args.use_api == "True"
    args.use_openai = args.use_openai == "True"
    
    if args.use_api == False:
        evaluator=LLM_Evaluator(
            choices=choices,
            k=args.shots,
            model_type=args.model_type,
            model_path=args.model_path,
            temperature=args.temperature,
            constrained_decoding=args.constrained_decoding,
            cot=args.cot
        )
    else:
        evaluator=API_Evaluator(
            choices=choices,
            k=args.shots,
            cot=args.cot,
            use_openai=args.use_openai,
            api_key=args.openai_key,
            model_name=args.model_path
        )
    evaluator.main_process(args)
