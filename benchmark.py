"""
Just a benchmark script for evaluating Bits-Per-Byte,Bits-Per-Character, and 
Perplexity-Per-Length for enwik8,text8, and WikiText2 respectively. Handles all 
my models + HF models. 

All other benchmarks reported are computed using:
    https://github.com/EleutherAI/lm-evaluation-harness/
"""

from transformers import GPT2TokenizerFast
import argparse
from benchmark.benchmark_map import DATASET_MAP
from benchmark.benchmark_metrics import METRIC_REGISTRY
import pandas as pd


def parse():
    parser = argparse.ArgumentParser(description="LM Benchmarks")

    parser.add_argument("--dataset", type=str)

    parser.add_argument("--model", type=str)

    parser.add_argument("--type", type=str)

    parser.add_argument("--eval-ctx", type=str)

    parser.add_argument("--hf-model", default=False, action="store_true")

    parser.add_argument("--bit-quantize", default=False, action="store_true")

    args = parser.parse_args()
    return args


def check_args(args):
    assert args.model in ["base", "medium*", "base*", "XL*", "medium"]
    assert args.type in ["GPT2"]


def main():

    args = parse()
    check_args(args)

    if not args.hf_model:

        from src.models.GPT2 import model_getter as model_getter

        if "*" in args.model:

            save_paths = {
                "base*": "checkpoints/127_weights.pth.tar",
                "medium*": "checkpoints/303_weights.pth.tar",
                "XL*": "checkpoints/1B_weights_8bit.pth.tar",
            }

            model = model_getter(
                args.model,
                vocab_size=50257,
                num_ctx=512,
                model_checkpoint=save_paths[args.model],
                **{
                    "fused_residuals": True,
                    "num_head": 8,
                    "use_alibi": True,
                    "quantized_state": True if "XL" in args.model else False,
                },
            )

        elif args.model == "medium":
            model = model_getter(
                "medium",
                vocab_size=50257,
                num_ctx=1024,
                model_checkpoint="checkpoints/354_weights.pth.tar",
                **{"fused_residuals": False, "use_alibi": False},
            )

    else:
        from transformers import AutoModelForCausalLM

        model_name = "gpt2" if args.model == "base" else "gpt2-medium"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.cuda()
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset_list = [item for item in args.dataset.split(",")]

    task_df = pd.DataFrame()

    for dataset in dataset_list:

        args.dataset = dataset
        dataset_tuple = DATASET_MAP[dataset]

        benchmark_function = METRIC_REGISTRY[dataset_tuple.metric]

        context_list = [int(item) for item in args.eval_ctx.split(",")]

        eval_value = []
        ctx = []
        for eval_ctx in context_list:
            stride, max_length = eval_ctx, eval_ctx
            assert dataset_tuple.metric in ["PPL", "BPB", "BPC"]
            metric = benchmark_function(
                model, args, tokenizer, stride=stride, max_length=max_length
            )
            eval_value.append(metric.cpu().numpy())
            ctx.append(eval_ctx)

        single_task_df = pd.DataFrame(
            {
                "task": [dataset_tuple.dataset_name] * len(context_list),
                "metric": [dataset_tuple.metric] * len(context_list),
                "value": eval_value,
                "eval context length": ctx,
            }
        )

        task_df = task_df.append(single_task_df)

    print(task_df)


if __name__ == "__main__":

    main()
