"""
Just a benchmark script for evaluating Bits-Per-Byte,Bits-Per-Character, and 
Perplexity-Per-Length for enwik8,text8, and WikiText2 respectively. All other
benchmarks reported are computed using:
    https://github.com/EleutherAI/lm-evaluation-harness/
"""

from transformers import GPT2TokenizerFast
import argparse
import torch
from benchmark.benchmark_map import DATASET_MAP
from benchmark.benchmark_metrics import METRIC_REGISTRY
import pandas as pd


def parse():
    parser = argparse.ArgumentParser(description="LM Benchmarks")

    parser.add_argument("--dataset", type=str)

    parser.add_argument("--model", type=str)

    parser.add_argument("--checkpoint", type=str)

    parser.add_argument("--type", type=str)

    parser.add_argument("--eval-ctx", type=str)

    parser.add_argument("--hf-model", default=False, action="store_true")

    args = parser.parse_args()
    return args


def check_args(args):
    assert args.dataset in DATASET_MAP.keys()
    assert args.model in ["base", "medium*", "base*"]
    assert args.type in ["GPT2"]


def main():

    args = parse()
    check_args(args)

    if not args.hf_model:

        from src.models.GPT2 import model_getter as model_getter_GPT2

        state_dict = torch.load(args.checkpoint, map_location="cpu")

        # ALiBi model, trained on 512 tokens
        model = model_getter_GPT2(
            args.model,
            vocab_size=50257,
            num_ctx=512,
            **{"fused_residuals": True, "num_head": 8, "use_alibi": True},
        )
        model.load_state_dict(state_dict)

    else:
        from transformers import AutoModelForCausalLM

        model_name = "gpt2" if args.model == "base" else "gpt2-medium"
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.cuda()
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    dataset_tuple = DATASET_MAP[args.dataset]

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

    task_df = pd.DataFrame(
        {
            "task": [dataset_tuple.dataset_name] * len(context_list),
            "metric": [dataset_tuple.metric] * len(context_list),
            "value": eval_value,
            "eval context length": ctx,
        }
    )

    print(task_df)


if __name__ == "__main__":

    main()
