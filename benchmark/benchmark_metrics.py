import torch.nn as nn
import torch
import torch.cuda.amp as amp
from datasets import load_dataset
from tqdm import tqdm
import gzip
from benchmark.benchmark_map import DATASET_MAP
import math


def benchmark_ppl(
    model, args, tokenizer, device="cuda", max_length=1024, stride=1024
):
    """
    Compute perplexity per length (PPL) on a given dataset.
    Only supports wikitext2 from huggingface

    Based on: https://huggingface.co/docs/transformers/perplexity
    """
    dataset_name = DATASET_MAP[args.dataset].dataset_name

    assert dataset_name in ["wikitext-2-raw-v1"]

    test = load_dataset("wikitext", dataset_name, split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.shape[1] < max_length:
            pass

        else:
            with torch.no_grad():
                with amp.autocast():
                    if args.hf_model:
                        output_logits = model(input_ids).logits
                    else:
                        output_logits = model(input_ids)
                    shift_logits = output_logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    neg_log_likelihood = loss * trg_len

            nlls.append(neg_log_likelihood)
    return torch.exp(torch.stack(nlls).sum() / end_loc)


def benchmark_bpb(
    model, args, tokenizer, device="cuda", max_length=1024, stride=1024
):
    """
    Compute bits per UTF-8 encoded byte (BPB). Only supports enwiki8

    """

    dataset_pth = DATASET_MAP[args.dataset].dataset_name

    def utf8len(s):
        return len(s.encode("utf-8"))

    with gzip.open(dataset_pth) as file:
        X = file.read(int(100e6))
        _, validation_X = X[: int(90e6)].decode("utf-8"), X[int(95e6) :].decode(
            "utf-8"
        )

    byte_length = utf8len(validation_X)

    encodings = tokenizer(validation_X, return_tensors="pt")

    token_length = encodings.input_ids[0].shape[0]

    nlls = []
    cnt = 0
    total = 0
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.shape[1] < max_length:
            pass

        else:
            with torch.no_grad():
                with amp.autocast():
                    if args.hf_model:
                        output_logits = model(input_ids).logits
                    else:
                        output_logits = model(input_ids)
                    shift_logits = output_logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    neg_log_likelihood = loss

            nlls.append(neg_log_likelihood)

    return (torch.stack(nlls).mean() / math.log(2)) * (
        token_length / byte_length
    )


def benchmark_bpc(
    model, args, tokenizer, device="cuda", max_length=1024, stride=1024
):
    """
    Compute bits per character (BPC). Only supports text8

    """

    dataset_pth = DATASET_MAP[args.dataset].dataset_name

    def utf8len(s):
        return len(s.encode("utf-8"))

    with gzip.open(dataset_pth) as file:
        X = file.read(int(100e6))
        _, validation_X = X[: int(90e6)].decode("utf-8"), X[int(95e6) :].decode(
            "utf-8"
        )

    byte_length = utf8len(validation_X)

    encodings = tokenizer(validation_X, return_tensors="pt")

    token_length = encodings.input_ids[0].shape[0]

    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        if input_ids.shape[1] < max_length:
            pass

        else:
            with torch.no_grad():
                with amp.autocast():
                    if args.hf_model:
                        output_logits = model(input_ids).logits
                    else:
                        output_logits = model(input_ids)
                    shift_logits = output_logits[..., :-1, :].contiguous()
                    shift_labels = target_ids[..., 1:].contiguous()

                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                    )

                    neg_log_likelihood = loss

            nlls.append(neg_log_likelihood)

    return (torch.stack(nlls).mean() / math.log(2)) * (
        token_length / byte_length
    )


METRIC_REGISTRY = {
    "PPL": benchmark_ppl,
    "BPB": benchmark_bpb,
    "BPC": benchmark_bpc,
}
