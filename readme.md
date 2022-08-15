# GPT*

GPT* is a collection of transformer models based on GPT2-Small, GPT2-Medium, and GPT2-XL with the following architecture modifications to speed up training and inference:

1. Parallel residual connections (as introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax))
2. Increasing the dimension of each attention head (as introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax)) 
3. [ALiBi](https://arxiv.org/abs/2108.12409) position embeddings (training at context length of 512 tokens)

### Models & Hyperparameters 🛠️:

| Hyperparameter      | Value (GPT-127*)           |Value (GPT-303*)           |Value (GPT-1B*)
|---------------------|----------------------------|----------------------------|----------------------------|
| n_parameters        | 127M                       |303M                       |1.009B                       |
| n_layers            | 6                          |8                          |18                           |
| d_model             | 1024                       |1536                       |2048                         |
| d_ff                | 4096                       |6144                       |8192                         |
| n_heads             | 8                          |8                          |8                            |
| d_head              | 128                        |192                        |256                          |
| n_vocab             | 50257 (GPT2 Tokenizer)     |50257 (GPT2 Tokenizer)     |50257 (GPT2 Tokenizer)       |
| Positional Encoding | ALiBi                      |ALiBi                      |ALiBi                        |
| n_ctx               | 512 Train / 1024 Inference |512 Train / 1024 Inference |512 Train / 1024 Inference|

### Dataset 📚: 
Both models were training for one epoch (roughly ~26B tokens) on a dataset consisting of: 
* [OpenWebText1](https://github.com/jcpeterson/openwebtext) and [OpenWebText2](https://arxiv.org/abs/2101.00027)
* [BookCorpus](https://arxiv.org/abs/2101.00027)
* [Bibiliotik/Books3](https://arxiv.org/abs/2101.00027)
* [PhilPapers](https://arxiv.org/abs/2101.00027)

### Benchmarks 🧪:
Benchmarks produced with [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). GPT-Neo and GPT-3 results are taken from [EleutherAI/gpt-neo](https://github.com/EleutherAI/gpt-neo#model-evaluations)

| Model        | LAMBADA (Acc) | LAMBADA (PPL) | WikiText (PPL) (2048 CTX) | Piqa (Acc) | Hellaswag (Acc) | Winogrande (Acc) | Training Tokens |
|--------------|---------------|---------------|----------------|------------|-----------------|------------------|-----------------|
| **GPT-127\***    | 36.23%        | 28.60         | 36.29          | 63.17%     | 28.07%          | 51.46%           | 26B            |
| GPT-Neo 125M | 37.36%        | 30.266        | 32.285         | 63.06%     | 28.67%          | 50.43%           | 300B            |
| GPT-3 125M   | 42.7%         | 18.6          | -              | 65.07%     | 33.7%           | 52.0%            | 300B            |
| **GPT-303\***    | 43.39%        | 15.89         | 28.57         | 65.2%     | 29.8%          | 49.3%           | 26B            |
| **GPT-354 (Mine)**    | 48.22%        | 13.29         | 28.177         | 65.67%     | 32.36%          | -           | 21B             |
| GPT-Neo 350M    | 47.27%        |  	13.876         | 22.5657         | 65.07%     | 32.16%         | 51.14%           | 300B            |
| GPT-3 350M    | 54.3%        |  	9.09         | -         | 70.2%     | 43.6%         | 52.1%           | 300B            |
| **GPT-1B\*** | 52.65         | 9.758         | 23.052 (1024 ctx) | 69.31%     | 33.36%          | 52.17%           | 26B             |
| GPT-2 1.5B   | 51.21%        | 10.634        | 17.48 (1024 ctx)  | 70.78%     | 40.03%          | 59.40%           | -               |
| GPT-Neo 1.3B | 57.23         | 7.498         | 13.10 (2048 ctx)  | 71.11%     | 38.66%         | 55.01%           | 300B            |

I have also included extra benchmarks increasing the ALiBi context length in [```traininglog.md```](traininglog.md).

### Training Log 📝:

For a full training log outlining my process and all of the training details see [```traininglog.md```](traininglog.md)

### Downloading Checkpoints 💾:
The following checkpoints are available for download:

- GPT-127* with optimizer states
- GPT-127* without optimizer states
- GPT-303* with optimizer states
- GPT-303* without optimizer states
- GPT-345M  without optimizer states (*unfortunately this was meant to be a throwaway model and as such, I deleted the optimizer states*)
- GPT-1B* with optimizer states
- GPT-1B* without optimizer states (can download 8bit quantized or full model)

To download a checkpoints, clone the repo and run: 
```
# options are "127", "303", "354", "1B", "1B8bit"
python download_checkpoints.py --model-size 127 --full-state
```

### Demo ⌨️:

Clone the repo: 
```
git clone https://github.com/fattorib/Faster-GPT.git
```

Install demo requirements:

```
pip install -r requirements-demo
```

Download a model checkpoint (this downloads the 354M param model):

```
python download_checkpoints.py --model-size 354
```

Launch gradio app locally (this runs the model we just downloaded). 
```
# options are  "base*", "medium*", "medium", "XL*",
python app.py --model-size 127
```

## References:

Some code in this repo has been modified from the following sources:
- minGPT: https://github.com/karpathy/minGPT
- GPT-NeoX: https://github.com/EleutherAI/gpt-neox/
- Attention with Linear Biases: https://github.com/ofirpress/attention_with_linear_biases
- Typical Sampling: https://github.com/cimeister/typical-sampling
- Nucleus Sampling: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317