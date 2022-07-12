# GPT*

GPT* is a pair of transformer models based on GPT2-Small and GPT2-Medium with the following architecture modifications to speed up training and inference:

1. Parallel residual connections (as introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax))
2. Increasing the dimension of each attention head (as introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax)) 
3. [ALiBi](https://arxiv.org/abs/2108.12409) position embeddings (training at context length of 512 tokens)

### Models & Hyperparameters 🛠️:

| Hyperparameter      | Value (GPT-127*)           |Value (GPT-303*)           |
|---------------------|----------------------------|----------------------------|
| n_parameters        | 127M                       |303M                       |
| n_layers            | 6                          |8                          |
| d_model             | 1024                       |1536                       |
| d_ff                | 4096                       |6144                       |
| n_heads             | 8                          |8                          |
| d_head              | 128                        |192                        |
| n_vocab             | 50257 (GPT2 Tokenizer)     |50257 (GPT2 Tokenizer)     |
| Positional Encoding | ALiBi                      |ALiBi                      |
| n_ctx               | 512 Train / 1024 Inference |512 Train / 1024 Inference |

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
| GPT-Neo 350M    | 47.27%        |  	13.876         | 22.5657         | 65.07%     | 32.16%         | 51.14%           | 300B            |
| GPT-3 350M    | 54.3%        |  	9.09         | -         | 70.2%     | 43.6%         | 52.1%           | 300B            |

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

To download a checkpoints, clone the repo and run: 
```
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

Download a model checkpoint (this downloads the 354M param model). *Options are 127,303,354*:

```
python download_checkpoints.py --model-size 354
```

Launch gradio app locally (this runs the model we just downloaded). *Options are base\*, medium\*, medium*:
```
python app.py --model-size medium
```

