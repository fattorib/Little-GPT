# GPT* Training Log

## Background

This projects explores applying some ideas for pretraining [large](https://github.com/kingoflolz/mesh-transformer-jax/) [transformer](https://arxiv.org/abs/2204.06745) [language](https://arxiv.org/abs/2203.15556) [models](https://arxiv.org/abs/2204.02311) to models of a smaller scale. Here, I trained GPT-127* and GPT-303*, a pair of 127M and 303M parameter decoder-only transformers. The models here are based off of the [GPT2](https://openai.com/blog/better-language-models/)/[GPT3](https://arxiv.org/abs/2005.14165) family of models.

## Model

Both models are decoder-only transformer models with model counts similar to GPT-Small and GPT-Medium. The models deviate from their respective models GPT2/3 in the following ways:

1. **Parallel Residual Connections**
    This architecture change was first introduced in [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax/) and replaces the forward pass: 
    ```python
    x = x + attn(ln1(x))
    x = x + mlp(ln2(x))
    ```
    with: 
    ```python
    x = x + mlp(ln1(x)) + attn(ln1(x))
    ```
    Early tests I performed showed around a 5-10% speedup during training. Larger scale models such as [PaLM](https://arxiv.org/abs/2204.02311) have used this technique and show little-to-no quality loss. Experiments I performed showed the same results.
2. **Increased Attention Dimension**
    The GPT-J repo also notes that reducing the total number of attention heads also provides a training and inference speedup with minimal quality loss. Their model considers a much larger 6B param transformer with an attention dimension of 256 (embedding dimension of 16384). I opted to increase the embedding dimension from 768 to 1024 and decrease the total number of heads from 12 to 8. To accomodate the dimensionality increase, the total number of layers is halved from 12 to 6. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361?context=cs.LG) shows very little performance differences between model depth, when parameters are kept constant. Experiments I performed showed between a 10-15% training speedup.
3. **ALiBi Position Embeddings**
    The final, and most significant model speedup comes from replacing the standard learned position embeddings with [ALiBi](https://arxiv.org/abs/2108.12409) position embeddings. For this project, I targeted an inference sequence length of 1024. While I experimented early on with a train sequence length of 256, the extrapolation performance to 1024 tokens was quite poor so I settled for a train sequence length of 512 tokens. *Unless stated otherwise, all benchmarks are performed at 1024 sequence length.*

## Dataset

While GPT2 is trained with dropout for multiple epochs on Webtext, Following [One Epoch Is All You Need](https://arxiv.org/abs/1906.06669), many newer models are trained for a single epoch, without dropout on a much larger corpus of text (for example, 300B total training tokens for GPT3). I followed this same methodology and expanded upon the [OpenWebText](https://github.com/jcpeterson/openwebtext) dataset by including 3 other datasets from [The Pile](https://arxiv.org/abs/2101.00027). The completed dataset, I have named WebText++, consists of the following datasets:

| **Dataset**    | **Relative Epochs Elapsed During Training** |
|----------------|---------------------------------------------|
| OpenWebText    | 1.0                                         |
| BookCorpus2    | 1.0                                         |
| Books3         | 0.5                                         |
| PhilPapers     | 1.0                                         |
| OpenWebText2   | 1.0                                         |

BookCorpus2, Books3, PhilPapers, and OpenWebText2 are all components of The Pile. For OpenWebText2, I chose to only include documents after the creation of OpenWebText to avoid overlap. These datasets were selected as they are all sources of general purpose and high quality english writing. 

*No form of deduplication was performed on the data and every dataset was sampled at most once.*

*Early on in my experimentation with creating datasets, I only intended to use part of Books3 and deleted half of the total files. If I had to do this again, I would have kept the complete corpus for training*

After cleaning all text with [ftfy](https://ftfy.readthedocs.io), the documents were tokenized with GPT2's tokenizer and concatenated together into a single stream of tokens separated with an ```<|endoftext|>``` token. After tokenization, the total dataset consists of around 26B BPE tokens. 

*Due to an error in my preprocessing code, each individual document had the ```<|endoftext|>``` token appended to both the beginning and end of the text. While this didn't impact performance at all, it is worth pointing out.*

## Training Details

Wherever possible, I tried follow training heuristics from other similar-sized models. The only signifcant deviation comes from the batch size. Most papers I read, trained models under 1B parameters with a batch size of around 0.5M tokens. For GPT2 and GPT3, these would correspond to batch sizes of 512 and 256 respectively. 

*Given that I was training with a sequence length of 512, this corresponds to a total batch size of 1024 samples.*

The full list of training hyperparameters is as follows:
| Hyperparameter     | Value |
|--------------------|-------|
| Batch Size         | 1024  |
| Sequence Length    | 512   |
| Warmup Steps       | 4000  |
| Total Steps        | 51000 |
| Weight Decay       | 0.1   |
| Min Learning Rate  | 6e-5  |
| Min Learning Rate  | 6e-4  |
| Initial LR         | 0     |

The learning rate schedule follow [Cosine Annealing](https://arxiv.org/abs/1608.03983v5) with a warmup of 4000 steps and a decay to 10% of the original learning rate which was held constant for the final 10% of the training tokens. 

GPT-127* training was conducted on a VM with 4 RTX A5000s for a total of 248 GPU Hours with [HuggingFace Accelerate](https://github.com/huggingface/accelerate) for the distributed training. GPT-303* training was conducted on a VM with 4 RTX 3090s using the same codebase for a total of 412 GPU HOURS. Overall, this made distributed training very easy with the caveat that DeepSpeed training was not possible. While the HuggingFace team is working to add in proper support for DeepSpeed, at the time of training, I ran into too many issues, especially with saving and resuming model and optimizer checkpoints, to feel comfortable starting a long (and expensive) training job just for it to error out with no way to resume my progress.  

# Reference Models

Early on in my experimenation, I pre-trained a GPT2-Small and GPT2-Medium model from scratch on a slightly smaller dataset than WebText++ (OpenWebText2 and PhilPapers were not included). This dataset was around 25% smaller, consisting of 21B total training tokens. The model and training setup followed the method described in GPT2's paper. See next section below for more details of my GPT2-Medium training.

**Training Time**: Despite training on a dataset 25% smaller, my GPT2-Small required a total of 240 GPU Hours. This gives us a rough estimate of our training speedup from GPT-127* as 25% overall! 

**Benchmarks**:

Benchmarks were performed using [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) 

| Model        | LAMBADA (Acc) | LAMBADA (PPL) | WikiText (PPL) | Piqa (Acc) | Hellaswag (Acc) | Winogrande (Acc) | Training Tokens |
|--------------|---------------|---------------|----------------|------------|-----------------|------------------|-----------------|
| **GPT-127\***    | 36.23%        | 28.60         | 37.99          | 63.17%     | 28.07%          | 51.46%           | 26B            |
| **GPT-125M (Mine)**    | 36.52%        | 30.0494         | 37.863         | 62.46%     | 28.57%          | -           | 21B            |
| GPT-Neo 125M | 37.36%        | 30.266        | 32.285         | 63.06%     | 28.67%          | 50.43%           | 300B            |
| GPT-3 125M   | 42.7%         | 18.6          | -              | 65.07%     | 33.7%           | 52.0%            | 300B            |


## GPT-303* Training Log

Early on in my experimentation, on the same 21B token corpus, I also trained a 354M param model following GPT2-Medium. The benchmark results for my GPT2-354M param are given below with comparisons to GPT3 and GPT-Neo models of the same size:

| Model        | LAMBADA (Acc) | LAMBADA (PPL) | WikiText (PPL) | Piqa (Acc) | Hellaswag (Acc) | Winogrande (Acc) | Training Tokens |
|--------------|---------------|---------------|----------------|------------|-----------------|------------------|-----------------|
| **GPT-303\***   | 43.39%        | 15.89         | 28.57         | 65.2%     | 29.8%          | 49.3%           | 26B            |
| **GPT-354 (Mine)**    | 48.22%        | 13.29         | 28.177         | 65.67%     | 32.36%          | 51.3%           | 21B             |
| GPT-Neo 350M    | 47.27%        |  	13.876         | 22.5657         | 65.07%     | 32.16%         | 51.14%           | ~300B   |
| GPT-3 350M    | 54.3%        |  	9.09         | 22.5657         | 70.2%     | 43.6%         | 52.1%           | 300B             |

My 354M param model I trained was able to generate relatively coherent sounding text. I was pleasantly surprised by how well this model performed and was able to find the time & compute to train GPT-303* on WebText++. 

## GPT-1B* Training Log

(WIP) GPT-1B* is trained with the same setup with the following training changes:

1. 8-bit optimizers from [bitsandbytes](https://github.com/facebookresearch/bitsandbytes)
2. Staged Sequence Length Warmup from 128 to 512. Over the first 30% steps of training (15000 steps) the train sequence length was warmed up from 128 to 512 in stages of 3750 steps. The sequence length progression was (128,192,256,384,512).

| Model        | LAMBADA (Acc) | LAMBADA (PPL) |   WikiText (PPL)  | Piqa (Acc) | Hellaswag (Acc) | Winogrande (Acc) | Training Tokens |
|--------------|:-------------:|:-------------:|:-----------------:|:----------:|:---------------:|:----------------:|:---------------:|
| **GPT-1B\*** | 52.65         | 9.758         | 23.052 (1024 ctx) | 69.31%     | 33.36%          | 52.17%           | 26B             |
| GPT-2 1.5B   | 51.21%        | 10.634        | 17.48 (1024 ctx)  | 70.78%     | 40.03%          | 59.40%           | -               |
| GPT-Neo 1.3B | 57.23         | 7.498         | 13.10 (2048 ctx)  | 71.11%     | 38.66%         | 55.01%           | 300B            |

(more to add here)

## More Benchmarks 

Extra benchmarks varying ALiBi context length:

**GPT-127\*:**
| Model               | enwik8 (BPB) | text8 (BPC) | WikiText (PPL) |
|---------------------|--------------|-------------|----------------|
| GPT-127* (1024 ctx) | 1.341        | 1.342       | 37.99          |
| GPT-127* (2048 ctx) | 1.327        | 1.334       | 36.29          |
| GPT-127* (3072 ctx) | 1.323        | 1.333       | 35.30          |
| GPT-127* (4096 ctx) | 1.321        | 1.332       | 35.59          |
| GPT-127* (8192 ctx) | 1.317        | 1.331      | 35.29          |

**GPT-303\*:**
| Model               | enwik8 (BPB) | text8 (BPC) | WikiText (PPL) |
|---------------------|--------------|-------------|----------------|
| GPT-303* (1024 ctx) | 1.251        | 1.270       | 29.91          |
| GPT-303* (2048 ctx) | 1.239        | 1.266       | 28.57          |
| GPT-303* (3072 ctx) | 1.237        | 1.267       | 27.87          |
| GPT-303* (4096 ctx) | 1.236        | 1.268       | 28.15          |
| GPT-303* (8192 ctx) | OOM          | OOM         | OOM            |

Can be run with the command: 
```
bash benchmark.sh
```

## Summary 

This project was my first introduction to training language models. Throught it, some of the main thing I learned were:

- How to train large language models and what the decoder-only architecture actually is
- How to easily setup distributed training in PyTorch 
- How input documents are cleaned, tokenized and prepared for training
- How to handle creating and streaming large datasets that cannot fit in memory 
- Different decoding strategies for generating text (Greedy, Top-K, Nucleus, etc)
- About the different common language modelling metrics (PPL, BPB, etc) and datasets used to benchmark model performance
- More about attention!
- The benefits of one-epoch training and the performance gains from expanding the training dataset

I have worked on this project for the last 7 months or so and at this stage I am comfortable leaving it for now to work on other (read: less expensive) projects. If I were to continue this project, in no particular order I would want to:

- Train the models for longer. Both GPT-127* and GPT-303* are undertrained. Both validation loss curves constantly decreased during training without ever actually increasing! I suspect I could squeeze out some more performance by just letting the models train for longer, and letting them pass into a second epoch of training. The GPT-NeoX team did this during their training of their 20B param model and note that the validation loss continued to decrease. 

- Create a much larger dataset (~50B to 100B tokens). While The Pile would be my starting point, I would also be interested in scraping my own smaller dataset to include. 


Thanks for reading!