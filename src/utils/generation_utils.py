"""
Helper utilities and classes for text generation from models. Supports:

1. Greedy decoding
2. Top-p sampling
3. Top-k sampling
4. Typical sampling

"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import GPT2Tokenizer
from tokenizers import Tokenizer
from typing import Tuple, List


def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def top_p_logits(
    logits: torch.Tensor,
    top_p: float = 0.0,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
            ..., :-1
        ].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[:, indices_to_remove] = filter_value
    return logits


def typical_sampling_logits(
    logits: torch.Tensor,
    mass: float = 0.2,
    min_tokens_to_keep: int = 1,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:

    # Entropy calculation

    normalized = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    shifted_scores = torch.abs((-normalized) - ent)
    sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
    sorted_logits = logits.gather(-1, sorted_indices)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    last_ind = (cumulative_probs < mass).sum(dim=1)
    last_ind[last_ind < 0] = 0
    sorted_indices_to_remove = sorted_scores > sorted_scores.gather(
        1, last_ind.view(-1, 1)
    )
    if min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
        sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove
    )

    logits = logits.masked_fill(indices_to_remove, filter_value)
    return logits


class TextGenerator:
    """
    This class stores all the text generation methods and functions.
    """

    def __init__(
        self,
        seq_len: int,
        tokenizer: Tokenizer,
    ) -> None:
        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.vocab_size = 32001
            self.seq_len = seq_len
            self.pad_token = tokenizer.token_to_id("<EOS>")

        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.vocab_size = 50257
            self.seq_len = seq_len
            self.pad_token = self.tokenizer.eos_token_id

    def generate_text_from_prompt(
        self,
        model: torch.nn.Module,
        prompt: str,
        steps: int,
        temperature: float,
        sample: bool = True,
        top_k: int = None,
        top_p: float = None,
        typical_sampling: bool = None,
        tau: float = None,
        repetition_penalty: float = 1.0,
        device: str = "cpu",
    ) -> Tuple[str, str, List[float]]:

        output, step, logprobs = self.generate_tokens(
            model=model,
            prompt=prompt,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            top_p=top_p,
            typical_sampling=typical_sampling,
            tau=tau,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        full_gen, new_gen = self.token_to_text(prompt, output, step)
        return full_gen, new_gen, logprobs

    def token_to_text(
        self, input: str, tok: torch.Tensor, step: int
    ) -> Tuple[str, str]:
        """
        Convert encoded tokens back to string and join with original prompt text
        """

        new_words = list(tok[0, -step:])

        generated_text = self.tokenizer.decode(new_words)
        return input + "".join(generated_text), generated_text

    @torch.no_grad()
    def generate_tokens(
        self,
        model: torch.nn.Module,
        prompt: str,
        steps: int,
        temperature: float,
        top_k: int,
        top_p: float = 0.0,
        typical_sampling: bool = False,
        tau: float = 0.2,
        repetition_penalty: float = 1.0,
        sample: bool = True,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, int, List[float]]:
        model.eval()
        logprobs = []

        tokens = torch.tensor(
            self.tokenizer.encode(prompt.strip()),
            dtype=torch.long,
            device=device,
        )

        x = tokens.view(1, -1)

        num_token = self.seq_len

        if x.shape[1] > num_token:

            x_cond = x[:, -num_token:]

        else:
            x_cond = x

        layer_past = None

        generated_tokens = []

        for step in tqdm(range(steps)):
            if device != 'cpu':
                # autocast hangs with CPU 
                with torch.autocast(device_type=device):
                    logits, layer_past = model(
                        x_cond, use_cache=True, past_states=layer_past
                    )
            else:
                logits, layer_past = model(
                        x_cond, use_cache=True, past_states=layer_past
                    )

            logits = logits[:, -1, :] / temperature

            for prev_gen_token in generated_tokens:
                if logits[:, prev_gen_token] < 0:
                    logits[:, prev_gen_token] *= repetition_penalty
                else:
                    logits[:, prev_gen_token] /= repetition_penalty

            if top_p > 0.0:
                logits = top_p_logits(logits, top_p=top_p)

            elif typical_sampling:
                logits = typical_sampling_logits(logits, mass=tau)
            else:
                logits = top_k_logits(logits, k=top_k)

            probs = F.softmax(logits, dim=-1)

            # This just sets the prob for <|endoftext|> to 0
            # probs[:, 50256] = 0

            if not sample:
                x_cond = torch.topk(probs, k=1).indices
                x = torch.cat((x[:, :], x_cond), axis=1)
                if x_cond.item() == self.pad_token:
                    return x[:, :], step
            else:
                x_cond = torch.multinomial(probs, num_samples=1)
                logprobs.append(torch.log(probs[:, x_cond]).item())
                # If we hit end of text, return as-is
                if x_cond.item() == self.pad_token:
                    return x[:, :], step, logprobs
                else:
                    x = torch.cat((x[:, :], x_cond), axis=1)

                if x_cond.item() not in generated_tokens:
                    generated_tokens.append(x_cond.item())

        return x, steps, logprobs
