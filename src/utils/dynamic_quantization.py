""" 
Code adapted from: https://huggingface.co/hivemind/gpt-j-6B-8bit

Small changes made to ensure compatability with StableEmbedding layer from bitsandbytes

Example Use:

NOTE: This code was created around a week prior to the release of: https://github.com/TimDettmers/bitsandbytes

...
model = model_getter()

#updates model inplace
bnbfy_(model)

"""

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_blockwise, quantize_blockwise
from torch.cuda.amp import custom_bwd, custom_fwd


class DequantizeAndLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        weights_quantized: torch.ByteTensor,
        absmax: torch.FloatTensor,
        code: torch.FloatTensor,
        bias: torch.FloatTensor,
    ):
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        ctx.save_for_backward(input, weights_quantized, absmax, code)
        ctx._has_bias = bias is not None
        return F.linear(input, weights_deq, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        assert (
            not ctx.needs_input_grad[1]
            and not ctx.needs_input_grad[2]
            and not ctx.needs_input_grad[3]
        )
        input, weights_quantized, absmax, code = ctx.saved_tensors
        # grad_output: [*batch, out_features]
        weights_deq = dequantize_blockwise(weights_quantized, absmax=absmax, code=code)
        grad_input = grad_output @ weights_deq
        grad_bias = grad_output.flatten(0, -2).sum(dim=0) if ctx._has_bias else None
        return grad_input, None, None, None, grad_bias


class FrozenBNBLinear(nn.Module):
    def __init__(self, weight, absmax, code, bias=None):
        assert isinstance(bias, nn.Parameter) or bias is None
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.bias = bias

    def forward(self, input):
        return DequantizeAndLinear.apply(
            input, self.weight, self.absmax, self.code, self.bias
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "FrozenBNBLinear":
        weights_int8, state = quantize_blockise_lowmemory(linear.weight)
        return cls(weights_int8, *state, linear.bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features}, {self.out_features})"


class FrozenBNBStableEmbedding(nn.Module):
    def __init__(self, weight, absmax, code, ln_weight, ln_bias):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("absmax", absmax.requires_grad_(False))
        self.register_buffer("code", code.requires_grad_(False))
        self.register_buffer("ln_weight", ln_weight.requires_grad_(False))
        self.register_buffer("ln_bias", ln_bias.requires_grad_(False))

    def forward(self, x, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable
            weight_deq = dequantize_blockwise(
                self.weight, absmax=self.absmax, code=self.code
            )

            emb = F.embedding(x, weight_deq, **kwargs)
            return F.layer_norm(
                emb,
                normalized_shape=[self.embedding_dim],
                weight=self.ln_weight,
                bias=self.ln_bias,
            )

    @classmethod
    def from_embedding(
        cls, embedding: bnb.nn.StableEmbedding
    ) -> "FrozenBNBStableEmbedding":
        weights_int8, state = quantize_blockise_lowmemory(embedding.weight)
        ln_weight, ln_bias = embedding.norm.weight, embedding.norm.bias
        return cls(weights_int8, *state, ln_weight, ln_bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"


def quantize_blockise_lowmemory(matrix: torch.Tensor, chunk_size: int = 2 ** 20):
    assert chunk_size % 4096 == 0
    code = None
    chunks = []
    absmaxes = []
    flat_tensor = matrix.view(-1)
    for i in range((matrix.numel() - 1) // chunk_size + 1):
        input_chunk = flat_tensor[i * chunk_size : (i + 1) * chunk_size].clone()
        quantized_chunk, (absmax_chunk, code) = quantize_blockwise(
            input_chunk, code=code
        )
        chunks.append(quantized_chunk)
        absmaxes.append(absmax_chunk)

    matrix_i8 = torch.cat(chunks).reshape_as(matrix)
    absmax = torch.cat(absmaxes)
    return matrix_i8, (absmax, code)


def bnbfy_(model):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                print(name, child)
                setattr(module, name, FrozenBNBLinear.from_linear(child))

            elif isinstance(child, bnb.nn.StableEmbedding):
                print(name, child)
                setattr(module, name, FrozenBNBStableEmbedding.from_embedding(child))
