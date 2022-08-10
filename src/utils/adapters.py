"""
Functions for adding adapter modules to a pretrained GPT2 model.
"""

import torch
import torch.nn as nn
from typing import Tuple


def _weights_init(m):
    if isinstance(m, (nn.Linear)):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BottleneckAdapter(nn.Module):
    def __init__(self, embedding_dim: int, reduction_factor: int) -> None:
        super().__init__()

        bottleneck_dim = embedding_dim // reduction_factor
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=bottleneck_dim),
            nn.GELU(),
            nn.Linear(in_features=bottleneck_dim, out_features=embedding_dim),
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        out = self.bottleneck(x)

        return out + residual


def add_adapters(
    model: torch.nn.Module, reduction_factor: int, retrain_embeddings: bool = False
) -> torch.nn.Module:

    # Adds basic adapters to a given model

    for i in range(model.N):

        # Create adapter block(s)
        adapter_attn = BottleneckAdapter(model.embedding_dim, reduction_factor)
        adapter_ff = BottleneckAdapter(model.embedding_dim, reduction_factor)

        model.blocks[i].adapter_attn = adapter_attn
        model.blocks[i].adapter_ff = adapter_ff

        if model.fused_residuals == False:

            def forward_adapter(self, x: torch.Tensor,use_cache: bool = False,layer_past: Tuple[torch.Tensor, torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
                attn_out = self.attn(self.ln1(x), use_cache, layer_past)
                x = x + self.adapter_attn(attn_out[0])
                x = x + self.adapter_ff(self.mlp(self.ln2(x)))

                return x, attn_out[1]

        else:

            def forward_adapter(self, x: torch.Tensor,use_cache: bool = False,layer_past: Tuple[torch.Tensor, torch.Tensor] = None,) -> Tuple[torch.Tensor, torch.Tensor]:
                mlp_out = self.mlp(self.ln1(x))
                attn_out = self.attn(self.ln1(x), use_cache, layer_past)
                x = x + self.adapter_ff(mlp_out) + self.adapter_attn(attn_out[0])
                return x, attn_out[1]

        bound_method = forward_adapter.__get__(
            model.blocks[i], model.blocks[i].__class__
        )
        setattr(model.blocks[i], "forward", bound_method)

    if retrain_embeddings:
        # Need to modify the LM head too
        embedding_shape = model.wte.weight.shape

        delattr(model, "lm_head")

        model.lm_head = nn.Linear(
            in_features=embedding_shape[1],
            out_features=embedding_shape[0],
            bias=False,
        )
        model.lm_head.weight = model.wte.weight

    return model


def prepare_adapter_training(model: torch.nn.Module, retrain_embeddings: bool = False) -> torch.nn.Module:

    # freezes all models params except for LN and adapter params. (Also adds new LN params?)

    for key, value in model.named_parameters():
        if "adapter" in key or "ln" in key or "norm" in key:
            value.requires_grad = True
        elif "wte" in key and retrain_embeddings:
            value.requires_grad = True
        else:
            value.requires_grad = False

    return model