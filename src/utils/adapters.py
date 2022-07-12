"""
Functions for adding adapter modules to a pretrained GPT2 model.
"""

import torch
import torch.nn as nn


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
    model: torch.nn.Module, reduction_factor: int
) -> torch.nn.Module:

    # Adds basic adapters to a given model

    for i in range(model.N):

        # Create adapter block(s)
        adapter_attn = BottleneckAdapter(model.embedding_dim, reduction_factor)
        adapter_ff = BottleneckAdapter(model.embedding_dim, reduction_factor)

        model.blocks[i].adapter_attn = adapter_attn
        model.blocks[i].adapter_ff = adapter_ff

        if model.fused_residuals == False:

            def forward_adapter(self, x):
                x = x + self.adapter_attn(self.attn(self.ln1(x)))
                x = x + self.adapter_ff(self.mlp(self.ln2(x)))
                return x

        else:

            def forward_adapter(self, x):
                x = (
                    x
                    + self.adapter_ff(self.mlp(self.ln1(x)))
                    + self.adapter_attn(self.attn(self.ln1(x)))
                )
                return x

        bound_method = forward_adapter.__get__(
            model.blocks[i], model.blocks[i].__class__
        )
        setattr(model.blocks[i], "forward", bound_method)

    # Need to modify the LM head too
    embedding_shape = model.wte.weight.shape
    # Keep weights tied to embeddings
    # lm_head_weight = copy.deepcopy(model.lm_head.weight)

    delattr(model, "lm_head")

    model.lm_head = nn.Linear(
        in_features=embedding_shape[1],
        out_features=embedding_shape[0],
        bias=False,
    )
    model.lm_head.weight = model.wte.weight

    return model


def prepare_adapter_training(model: torch.nn.Module) -> torch.nn.Module:

    # freezes all models params except for LN and adapter params. (Also adds new LN params?)

    for key, value in model.named_parameters():
        if "adapter" in key or "ln" in key or "wte" in key or "norm" in key:
            value.requires_grad = True
        else:
            value.requires_grad = False

    return model
