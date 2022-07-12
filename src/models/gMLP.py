import torch
import torch.nn as nn
import copy
from einops.layers.torch import Rearrange
from einops import rearrange
import math
import torch.nn.functional as F
from scipy.linalg import toeplitz
import collections
from typing import Union, Tuple

"""
Module class for an autoregressive gMLP. Implementation is based on the paper
`Pay Attention to MLPs` <https://arxiv.org/abs/2105.08050> wherever possible

"""


def _weights_init(m):
    if isinstance(m, (nn.Linear)):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

        if isinstance(m, nn.Linear) and m.bias is None:
            m.weight.data.normal_(mean=0.0, std=0.02)

    if isinstance(m, (nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

    for name, p in m.named_parameters():
        if "fc_causal" in name and "weight" in name:
            with torch.no_grad():
                linear_weight_shape = p.data.shape[0]
                proxy_weight = torch.ones((linear_weight_shape, 1))
                proxy_weight.normal_(
                    mean=0,
                    std=1e-6,
                )
                p.data.copy_(torch.tril(torch.tensor(toeplitz(proxy_weight))))
        if "fc_causal" in name and "bias" in name:
            nn.init.ones_(p.data)


def _embedding_init(m):
    if isinstance(m, (nn.Embedding)):
        m.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(m, nn.Linear) and m.bias is None:
        m.weight.data.normal_(mean=0.0, std=0.02)


class SpatialGatingUnit(nn.Module):
    def __init__(self, n_ctx: int, embedding_dim: int) -> None:
        super().__init__()
        """
        Basic SGU. We don't use channel splitting. 
        """
        self.n_ctx = n_ctx
        self.embedding_dim = embedding_dim

        self.token_mix = nn.Sequential(
            collections.OrderedDict(
                [
                    ("ln", nn.LayerNorm(4 * self.embedding_dim)),
                    ("reshape_1", Rearrange("b n d -> b d n")),
                    ("fc_causal", nn.Linear(n_ctx, n_ctx)),
                    ("reshape_2", Rearrange("b d n -> b n d")),
                ]
            )
        )

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        v = self.token_mix(x)

        return v


class StaticSGU(nn.Module):
    def __init__(
        self, n_ctx: int, embedding_dim: int, kernel_size: int, decay_pow: int
    ) -> None:
        super().__init__()
        """
        Static learned token-mixing matrix
        """
        self.n_ctx = n_ctx
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        k = torch.ones(self.n_ctx, self.n_ctx).tril_().triu(-1 * kernel_size)
        k = k / (k.cumsum(-2) ** decay_pow + 1)
        self.register_buffer("mixing_kernel", k)
        self.ln = nn.LayerNorm(4 * self.embedding_dim)

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.ln(x)

        x = rearrange(x, "b n d -> b d n")

        x = x.matmul(self.mixing_kernel.T)

        x = rearrange(x, "b d n -> b n d")

        return x


class gMLPBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        embedding_dim: int,
        tiny_attn: bool,
        hidden_dropout: float,
        attn_dropout: float,
        static: bool,
        decay_pow=None,
        kernel_size=None,
    ) -> None:
        super().__init__()
        """
        Simple gMLP/aMLP block.
        """

        if static:
            self.sgu = StaticSGU(
                n_ctx=n_ctx,
                embedding_dim=embedding_dim,
                kernel_size=kernel_size,
                decay_pow=decay_pow,
            )
        else:
            self.sgu = SpatialGatingUnit(
                n_ctx=n_ctx, embedding_dim=embedding_dim
            )

        self.attn = None
        if tiny_attn:
            self.attn = TinySelfAttention(
                embedding_dim=embedding_dim,
                d_out=4 * embedding_dim,
                d_attn=64,
                block_size=n_ctx,
                attn_dropout=attn_dropout,
                num_layers=0,
            )

        self.proj_in = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.proj_out = nn.Linear(4 * embedding_dim, embedding_dim)

        self.act = nn.GELU()

        self.drop = nn.Dropout(hidden_dropout)

        self.ln = nn.LayerNorm(embedding_dim)

        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)

        if self.attn is not None:
            attn_out = self.attn(x)

            x = self.act(self.proj_in(x))

            x = self.drop(x)

            v = self.sgu(x)

            x = (v + attn_out) * x
        else:
            x = self.act(self.proj_in(x))

            x = self.drop(x)

            v = self.sgu(x)

            x = (v) * x

        return residual + self.proj_out(x)


class TinySelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.

    Minor modifications from `https://github.com/karpathy/minGPT/`
    """

    def __init__(
        self,
        embedding_dim: int,
        d_out: int,
        d_attn: int,
        block_size: int,
        attn_dropout: float,
        num_layers: int,
    ) -> None:
        super().__init__()
        # key, query, value projections for all heads
        self.key = nn.Linear(embedding_dim, d_attn)
        self.query = nn.Linear(embedding_dim, d_attn)
        self.value = nn.Linear(embedding_dim, d_attn)
        self.d_attn = d_attn
        self.attn_drop = nn.Dropout(attn_dropout)
        self.fc_resid = nn.Linear(d_attn, d_out)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(block_size, block_size, dtype=torch.uint8)
            ).view(1, 1, block_size, block_size),
        )
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        k = (
            self.key(x).view(B, T, 1, self.d_attn).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, 1, self.d_attn).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, 1, self.d_attn).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, self.d_attn)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.fc_resid(y)
        return y


class gMLP(nn.Module):
    def __init__(
        self,
        num_ctx: int,
        embedding_dim: int,
        N: int,
        vocab_size: int,
        tiny_attn: bool,
        tied_head=False,
        attn_dropout=0.0,
        hidden_dropout=0.0,
        embedding_dropout=0.0,
        static=False,
        kernel_size=None,
        decay_pow=None,
    ) -> None:
        super().__init__()
        self.num_ctx = num_ctx
        self.embedding_dim = embedding_dim
        self.N = N
        self.vocab_size = vocab_size
        self.tiny_attn = tiny_attn
        self.tied_head = tied_head

        """
        gMLP/aMLP model as described in `Pay Attention to MLPs`
        <https://arxiv.org/abs/2105.08050>.

        Modifications have been made to support causal language modelling
        """

        self.wte = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.gmlpblocks = nn.ModuleList(
            [
                copy.deepcopy(
                    gMLPBlock(
                        n_ctx=self.num_ctx,
                        embedding_dim=self.embedding_dim,
                        tiny_attn=tiny_attn,
                        attn_dropout=attn_dropout,
                        hidden_dropout=hidden_dropout,
                        static=static,
                        kernel_size=kernel_size,
                        decay_pow=decay_pow,
                    )
                )
                for _ in range(self.N)
            ]
        )

        self.ln = nn.LayerNorm(self.embedding_dim)
        self.dropout = nn.Dropout(p=embedding_dropout)
        self.lm_head = nn.Linear(
            self.embedding_dim, self.vocab_size, bias=False
        )

        if self.tied_head:
            self.lm_head.weight = self.wte.weight

        self.apply(_embedding_init)

    def prepare(self) -> None:
        """
        This function creates an upper triangular mask for the SGU and registers
        a hook that fires upon calling `.backward()`. This hook zeros out all
        gradients outside of the upper triangular mask (ensuring that at token
        index t, only tokens 1 to t are mixed).
        """
        mask = torch.tril(
            torch.ones_like(self.gmlpblocks[0].sgu.token_mix.fc_causal.weight)
        )

        for i in range(self.N):
            self.gmlpblocks[i].sgu.token_mix.fc_causal.weight.register_hook(
                self.get_zero_grad_hook(mask)
            )

    @staticmethod
    def get_zero_grad_hook(mask):
        def hook(grad):
            return grad * mask

        return hook

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        x = self.dropout(self.wte(x))

        for block in self.gmlpblocks:
            x = block(x)

        x = self.ln(x)

        logits_lm = self.lm_head(x)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits_lm[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            return logits_lm, loss
        else:
            return logits_lm


def create_gmlp_qa(vocab_size: int, num_ctx: int, **kwargs) -> gMLP:
    """
    QA model for testing
    """
    return gMLP(
        num_ctx=num_ctx, embedding_dim=128, N=4, vocab_size=vocab_size, **kwargs
    )


def create_gmlp_base(vocab_size: int, num_ctx: int, **kwargs) -> gMLP:
    """
    Matches the model size of the original GPT2-117M model
    """
    return gMLP(
        num_ctx=num_ctx,
        embedding_dim=768,
        N=12,
        vocab_size=vocab_size,
        tied_head=True ** kwargs,
    )


def create_gmlp_medium(vocab_size: int, num_ctx: int, **kwargs) -> gMLP:
    return gMLP(
        num_ctx=num_ctx,
        embedding_dim=1024,
        N=24,
        vocab_size=vocab_size,
        num_head=16,
        **kwargs
    )


def model_getter(
    model_name: str, vocab_size: int, num_ctx: int, **kwargs
) -> gMLP:
    MODELS_DICT = {
        "qa": create_gmlp_qa,
        "base": create_gmlp_base,
        "medium": create_gmlp_medium,
    }

    return MODELS_DICT[model_name](vocab_size, num_ctx, **kwargs)
