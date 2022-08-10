import torch.nn as nn
import torch 

class FrozenStableEmbedding(nn.Module):
    def __init__(self, weight, ln_weight, ln_bias):
        super().__init__()
        self.num_embeddings, self.embedding_dim = weight.shape
        self.register_buffer("weight", weight.requires_grad_(False))
        self.register_buffer("ln_weight", ln_weight.requires_grad_(False))
        self.register_buffer("ln_bias", ln_bias.requires_grad_(False))

    def forward(self, x, **kwargs):
        with torch.no_grad():
            # note: both quantuized weights and input indices are *not* differentiable

            emb = F.embedding(x, self.weight, **kwargs)
            return F.layer_norm(
                emb,
                normalized_shape=[self.embedding_dim],
                weight=self.ln_weight,
                bias=self.ln_bias,
            )

    @classmethod
    def from_embedding(
        cls, embedding: torch.nn.Module
    ) -> "FrozenStableEmbedding":
        weights = embedding.weight
        ln_weight, ln_bias = embedding.norm.weight, embedding.norm.bias
        return cls(weights, ln_weight, ln_bias)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.num_embeddings}, {self.embedding_dim})"