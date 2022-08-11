""" 
bitsandbytes is not currently supported on Windows. GPT-1B* uses bnb.nn.StableEmbedding
for the embedding layers. We extract these values and put them into our own custom StableEmbedding layer

"""
import bitsandbytes as bnb
from src.models.stableembedding import FrozenStableEmbedding


def unbnbfy_(model):
    for module in list(model.modules()):
        for name, child in module.named_children():
            if isinstance(child, bnb.nn.StableEmbedding):
                print(name, child)
                setattr(
                    module, name, FrozenStableEmbedding.from_embedding(child)
                )
