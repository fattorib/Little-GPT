from typing import NamedTuple


class Dataset(NamedTuple):
    """
    Holds Huggingface dataset name (where applicable)
    or path to dataset and desired metric (from PPL, BPB, BPC)
    """

    dataset_name: str
    metric: str


DATASET_MAP = {
    "WikiText2": Dataset("wikitext-2-raw-v1", "PPL"),
    "enwik8": Dataset("benchmark/data/enwik8.gz", "BPB"),
    "text8": Dataset("benchmark/data/text8.gz", "BPC"),
}
