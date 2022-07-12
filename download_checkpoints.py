"""
Download completed model states
"""

import requests
import argparse
from tqdm import tqdm


def parse():
    parser = argparse.ArgumentParser(description="Download trained models")

    parser.add_argument("--model-size", default="127", type=str)

    parser.add_argument("--full-state", default=False, action="store_true")

    args = parser.parse_args()
    return args


CHECKPOINT_MAP_COMPLETE = {
    "127": "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/127_complete_state.pth.tar",
    "303": "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/303_complete_state.pth.tar",
}


CHECKPOINT_MAP_INFERENCE = {
    "127": "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/127_model_only.pth.tar",
    "303": "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/303_model_only.pth.tar",
    "354": "https://bfattoripublic.s3.ca-central-1.amazonaws.com/models/GPT2_345.pth.tar",
}


def main():

    global args
    args = parse()

    assert args.model_size in ["127", "303", "354"]

    if args.full_state:
        download_path = CHECKPOINT_MAP_COMPLETE[args.model_size]
        save_path = f"checkpoints/{args.model_size}_complete_state.pth.tar"
    else:
        download_path = CHECKPOINT_MAP_INFERENCE[args.model_size]
        save_path = f"checkpoints/{args.model_size}_weights.pth.tar"

    r = requests.get(download_path, stream=True)

    with open(save_path, "wb") as f:
        chunk_size = 1000
        file_size = int(r.headers["content-length"])
        with tqdm(
            ncols=100, desc="Downloading... ", total=file_size, unit_scale=True
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)


if __name__ == "__main__":
    main()
