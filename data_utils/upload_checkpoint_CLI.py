"""
This is called through a subprocess and performs a background compression and upload of files.
Ensures model checkpointing is not a bottleneck on slower connections
"""

from data_utils.upload_checkpoint import upload_checkpoint
import argparse
import os


def parse():
    parser = argparse.ArgumentParser(description="Background Checkpoint")
    parser.add_argument("--dir", type=str)
    parser.add_argument("--prefix", type=str)
    parser.add_argument("--path", type=str)

    args = parser.parse_args()
    return args


def main():
    args = parse()

    os.system("tar -cf checkpoint.tar checkpoints")
    os.system("gzip -f checkpoint.tar")

    upload_checkpoint(directory=args.dir, prefix=args.prefix, path=args.path)


if __name__ == "__main__":
    main()
