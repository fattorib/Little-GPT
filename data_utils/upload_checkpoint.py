from datetime import datetime

import boto3
from keys import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY


def upload_checkpoint(directory: str, prefix: str, path: str) -> None:
    """
    Uploads a model checkpoints to S3 bucket. Useful for saving, resuming training.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )
    now = datetime.today().strftime("%Y_%m_%d")
    with open(path, "rb") as f:
        s3.upload_fileobj(
            f,
            directory,
            f"model_checkpoints/model_checkpoint_{prefix}_{now}.tar.gz",
        )


if __name__ == "__main__":
    upload_checkpoint()
