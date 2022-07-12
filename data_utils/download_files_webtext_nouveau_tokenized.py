"""
Download tokenized files from S3 bucket
"""

import boto3
from keys import AWS_ACCESS_KEY, AWS_SECRET_ACCESS_KEY

if __name__ == "__main__":
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

    s3.download_file(
        "openwebtxtbf",
        f"plusplus/webtext_nouveau.tar.gz",
        f"data/raw/train/webtextnouveau_train.tar.gz",
    )
