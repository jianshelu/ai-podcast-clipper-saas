from __future__ import annotations

import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .provider import StorageProvider


class S3Storage(StorageProvider):
    def __init__(self) -> None:
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        endpoint = os.getenv("OSS_ENDPOINT")
        bucket_name = os.getenv("OSS_BUCKET")
        if not access_key_id or not access_key_secret or not endpoint or not bucket_name:
            raise RuntimeError("Missing OSS configuration env vars")

        region = os.getenv("OSS_REGION", "us-east-1")
        signature_version = os.getenv("S3_SIGNATURE_VERSION", "s3v4")
        path_style = os.getenv("OSS_PATH_STYLE", "").lower() in ("1", "true", "yes")

        config = Config(
            signature_version=signature_version,
            s3={"addressing_style": "path" if path_style else "virtual"},
        )
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            region_name=region,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=access_key_secret,
            config=config,
        )
        self.bucket = bucket_name

    def head(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            print(f"[s3] head failed key={key} code={code} request_id={exc.response.get('ResponseMetadata', {}).get('RequestId')}")
            raise

    def download(self, key: str, destination: str) -> None:
        try:
            self.client.download_file(self.bucket, key, destination)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            print(f"[s3] download failed key={key} code={code} request_id={exc.response.get('ResponseMetadata', {}).get('RequestId')}")
            raise

    def upload(self, source: str, key: str) -> None:
        try:
            self.client.upload_file(source, self.bucket, key)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            print(f"[s3] upload failed key={key} code={code} request_id={exc.response.get('ResponseMetadata', {}).get('RequestId')}")
            raise
