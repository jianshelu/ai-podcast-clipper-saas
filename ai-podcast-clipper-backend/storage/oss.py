from __future__ import annotations

import os
import oss2

from .provider import StorageProvider


class OssStorage(StorageProvider):
    def __init__(self) -> None:
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
        endpoint = os.getenv("OSS_ENDPOINT")
        bucket_name = os.getenv("OSS_BUCKET")
        if not access_key_id or not access_key_secret or not endpoint or not bucket_name:
            raise RuntimeError("Missing OSS configuration env vars")

        auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket = oss2.Bucket(auth, endpoint, bucket_name)

    def head(self, key: str) -> bool:
        try:
            self.bucket.head_object(key)
            return True
        except oss2.exceptions.OssError as exc:
            if exc.status == 404 or exc.code in {"NoSuchKey", "NotFound"}:
                return False
            print(
                f"[oss] head failed key={key} code={exc.code} request_id={exc.request_id} status={exc.status}"
            )
            raise

    def download(self, key: str, destination: str) -> None:
        try:
            self.bucket.get_object_to_file(key, destination)
        except oss2.exceptions.OssError as exc:
            print(
                f"[oss] download failed key={key} code={exc.code} request_id={exc.request_id} status={exc.status}"
            )
            raise

    def upload(self, source: str, key: str) -> None:
        try:
            self.bucket.put_object_from_file(key, source)
        except oss2.exceptions.OssError as exc:
            print(
                f"[oss] upload failed key={key} code={exc.code} request_id={exc.request_id} status={exc.status}"
            )
            raise
