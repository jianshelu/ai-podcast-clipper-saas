from __future__ import annotations

import os

from .oss import OssStorage
from .s3 import S3Storage
from .provider import StorageProvider


def get_storage() -> StorageProvider:
    backend = os.getenv("OSS_BACKEND", "s3").lower()
    if backend == "oss":
        return OssStorage()
    return S3Storage()
