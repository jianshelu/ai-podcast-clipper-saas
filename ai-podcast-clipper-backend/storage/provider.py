from __future__ import annotations

from typing import Protocol


class StorageProvider(Protocol):
    def head(self, key: str) -> bool:
        ...

    def download(self, key: str, destination: str) -> None:
        ...

    def upload(self, source: str, key: str) -> None:
        ...
