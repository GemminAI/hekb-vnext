"""Blob files of a data dir (``blobs/aa/bb/<sha256>``), read and written by content hash."""

from __future__ import annotations

import hashlib
from pathlib import Path


def blob_path(data_dir: Path, sha: str) -> Path:
    return Path(data_dir) / "blobs" / sha[:2] / sha[2:4] / sha


def read_blob(data_dir: Path, sha: str) -> bytes:
    data = blob_path(data_dir, sha).read_bytes()
    if hashlib.sha256(data).hexdigest() != sha:
        raise ValueError(f"blob {sha} does not match its content hash")
    return data


def write_blob(data_dir: Path, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = blob_path(data_dir, sha)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"blob {sha} exists with different bytes")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return sha
