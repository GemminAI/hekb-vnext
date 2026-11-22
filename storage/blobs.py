"""Layer 4 (blobs): `blobs/<sha256>`, immutable, content-addressed, no delete
(HEKBv3 改修設計書 v0.1 §3.3). Generic byte store -- unlike `storage/trajectory_blob.py`
(a specific binary layout for one kind of payload), this holds whatever bytes a caller
PUTs, addressed by the SHA-256 of exactly those bytes (the "受け取ったバイト列のsha256"
in §5's own description of `PUT /v3/blobs/{sha256}`).

[HEKBV3-BUILD001, disclosed]: a Trajectory's `blob_sha256` (written into
`ext.hekbv3.trajectory.blob_sha256`) is therefore the hash of the *whole* TrajectoryBlob
file (header + zstd payload + tail) as stored here -- not `storage.trajectory_blob`'s own
internal `trajectory_id` (which is the SHA-256 of only the raw *uncompressed* payload,
used for that format's own internal integrity check). These are two different hashes of
two different byte ranges; this module's content-address is always "hash of the full
blob bytes passed to `put_blob`", matching a generic blob store's usual contract.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


class BlobError(ValueError):
    pass


def blob_path(blobs_root: Path, sha256_hex: str) -> Path:
    return blobs_root / sha256_hex[:2] / sha256_hex[2:4] / sha256_hex


def put_blob(blobs_root: Path, claimed_sha256_hex: str, data: bytes) -> str:
    """Verifies `sha256(data) == claimed_sha256_hex`, then writes it (idempotent: if the
    exact same bytes are already there, this is a no-op; content-addressing makes a
    mismatch under the same hash impossible by construction, not something to special-case).
    Returns the (lowercased) hex digest actually used as the path key."""
    actual = hashlib.sha256(data).hexdigest()
    claimed = claimed_sha256_hex.lower()
    if actual != claimed:
        raise BlobError(f"sha256 mismatch: claimed {claimed}, computed {actual}")
    path = blob_path(blobs_root, actual)
    if path.exists():
        return actual
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_bytes(data)
    tmp_path.replace(path)
    return actual


def get_blob(blobs_root: Path, sha256_hex: str) -> bytes:
    path = blob_path(blobs_root, sha256_hex.lower())
    if not path.exists():
        raise BlobError(f"no blob found for sha256={sha256_hex}")
    return path.read_bytes()


def has_blob(blobs_root: Path, sha256_hex: str) -> bool:
    return blob_path(blobs_root, sha256_hex.lower()).exists()
