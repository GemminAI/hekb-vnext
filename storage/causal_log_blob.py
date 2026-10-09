"""HEKBV3-BUILD004: causal-log blob encoding (D-03/D-24 -- the 38-port causal log, kept
separately from the `ports` board blob, which only ever carried the already-quantized
32-cell board values; the causal log is the full per-token P01-P38 readings plus the raw
NVS cycle (`cycle_raw`: judgment/reason -- `control.reason`, `control.tier_label`,
`risk.state`, `field.classification`, etc.) -- see
HEKBv3_改修設計書_v0.1_追補A_経験と知識.md §3 ("経験には outcome...失敗を必ず記録する").

Unlike `storage/trajectory_blob.py` (a fixed-layout binary format for homogeneous numeric
arrays), the causal log is semi-structured JSON of varying shape per record (`cycle_raw`
nests dicts/lists/strings/floats/None) -- storing it as canonical JSON bytes in the
existing generic blob store (`storage/blobs.py`, Layer 4 per the design doc §3.3) is the
natural fit; no new binary layout is introduced.

One causal-log blob covers one session's full token sequence (mirrors the existing
`ports`/`trajectory` blobs, which are also one-per-session), as a JSON array of per-token
records, each:
    {"token_index": int, "seq": int, "trajectory_hash": str, "ports": {<port_id>: float},
     "cycle_raw": <the raw NVS cycle dict, verbatim>,
     "source_file": {"path": str, "sha256": str}, "line_number": int}
`source_file`/`line_number` are the task's own requirement ("元の生ログファイルのsha256と
行番号") -- provenance back to the exact raw-log byte range this record came from.
"""
from __future__ import annotations

import hashlib
import json

SCHEMA = "hekbv3.causal_log/1"


class CausalLogBlobError(ValueError):
    pass


def encode_causal_log_blob(records: list[dict]) -> tuple[bytes, str]:
    """records: one dict per token (see module docstring for required keys), already in
    token order. Returns (canonical JSON bytes, sha256 hex of those bytes). Canonical
    (sort_keys, no extra whitespace) so the same logical content always produces the same
    bytes -- needed for the ingest's own idempotency guarantee (same input -> same
    object_id across re-runs, K8)."""
    if not records:
        raise CausalLogBlobError("causal log has zero records")
    required = ("token_index", "seq", "trajectory_hash", "ports", "cycle_raw", "source_file", "line_number")
    for i, rec in enumerate(records):
        missing = [k for k in required if k not in rec]
        if missing:
            raise CausalLogBlobError(f"record {i} missing field(s): {missing}")
    data = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()


def decode_causal_log_blob(data: bytes) -> list[dict]:
    records = json.loads(data.decode("utf-8"))
    if not isinstance(records, list):
        raise CausalLogBlobError("causal log blob does not decode to a list")
    return records
