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
#: v2 adds a ``provenance`` envelope to every record (raw-log reference, token identity,
#: cycle_provenance, position_64d, HEXT reference-only note). v1 records stay valid and readable.
SCHEMA_V2 = "hekbv3.causal_log/2"
PROVENANCE_SCHEMA = "hekbv3.causal_log.provenance/1"
_STATUS = ("PRESENT", "MISSING")


class CausalLogBlobError(ValueError):
    pass


def _check_slot(i: int, name: str, slot: object) -> None:
    """A provenance slot is {"status": "PRESENT", "value": <original value>} or
    {"status": "MISSING", "missing_reason": <non-empty str>} -- never an empty fill-in."""
    if not isinstance(slot, dict) or slot.get("status") not in _STATUS:
        raise CausalLogBlobError(f"record {i}: provenance.{name} must carry status PRESENT or MISSING")
    if slot["status"] == "PRESENT":
        if "value" not in slot:
            raise CausalLogBlobError(f"record {i}: provenance.{name} is PRESENT but has no value")
    else:
        reason = slot.get("missing_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise CausalLogBlobError(f"record {i}: provenance.{name} is MISSING without a reason")
        if slot.get("value") is not None:
            raise CausalLogBlobError(f"record {i}: provenance.{name} is MISSING but carries a value")


def validate_provenance(i: int, rec: dict) -> None:
    """The v2 contract for one record (raises CausalLogBlobError)."""
    prov = rec.get("provenance")
    if not isinstance(prov, dict) or prov.get("schema") != PROVENANCE_SCHEMA:
        raise CausalLogBlobError(f"record {i}: provenance envelope ({PROVENANCE_SCHEMA}) is required")
    ref = prov.get("raw_ref")
    if not isinstance(ref, dict):
        raise CausalLogBlobError(f"record {i}: provenance.raw_ref is required")
    for key in ("file_sha256", "record_sha256"):
        v = ref.get(key)
        if not isinstance(v, str) or len(v) != 64:
            raise CausalLogBlobError(f"record {i}: provenance.raw_ref.{key} must be a sha256 hex string")
    if not isinstance(ref.get("line_number"), int) or ref["line_number"] < 1:
        raise CausalLogBlobError(f"record {i}: provenance.raw_ref.line_number (record position) is required")
    if prov.get("values_origin") != "ingest_time_raw_log":
        raise CausalLogBlobError(f"record {i}: provenance.values_origin must name the origin of the stored values")
    tok = prov.get("token")
    _check_slot(i, "token", tok)
    if isinstance(tok, dict) and tok.get("status") == "PRESENT":
        v = tok["value"]
        if not isinstance(v, dict) or "token_id" not in v or "step_index" not in v:
            raise CausalLogBlobError(f"record {i}: provenance.token.value needs token_id and step_index as separate fields")
        if v["step_index"] != rec.get("token_index"):
            raise CausalLogBlobError(f"record {i}: provenance.token.step_index != token_index")
    _check_slot(i, "cycle_provenance", prov.get("cycle_provenance"))
    _check_slot(i, "position_64d", prov.get("position_64d"))
    hext = prov.get("hext")
    if not isinstance(hext, dict) or hext.get("kind") != "reference_only" or hext.get("document") != "NOT_STORED":
        raise CausalLogBlobError(f"record {i}: provenance.hext must say reference_only / document NOT_STORED")


def encode_causal_log_blob(records: list[dict], *, schema: str = SCHEMA) -> tuple[bytes, str]:
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
    if schema == SCHEMA_V2:
        for i, rec in enumerate(records):
            validate_provenance(i, rec)
    elif schema != SCHEMA:
        raise CausalLogBlobError(f"unknown causal-log schema {schema!r}")
    data = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return data, hashlib.sha256(data).hexdigest()


def decode_causal_log_blob(data: bytes) -> list[dict]:
    records = json.loads(data.decode("utf-8"))
    if not isinstance(records, list):
        raise CausalLogBlobError("causal log blob does not decode to a list")
    return records
