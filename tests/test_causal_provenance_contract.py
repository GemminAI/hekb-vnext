"""CAUSAL-ALIGN002: the provenance contract of causal-log schema v2 (raw-log reference, token identity,
cycle_provenance / position_64d kept verbatim or MISSING with a reason, HEXT reference-only) and the
duplicate-event rule (identical retry vs conflicting duplicate). Throwaway data dirs only."""
from __future__ import annotations

import base64
import gzip
import hashlib
import importlib.util
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from storage.causal_log_blob import (  # noqa: E402
    PROVENANCE_SCHEMA,
    SCHEMA_V2,
    CausalLogBlobError,
    encode_causal_log_blob,
)

ALL_38 = tuple(f"P{n:02d}" for n in range(1, 39))
SPACE = {
    "model_ref": "toy-model", "layer": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": "a" * 64, "board_format": "36cell/1", "quant_step_log2": -30,
}


def _prov(t: int, *, token=None, cycle=None, pos=None) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA, "values_origin": "ingest_time_raw_log",
        "raw_ref": {"file_sha256": "f" * 64, "line_number": t + 1, "record_sha256": "r" * 64, "path_aux": "x"},
        "token": token or {"status": "PRESENT", "value": {"token_id": 1000 + t, "step_index": t, "gen": {"token_id": 1000 + t}}},
        "cycle_provenance": cycle or {"status": "PRESENT", "value": [{"hext_object_id": "h"}]},
        "position_64d": pos or {"status": "PRESENT", "value": [0.0] * 64},
        "hext": {"kind": "reference_only", "object_id": "h", "document": "NOT_STORED"},
    }


def _records(n: int = 3, **kw) -> list[dict]:
    return [
        {"token_index": t, "seq": t, "trajectory_hash": "t" * 64,
         "ports": {pid: {"status": "OK", "result": float(t)} for pid in ALL_38}, "cycle_raw": {"step": t},
         "source_file": {"path": "p", "sha256": "f" * 64}, "line_number": t + 1, "provenance": _prov(t, **kw)}
        for t in range(n)
    ]


def test_v2_happy_path_is_deterministic():
    a = encode_causal_log_blob(_records(), schema=SCHEMA_V2)
    b = encode_causal_log_blob(_records(), schema=SCHEMA_V2)
    assert a == b


def test_token_id_and_step_are_separate_fields():
    recs = _records()
    v = recs[1]["provenance"]["token"]["value"]
    assert v["token_id"] != v["step_index"]
    recs[1]["provenance"]["token"]["value"]["step_index"] = 99  # conflated / wrong step
    with pytest.raises(CausalLogBlobError, match="step_index"):
        encode_causal_log_blob(recs, schema=SCHEMA_V2)
    recs = _records()
    del recs[0]["provenance"]["token"]["value"]["token_id"]
    with pytest.raises(CausalLogBlobError, match="token_id"):
        encode_causal_log_blob(recs, schema=SCHEMA_V2)


@pytest.mark.parametrize("slot", ["token", "cycle_provenance", "position_64d"])
def test_missing_item_needs_a_reason_and_no_fill_in(slot):
    kw = {"token": None, "cycle": None, "pos": None}
    key = {"token": "token", "cycle_provenance": "cycle", "position_64d": "pos"}[slot]
    kw[key] = {"status": "MISSING", "value": None}
    with pytest.raises(CausalLogBlobError, match="MISSING without a reason"):
        encode_causal_log_blob(_records(**kw), schema=SCHEMA_V2)
    kw[key] = {"status": "MISSING", "value": None, "missing_reason": "raw record has no field"}
    encode_causal_log_blob(_records(**kw), schema=SCHEMA_V2)  # explicit reason: stored
    kw[key] = {"status": "MISSING", "value": [], "missing_reason": "r"}  # an empty fill-in is refused
    with pytest.raises(CausalLogBlobError, match="MISSING but carries a value"):
        encode_causal_log_blob(_records(**kw), schema=SCHEMA_V2)


def test_provenance_envelope_and_raw_reference_are_required():
    recs = _records()
    del recs[0]["provenance"]
    with pytest.raises(CausalLogBlobError, match="envelope"):
        encode_causal_log_blob(recs, schema=SCHEMA_V2)
    recs = _records()
    recs[0]["provenance"]["raw_ref"]["record_sha256"] = "short"
    with pytest.raises(CausalLogBlobError, match="record_sha256"):
        encode_causal_log_blob(recs, schema=SCHEMA_V2)
    recs = _records()
    recs[0]["provenance"]["hext"]["document"] = "RECONSTRUCTED"
    with pytest.raises(CausalLogBlobError, match="reference_only"):
        encode_causal_log_blob(recs, schema=SCHEMA_V2)


def test_v1_records_still_encode():
    recs = _records()
    for r in recs:
        del r["provenance"]
    encode_causal_log_blob(recs)  # schema v1 default, unchanged


# ---- raw-log reader: duplicates and missing fields ----------------------------------------------------
def _load_ingest():
    spec = importlib.util.spec_from_file_location("ingest_boardrecords", ROOT / "scripts" / "ingest_boardrecords.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _raw_line(t: int, *, gen=True, prov=True, pos=True, note="") -> str:
    rec = {"token_index": t, "seq": t, "trajectory_hash": f"h{t}{note}", "ports": {p: {} for p in ALL_38},
           "cycle_raw": {"hext": {"object_id": f"x{t}"}}}
    if gen:
        rec["gen"] = {"token_id": 500 + t, "is_generated": True}
    if prov:
        rec["cycle_provenance"] = [{"hext_object_id": f"x{t}"}]
    if pos:
        rec["position_64d"] = [0.5] * 64
    return json.dumps(rec, sort_keys=True)


def _write_raw(tmp_path: Path, lines: list[str]) -> Path:
    root = tmp_path / "raw"
    (root / "sessions").mkdir(parents=True)
    with gzip.open(root / "sessions" / "TRJ003_000_s1.jsonl.gz", "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return root


def _board(n: int) -> list[dict]:
    return [{"token_index": t, "seq": t} for t in range(n)]


def test_ingest_preserves_three_things_separately(tmp_path):
    m = _load_ingest()
    root = _write_raw(tmp_path, [_raw_line(t) for t in range(3)])
    recs, rel, sha = m._read_raw_causal_log(root, "s1", _board(3))
    p = recs[1]["provenance"]
    assert p["raw_ref"]["file_sha256"] == sha and p["raw_ref"]["line_number"] == 2
    assert p["raw_ref"]["record_sha256"] == hashlib.sha256(_raw_line(1).encode()).hexdigest()
    assert p["token"]["value"]["token_id"] == 501 and p["token"]["value"]["step_index"] == 1
    assert p["cycle_provenance"]["status"] == "PRESENT" and p["position_64d"]["status"] == "PRESENT"
    assert p["hext"] == {"kind": "reference_only", "object_id": "x1", "document": "NOT_STORED"}
    encode_causal_log_blob(recs, schema=SCHEMA_V2)


def test_ingest_missing_fields_are_marked_with_reasons(tmp_path):
    m = _load_ingest()
    root = _write_raw(tmp_path, [_raw_line(0, gen=False, prov=False, pos=False)])
    recs, _r, _s = m._read_raw_causal_log(root, "s1", _board(1))
    p = recs[0]["provenance"]
    for key in ("token", "cycle_provenance", "position_64d"):
        assert p[key]["status"] == "MISSING" and p[key]["value"] is None and p[key]["missing_reason"]
    encode_causal_log_blob(recs, schema=SCHEMA_V2)


def test_identical_retry_is_kept_once_conflict_is_refused(tmp_path):
    m = _load_ingest()
    root = _write_raw(tmp_path, [_raw_line(0), _raw_line(1), _raw_line(1), _raw_line(2)])
    recs, _r, _s = m._read_raw_causal_log(root, "s1", _board(3))
    assert [r["token_index"] for r in recs] == [0, 1, 2]
    assert recs[1]["provenance"]["duplicates"][0]["kind"] == "identical_retry"
    assert recs[1]["provenance"]["duplicates"][0]["line_number"] == 3
    root2 = _write_raw(tmp_path / "b", [_raw_line(0), _raw_line(1), _raw_line(1, note="X")])
    with pytest.raises(m.IngestError, match="DIFFERENT content"):
        m._read_raw_causal_log(root2, "s1", _board(2))


# ---- server: the contract is enforced on POST /v3/trajectories --------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HEKB_DATA_DIR", str(tmp_path / "experience"))
    monkeypatch.setenv("HEKB_INDEX_DIR", str(tmp_path / "index_local"))
    key = ec.generate_private_key(ec.SECP256R1())
    n = key.public_key().public_numbers()
    pub = tmp_path / "dev_public_key.raw"
    pub.write_bytes(b"\x04" + n.x.to_bytes(32, "big") + n.y.to_bytes(32, "big"))
    monkeypatch.setenv("HEKB_AUDIT_PUBLIC_KEY_PATH", str(pub))
    import store_service

    token = secrets.token_hex(32)
    store_service._bearer_guard.token = token

    def sign(payload: dict) -> str:
        return base64.b64encode(key.sign(store_service.canonical_bytes(payload), ec.ECDSA(hashes.SHA256()))).decode()

    with TestClient(store_service.app, headers={"Authorization": f"Bearer {token}"}) as c:
        c.sign = sign  # type: ignore[attr-defined]
        yield c


def _post(client, records, schema):
    from storage.trajectory_blob import encode_trajectory_blob

    rows = np.random.default_rng(1).normal(size=(len(records), 38)).tolist()
    fb, _h = encode_trajectory_blob(rows, dimensions=38, created_at_ms=0)
    ps = hashlib.sha256(fb).hexdigest()
    assert client.put(f"/v3/blobs/{ps}", content=fb).status_code == 201
    data = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    cs = hashlib.sha256(data).hexdigest()
    assert client.put(f"/v3/blobs/{cs}", content=data).status_code == 201
    sid = client.post("/v3/space-id", json=SPACE).json()["space_id"]
    payload = {
        "schema": "hekb.experience/2", "raw_input": "x", "resolved_meaning": None, "action": None, "title": "t", "tags": [],
        "ext": {"hekbv3": {"kind": "trajectory", "origin": "experience", "space": {**SPACE, "space_id": sid},
                           "ports": {"blob_sha256": ps, "n_steps": len(records), "dim": 38, "port_ids": list(ALL_38)},
                           "causal_log": {"blob_sha256": cs, "n_steps": len(records), "schema": schema},
                           "labels": {"category": "c", "split": "dev", "group": "g"}, "outcome": {"result": "n/a"}}},
    }
    return client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": client.sign(payload)})


def test_server_rejects_v2_without_provenance_and_accepts_complete(client):
    bad = _records()
    del bad[1]["provenance"]["raw_ref"]["line_number"]
    r = _post(client, bad, SCHEMA_V2)
    assert r.status_code == 422 and "provenance contract" in r.text
    good = _records()
    r = _post(client, good, SCHEMA_V2)
    assert r.status_code == 201, r.text
