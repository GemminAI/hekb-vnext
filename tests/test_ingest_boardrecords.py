"""HEKBV3-BUILD002 §2/§1 (確認): unit tests for scripts/ingest_boardrecords.py against
synthetic manifest.json + {session_id}.jsonl.gz shards (BoardRecord.to_wire() shape) --
validation48/ does not exist yet (checked: /media/psf/HEKBv3/exp_hekbv3_boardrecords/ is
not present on this VM), so these are the "実装と単体テストまで" stopping point, not a run
against the real Mac-produced data. Every fixture uses a throwaway tmp_path data dir."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

CORE_32 = [f"P{str(i).zfill(2)}" for i in range(1, 39) if f"P{str(i).zfill(2)}" not in
           ("P05", "P11", "P12", "P17", "P34", "P37")]
ANCHOR_POLICY = "a" * 64


def _board_record(session_id: str, seq: int, token_index: int, values: list[int],
                   anchor_policy: str = ANCHOR_POLICY) -> dict:
    return {
        "format": "36cell/1", "session_id": session_id, "seq": seq, "token_index": token_index,
        "prev_hash": "0" * 64, "anchor_policy_sha256": anchor_policy,
        "anchor_counts": {"config_bootstrap": 0, "dynamic": 0, "manual": 0},
        "causal_log_ref": "0" * 64,
        "cells": [[p, "OK", v] for p, v in zip(CORE_32, values, strict=True)],
        "calibration": "未較正", "summary": "", "crc": 0, "record_hash": "", "hmac": "",
    }


def _write_shard(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _make_session_shard(root: Path, session_id: str, n_steps: int, seed: int,
                         anchor_policy: str = ANCHOR_POLICY) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    records = []
    for t in range(n_steps):
        values = [int(v) for v in rng.integers(-1_000_000, 1_000_000, size=32)]
        records.append(_board_record(session_id, seq=t, token_index=t, values=values, anchor_policy=anchor_policy))
    rel_path = f"{session_id}.jsonl.gz"
    _write_shard(root / rel_path, records)
    return {"session_id": session_id, "file": rel_path, "n_records": n_steps}


def _manifest(sessions: list[dict], **overrides) -> dict:
    manifest = {
        "sessions": sessions, "board_format": "36cell/1", "quant_step_log2": -30,
        "format_version": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
        "anchor_policy_sha256": ANCHOR_POLICY,
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture
def two_sessions(tmp_path):
    shards_root = tmp_path / "shards"
    s0 = _make_session_shard(shards_root, "sess-0", n_steps=12, seed=0)
    s1 = _make_session_shard(shards_root, "sess-1", n_steps=12, seed=1)
    sessions = [
        {**s0, "category": "catA", "group": "g0", "split": "dev"},
        {**s1, "category": "catB", "group": "g1", "split": "dev"},
    ]
    manifest_path = shards_root / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(sessions)))
    return manifest_path, shards_root


def test_ingest_end_to_end_and_idempotent(two_sessions, tmp_path, monkeypatch):
    manifest_path, shards_root = two_sessions
    data_dir = tmp_path / "data"

    import ingest_boardrecords as ib

    result1 = ib.run_ingest(
        manifest_path=manifest_path, shards_root=shards_root, data_dir=data_dir,
        model_ref="toy-model", layer=7,
    )
    assert result1["n_sessions"] == 2
    object_ids_1 = {s["session_id"]: s["object_id"] for s in result1["sessions"]}

    result2 = ib.run_ingest(
        manifest_path=manifest_path, shards_root=shards_root, data_dir=data_dir,
        model_ref="toy-model", layer=7,
    )
    object_ids_2 = {s["session_id"]: s["object_id"] for s in result2["sessions"]}
    assert object_ids_1 == object_ids_2, "re-running ingest must produce the same object_id (idempotent)"
    assert result1["space_id"] == result2["space_id"]


def test_ingest_stops_on_record_count_mismatch(two_sessions, tmp_path):
    manifest_path, shards_root = two_sessions
    manifest = json.loads(manifest_path.read_text())
    manifest["sessions"][0]["n_records"] = 999  # shard actually has 12
    manifest_path.write_text(json.dumps(manifest))

    import ingest_boardrecords as ib

    with pytest.raises(ib.IngestError, match="n_records"):
        ib.run_ingest(
            manifest_path=manifest_path, shards_root=shards_root, data_dir=tmp_path / "data",
            model_ref="toy-model", layer=7,
        )


def test_ingest_stops_on_inconsistent_anchor_policy(tmp_path):
    shards_root = tmp_path / "shards"
    records = [
        _board_record("sess-x", seq=0, token_index=0, values=[0] * 32, anchor_policy=ANCHOR_POLICY),
        _board_record("sess-x", seq=1, token_index=1, values=[0] * 32, anchor_policy="b" * 64),
    ]
    _write_shard(shards_root / "sess-x.jsonl.gz", records)
    sessions = [{"session_id": "sess-x", "file": "sess-x.jsonl.gz", "n_records": 2,
                 "category": "catA", "group": "g0", "split": "dev"}]
    manifest_path = shards_root / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(sessions)))

    import ingest_boardrecords as ib

    with pytest.raises(ib.IngestError, match="anchor_policy_sha256"):
        ib.run_ingest(
            manifest_path=manifest_path, shards_root=shards_root, data_dir=tmp_path / "data",
            model_ref="toy-model", layer=7,
        )


def test_ingest_then_build_index_and_compose(two_sessions, tmp_path, monkeypatch):
    """Chains ingest -> /v3/index/{space_id}/build -> /v3/compose on a fresh TestClient
    bound to the same data dir -- the structural shape of BUILD002 §1's "第1の確認", run
    here against synthetic data since validation48/ does not exist yet."""
    manifest_path, shards_root = two_sessions
    data_dir = tmp_path / "data"

    import ingest_boardrecords as ib

    result = ib.run_ingest(
        manifest_path=manifest_path, shards_root=shards_root, data_dir=data_dir,
        model_ref="toy-model", layer=7,
    )
    space_id = result["space_id"]

    monkeypatch.setenv("HEKB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEKB_INDEX_DIR", str(tmp_path / "index_local"))
    monkeypatch.setenv("HEKB_AUDIT_PUBLIC_KEY_PATH", str(data_dir / ".ingest_dev_public_key.raw"))

    import store_service
    from starlette.testclient import TestClient

    store_service._bearer_guard.token = "t"
    client = TestClient(store_service.app, headers={"Authorization": "Bearer t"})

    build_resp = client.post(f"/v3/index/{space_id}/build", json={"L": 8})
    assert build_resp.status_code == 202, build_resp.text
    assert build_resp.json()["n_entries"] > 0

    query_object_id = result["sessions"][0]["object_id"]
    compose_resp = client.post("/v3/compose", json={
        "space_id": space_id, "query_object_id": query_object_id, "query_token_index": 10,
        "exclude": {"groups": [result["sessions"][0]["group"]]},
    })
    assert compose_resp.status_code == 200, compose_resp.text
    body = compose_resp.json()
    assert body["category_posterior"]
    assert all(v is None for v in body["value_by_horizon"].values())
