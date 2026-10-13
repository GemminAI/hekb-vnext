"""HEKBV3-BUILD004: pins the fix for the lost 38-port causal log (D-03/D-24 violation --
see REPORT_causal48.md). `scripts/ingest_boardrecords.py` never read the raw per-token
TRJ003 logs at all (only the already-reduced BoardRecord shard), so there was nothing in
`ext.hekbv3` carrying the full 38-port values or `cycle_raw` (judgment/reason). These
tests exercise the new `--raw-logs-root` path end to end against synthetic fixtures
shaped like the real raw logs (same keys: token_index, seq, trajectory_hash, ports,
cycle_raw) and BoardRecord shards (same keys as tests/test_ingest_boardrecords.py)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_ingest_boardrecords import ANCHOR_POLICY, _manifest, _make_session_shard  # noqa: E402


def _raw_log_record(token_index: int, seq: int, n_ports: int = 38) -> dict:
    ports = {f"P{str(i).zfill(2)}": float(token_index * 1000 + i) for i in range(1, n_ports + 1)}
    return {
        "token_index": token_index, "seq": seq,
        "trajectory_hash": f"th-{token_index:04d}",
        "ports": ports,
        "cycle_raw": {
            "cycle": seq,
            "control": {"tier_label": "L1", "reason": f"reason-{token_index}"},
            "risk": {"state": "UNDEFINED"},
            "field": {"classification": "FLAT"},
        },
    }


def _write_raw_log(path: Path, n_steps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for t in range(n_steps):
            fh.write(json.dumps(_raw_log_record(t, t)) + "\n")


@pytest.fixture
def two_sessions_with_raw_logs(tmp_path):
    shards_root = tmp_path / "shards"
    s0 = _make_session_shard(shards_root, "sess-0", n_steps=12, seed=0)
    s1 = _make_session_shard(shards_root, "sess-1", n_steps=12, seed=1)
    sessions = [
        {**s0, "category": "catA", "group": "g0", "split": "dev"},
        {**s1, "category": "catB", "group": "g1", "split": "dev"},
    ]
    manifest_path = shards_root / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(
        sessions, nvs_kernel_commit="kernel" + "c" * 34, nvs_runtime_commit="runtime" + "r" * 33,
    )))

    raw_logs_root = tmp_path / "raw"
    _write_raw_log(raw_logs_root / "sessions" / "TRJ003_000_sess-0.jsonl.gz", n_steps=12)
    _write_raw_log(raw_logs_root / "sessions" / "TRJ003_001_sess-1.jsonl.gz", n_steps=12)
    return manifest_path, shards_root, raw_logs_root


def _get_blob(client, sha256: str) -> bytes:
    resp = client.get(f"/v3/blobs/{sha256}")
    assert resp.status_code == 200, resp.text
    return resp.content


def test_ingest_with_raw_logs_root_adds_causal_log(two_sessions_with_raw_logs, tmp_path, monkeypatch):
    manifest_path, shards_root, raw_logs_root = two_sessions_with_raw_logs
    data_dir = tmp_path / "data"

    import ingest_boardrecords as ib

    result = ib.run_ingest(
        manifest_path=manifest_path, shards_root=shards_root, data_dir=data_dir,
        raw_logs_root=raw_logs_root,
    )
    assert result["n_sessions"] == 2
    assert all(s["has_causal_log"] for s in result["sessions"])
    assert result["nvs_kernel_commit"] == "kernel" + "c" * 34
    assert result["nvs_runtime_commit"] == "runtime" + "r" * 33

    monkeypatch.setenv("HEKB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEKB_AUDIT_PUBLIC_KEY_PATH", str(data_dir / ".ingest_dev_public_key.raw"))
    import store_service
    from starlette.testclient import TestClient

    store_service._bearer_guard.token = "t"
    client = TestClient(store_service.app, headers={"Authorization": "Bearer t"})

    from storage.causal_log_blob import decode_causal_log_blob

    for s in result["sessions"]:
        record = store_service._read_json(store_service._object_path(s["object_id"]))
        causal_log = record["payload"]["ext"]["hekbv3"]["causal_log"]
        assert causal_log["n_steps"] == 12
        assert causal_log["schema"] == "hekbv3.causal_log/1"
        assert causal_log["source_file"]["path"] == f"sessions/TRJ003_00{'0' if s['session_id']=='sess-0' else '1'}_{s['session_id']}.jsonl.gz"

        data = _get_blob(client, causal_log["blob_sha256"])
        decoded = decode_causal_log_blob(data)
        assert len(decoded) == 12
        for t, rec in enumerate(decoded):
            assert rec["token_index"] == t
            assert rec["seq"] == t
            assert rec["trajectory_hash"] == f"th-{t:04d}"
            assert len(rec["ports"]) == 38
            assert rec["cycle_raw"]["control"]["reason"] == f"reason-{t}"
            assert rec["cycle_raw"]["risk"]["state"] == "UNDEFINED"
            assert rec["line_number"] == t + 1
            assert rec["source_file"]["sha256"] == causal_log["source_file"]["sha256"]


def test_ingest_causal_log_idempotent(two_sessions_with_raw_logs, tmp_path):
    manifest_path, shards_root, raw_logs_root = two_sessions_with_raw_logs
    data_dir = tmp_path / "data"

    import ingest_boardrecords as ib

    result1 = ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root,
                             data_dir=data_dir, raw_logs_root=raw_logs_root)
    result2 = ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root,
                             data_dir=data_dir, raw_logs_root=raw_logs_root)
    ids1 = {s["session_id"]: s["object_id"] for s in result1["sessions"]}
    ids2 = {s["session_id"]: s["object_id"] for s in result2["sessions"]}
    assert ids1 == ids2


def test_ingest_stops_on_raw_log_token_mismatch(two_sessions_with_raw_logs, tmp_path):
    manifest_path, shards_root, raw_logs_root = two_sessions_with_raw_logs
    # corrupt sess-0's raw log: shift token_index by 1 on every record
    bad_path = raw_logs_root / "sessions" / "TRJ003_000_sess-0.jsonl.gz"
    bad_path.unlink()
    with gzip.open(bad_path, "wt", encoding="utf-8") as fh:
        for t in range(12):
            rec = _raw_log_record(t + 1, t)  # token_index off by one
            fh.write(json.dumps(rec) + "\n")

    import ingest_boardrecords as ib

    with pytest.raises(ib.IngestError, match="does not align"):
        ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root,
                       data_dir=tmp_path / "data", raw_logs_root=raw_logs_root)


def test_ingest_stops_on_missing_raw_log(two_sessions_with_raw_logs, tmp_path):
    manifest_path, shards_root, raw_logs_root = two_sessions_with_raw_logs
    (raw_logs_root / "sessions" / "TRJ003_001_sess-1.jsonl.gz").unlink()

    import ingest_boardrecords as ib

    with pytest.raises(ib.IngestError, match="no raw log file found"):
        ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root,
                       data_dir=tmp_path / "data", raw_logs_root=raw_logs_root)


def test_ingest_without_raw_logs_root_has_no_causal_log(two_sessions_with_raw_logs, tmp_path):
    manifest_path, shards_root, _raw_logs_root = two_sessions_with_raw_logs

    import ingest_boardrecords as ib

    result = ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root, data_dir=tmp_path / "data")
    assert all(not s["has_causal_log"] for s in result["sessions"])


def test_ingest_writes_provenance_manifest(two_sessions_with_raw_logs, tmp_path, monkeypatch):
    manifest_path, shards_root, raw_logs_root = two_sessions_with_raw_logs
    data_dir = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["ingest_boardrecords.py", "--manifest", str(manifest_path),
                                       "--data-dir", str(data_dir), "--raw-logs-root", str(raw_logs_root)])

    import ingest_boardrecords as ib

    result = ib.run_ingest(manifest_path=manifest_path, shards_root=shards_root,
                            data_dir=data_dir, raw_logs_root=raw_logs_root)
    ib._write_provenance_manifest(data_dir, result=result, manifest_path=manifest_path,
                                   raw_logs_root=raw_logs_root)

    prov = json.loads((data_dir / "ingest_manifest.json").read_text())
    assert prov["argv"] == sys.argv
    assert prov["nvs_kernel_commit"] == "kernel" + "c" * 34
    assert prov["nvs_runtime_commit"] == "runtime" + "r" * 33
    assert prov["source_manifest_sha256"] == __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    assert prov["hekb_vnext_v3_commit"]  # non-empty: this repo's own commit, for reproducibility


def test_causal_log_validated_by_require_hekbv3_trajectory_ext(tmp_path, monkeypatch):
    """store_service-level: a referenced causal_log's blob must exist (same contract as
    `ports`/`trajectory`) -- this payload's `ports` blob doesn't exist either, so the
    exact rejection reason isn't pinned down to causal_log specifically here; see
    test_require_causal_log.py for causal_log's own dedicated (and, since
    feature/require-causal-log, no-longer-optional-by-default) behavior."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("HEKB_DATA_DIR", str(data_dir))
    monkeypatch.setenv("HEKB_AUDIT_PUBLIC_KEY_PATH", str(tmp_path / "nonexistent-key"))

    import store_service
    from store_service import ExperiencePayload

    space = {
        "model_ref": "m", "layer": 1, "projection_sha256": "0" * 64, "w_sha256": "0" * 64,
        "anchor_policy_sha256": "0" * 64, "wiring_policy_sha256": "0" * 64, "time_policy_sha256": "0" * 64,
        "board_format": "36cell/1", "quant_step_log2": -30,
    }
    space_id = store_service.compute_space_id(space)
    payload = ExperiencePayload(
        schema="hekb.experience/2", raw_input="x", resolved_meaning=None, action=None,
        title="t", tags=[],
        ext={"hekbv3": {
            "kind": "trajectory", "origin": "experience",
            "space": {**space, "space_id": space_id},
            "ports": {"blob_sha256": "a" * 64},
            "causal_log": {"blob_sha256": "b" * 64},
        }},
    )
    with pytest.raises(Exception) as excinfo:
        store_service._require_hekbv3_trajectory_ext(payload)
    assert "causal_log" in str(excinfo.value) or "not found" in str(excinfo.value)
