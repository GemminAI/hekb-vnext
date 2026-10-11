"""feature/require-causal-log: POST /v3/trajectories now rejects a trajectory with no
causal_log unless the caller explicitly says why (`causal_log_omitted_reason`), and
validates a *present* causal_log's actual content -- every one of the 38 Core Ports and
`cycle_raw`, for every token -- not just that its blob reference exists (D-03/D-24).

Every fixture uses a throwaway tmp_path data dir and a fresh dev keypair (never the
real 8300 service, never /media/psf/HEKBv3)."""
from __future__ import annotations

import base64
import hashlib
import secrets
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ALL_38 = tuple(f"P{n:02d}" for n in range(1, 39))
SPACE = {
    "model_ref": "toy-model", "layer": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": "a" * 64, "board_format": "36cell/1", "quant_step_log2": -30,
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HEKB_DATA_DIR", str(tmp_path / "experience"))
    monkeypatch.setenv("HEKB_INDEX_DIR", str(tmp_path / "index_local"))

    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    raw_point = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")
    public_key_path = tmp_path / "dev_public_key.raw"
    public_key_path.write_bytes(raw_point)
    monkeypatch.setenv("HEKB_AUDIT_PUBLIC_KEY_PATH", str(public_key_path))

    import store_service

    token = secrets.token_hex(32)
    store_service._bearer_guard.token = token

    def sign(payload: dict) -> str:
        message = store_service.canonical_bytes(payload)
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(signature).decode("ascii")

    with TestClient(store_service.app, headers={"Authorization": f"Bearer {token}"}) as test_client:
        test_client.sign = sign  # type: ignore[attr-defined]
        test_client.data_dir = tmp_path / "experience"  # type: ignore[attr-defined]
        yield test_client


def _put_ports_blob(client: TestClient, n_steps: int) -> str:
    from storage.trajectory_blob import encode_trajectory_blob

    rng = np.random.default_rng(1)
    rows = rng.normal(size=(n_steps, 38)).tolist()
    file_bytes, _header = encode_trajectory_blob(rows, dimensions=38, created_at_ms=0)
    sha = hashlib.sha256(file_bytes).hexdigest()
    resp = client.put(f"/v3/blobs/{sha}", content=file_bytes, headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 201, resp.text
    return sha


def _put_causal_log_blob(client: TestClient, records: list[dict]) -> str:
    from storage.causal_log_blob import encode_causal_log_blob

    data, sha256 = encode_causal_log_blob(records)
    resp = client.put(f"/v3/blobs/{sha256}", content=data, headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 201, resp.text
    return sha256


def _complete_causal_log_records(n_steps: int) -> list[dict]:
    return [
        {
            "token_index": t, "seq": t, "trajectory_hash": "t" * 64,
            "ports": {pid: {"status": "OK", "result": float(t)} for pid in ALL_38},
            "cycle_raw": {"step": t},
            "source_file": {"path": "synthetic", "sha256": "s" * 64}, "line_number": t + 1,
        }
        for t in range(n_steps)
    ]


def _base_payload(client: TestClient, *, ports_sha: str, n_steps: int, extra_hekbv3: dict) -> dict:
    space_resp = client.post("/v3/space-id", json=SPACE)
    assert space_resp.status_code == 200, space_resp.text
    space_id = space_resp.json()["space_id"]
    ext_hekbv3 = {
        "kind": "trajectory", "origin": "experience",
        "space": {**SPACE, "space_id": space_id},
        "ports": {"blob_sha256": ports_sha, "n_steps": n_steps, "dim": 38, "port_ids": list(ALL_38)},
        "labels": {"category": "cat", "split": "dev", "group": "g"},
        "outcome": {"result": "n/a"},
        **extra_hekbv3,
    }
    return {
        "schema": "hekb.experience/2", "raw_input": "x", "resolved_meaning": None,
        "action": None, "title": "t", "tags": [],
        "ext": {"hekbv3": ext_hekbv3},
    }


def _post(client: TestClient, payload: dict):
    sig = client.sign(payload)
    return client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": sig})


def test_missing_causal_log_with_no_flag_is_rejected(client: TestClient) -> None:
    n_steps = 5
    ports_sha = _put_ports_blob(client, n_steps)
    payload = _base_payload(client, ports_sha=ports_sha, n_steps=n_steps, extra_hekbv3={})

    resp = _post(client, payload)
    assert resp.status_code == 422, resp.text
    assert "causal_log" in resp.text


def test_empty_omitted_reason_is_still_rejected(client: TestClient) -> None:
    n_steps = 5
    ports_sha = _put_ports_blob(client, n_steps)
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log_omitted_reason": "   "},
    )

    resp = _post(client, payload)
    assert resp.status_code == 422, resp.text


def test_explicit_omitted_reason_is_accepted_and_recorded(client: TestClient) -> None:
    n_steps = 5
    ports_sha = _put_ports_blob(client, n_steps)
    reason = "pre-causal_log TRJ003 run (NVS-Kernel v6.1.3)"
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log_omitted_reason": reason},
    )

    resp = _post(client, payload)
    assert resp.status_code == 201, resp.text
    object_id = resp.json()["object_id"]

    stored = (client.data_dir / "objects" / f"{object_id}.json").read_text()
    assert reason in stored  # never silently dropped -- D-03/D-24


def test_causal_log_missing_a_port_in_one_record_is_rejected(client: TestClient) -> None:
    n_steps = 3
    ports_sha = _put_ports_blob(client, n_steps)
    records = _complete_causal_log_records(n_steps)
    del records[1]["ports"]["P17"]  # one port missing, one token in
    causal_sha = _put_causal_log_blob(client, records)
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log": {"blob_sha256": causal_sha, "n_steps": n_steps}},
    )

    resp = _post(client, payload)
    assert resp.status_code == 422, resp.text
    assert "P17" in resp.text and "record 1" in resp.text


def test_causal_log_missing_cycle_raw_in_one_record_is_rejected(client: TestClient) -> None:
    n_steps = 3
    ports_sha = _put_ports_blob(client, n_steps)
    records = _complete_causal_log_records(n_steps)
    # encode_causal_log_blob itself requires the *key* to be present, so this sets the
    # value to None instead of deleting the key -- still something the server-side
    # check must reject, just at the value level rather than the key level.
    records[2]["cycle_raw"] = None
    causal_sha = _put_causal_log_blob(client, records)
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log": {"blob_sha256": causal_sha, "n_steps": n_steps}},
    )

    resp = _post(client, payload)
    assert resp.status_code == 422, resp.text
    assert "cycle_raw" in resp.text and "record 2" in resp.text


def test_causal_log_n_steps_mismatch_is_rejected(client: TestClient) -> None:
    n_steps = 4
    ports_sha = _put_ports_blob(client, n_steps)
    records = _complete_causal_log_records(n_steps)
    causal_sha = _put_causal_log_blob(client, records)
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log": {"blob_sha256": causal_sha, "n_steps": n_steps + 1}},
    )

    resp = _post(client, payload)
    assert resp.status_code == 422, resp.text
    assert "n_steps" in resp.text


def test_complete_causal_log_is_accepted(client: TestClient) -> None:
    n_steps = 3
    ports_sha = _put_ports_blob(client, n_steps)
    records = _complete_causal_log_records(n_steps)
    causal_sha = _put_causal_log_blob(client, records)
    payload = _base_payload(
        client, ports_sha=ports_sha, n_steps=n_steps,
        extra_hekbv3={"causal_log": {"blob_sha256": causal_sha, "n_steps": n_steps}},
    )

    resp = _post(client, payload)
    assert resp.status_code == 201, resp.text
