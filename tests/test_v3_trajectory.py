"""HEKBV3-BUILD001: HTTP-layer tests for /v3/blobs, /v3/trajectories, /v3/index/*/build,
/v3/compose -- K5, K6 (run alongside the rest of tests/), K7, and the save->build->compose
(exclude)->restart->same-compose reuse loop (the restart leg is in
test_v3_restart_reproducibility.py, a real subprocess test, per K8's own "別プロセス").
Every fixture uses a throwaway tmp_path data dir and a fresh dev keypair (never the real
8300 service, never /media/psf/HEKBv3)."""
from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sys
from pathlib import Path

import numpy as np
import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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
        yield test_client


SPACE = {
    "model_ref": "toy-model", "layer": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": "a" * 64, "wiring_policy_sha256": "b" * 64, "time_policy_sha256": "c" * 64,
    "board_format": "36cell/1", "quant_step_log2": -30,
}
ANCHOR_10 = ("P01", "P03", "P09", "P13", "P18", "P19", "P25", "P28", "P31", "P35")
ALL_38 = tuple(f"P{str(i).zfill(2)}" for i in range(1, 39))


def _put_blob(client: TestClient, points: list[list[float]], dim: int) -> str:
    from storage.trajectory_blob import encode_trajectory_blob
    file_bytes, _header = encode_trajectory_blob(points, dimensions=dim)
    sha = hashlib.sha256(file_bytes).hexdigest()
    resp = client.put(f"/v3/blobs/{sha}", content=file_bytes, headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 201, resp.text
    return sha


def _make_ports_rows(rng: np.random.Generator, n_steps: int) -> list[list[float]]:
    rows = []
    for t in range(n_steps):
        row = rng.normal(size=38).tolist()
        if t == 0:
            for i, p in enumerate(ALL_38):
                if p not in ("P02", "P04"):  # a couple "always-OK" ports, rest ERROR at token0
                    row[i] = float("nan")
        rows.append(row)
    return rows


def _make_causal_log_blob(client: TestClient, rng: np.random.Generator, n_steps: int) -> tuple[str, int]:
    """feature/require-causal-log: every trajectory these tests post now needs a
    causal_log with all 38 Core Ports + cycle_raw per token -- independent of whatever
    width `ports_rows` the test's own board-cells blob carries (that one can legitimately
    be the 32-cell board subset; the causal log never is). Returns (blob_sha256,
    n_steps)."""
    from storage.causal_log_blob import encode_causal_log_blob

    records = [
        {
            "token_index": t, "seq": t, "trajectory_hash": "t" * 64,
            "ports": dict(zip(ALL_38, rng.normal(size=38).tolist(), strict=True)),
            "cycle_raw": {"step": t},
            "source_file": {"path": "synthetic", "sha256": "s" * 64}, "line_number": t + 1,
        }
        for t in range(n_steps)
    ]
    data, sha256 = encode_causal_log_blob(records)
    resp = client.put(f"/v3/blobs/{sha256}", content=data, headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 201, resp.text
    return sha256, n_steps


def _post_trajectory(client: TestClient, *, seed: int, category: str, group: str, space_id: str | None = None, n_steps: int = 20):
    rng = np.random.default_rng(seed)
    positions = [list(v / np.linalg.norm(v)) for v in rng.normal(size=(n_steps, 64))]
    ports_rows = _make_ports_rows(rng, n_steps)

    pos_sha = _put_blob(client, positions, dim=64)
    ports_sha = _put_blob(client, ports_rows, dim=38)
    causal_sha, causal_n_steps = _make_causal_log_blob(client, rng, n_steps)

    space_resp = client.post("/v3/space-id", json=SPACE)
    assert space_resp.status_code == 200, space_resp.text
    computed_space_id = space_id or space_resp.json()["space_id"]

    payload = {
        "schema": "hekb.experience/2",
        "raw_input": f"trajectory seed={seed}",
        "resolved_meaning": None, "action": None, "title": f"traj-{seed}", "tags": ["trj"],
        "ext": {"hekbv3": {
            "kind": "trajectory", "origin": "experience",
            "space": {**SPACE, "space_id": computed_space_id},
            "trajectory": {"blob_sha256": pos_sha, "n_steps": n_steps, "dim": 64},
            "ports": {"blob_sha256": ports_sha, "n_steps": n_steps, "dim": 38, "port_ids": list(ALL_38)},
            "causal_log": {"blob_sha256": causal_sha, "n_steps": causal_n_steps},
            "labels": {"category": category, "split": "dev", "group": group},
            "outcome": {"result": "n/a"},
        }},
    }
    sig = client.sign(payload)
    resp = client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": sig})
    return resp, computed_space_id


# [HEKBV3-BUILD002] BoardRecord's 32 non-excluded cells (nvs_edge/board.py CORE_IDS):
# ALL_38 minus these 6 ports.
_BOARDRECORD_EXCLUDED = ("P05", "P11", "P12", "P17", "P34", "P37")
CORE_32 = tuple(p for p in ALL_38 if p not in _BOARDRECORD_EXCLUDED)
QUANT_STEP_LOG2 = -30


def _post_trajectory_no_position(client: TestClient, *, seed: int, category: str, group: str,
                                  space_id: str | None = None, n_steps: int = 20):
    """[HEKBV3-BUILD002] Like `_post_trajectory`, but with no ext.hekbv3.trajectory (position)
    field, and a ports blob holding BoardRecord-style already-quantized int64 values over the
    32 non-excluded cells (dtype=DTYPE_INT64, quant_step_log2 recorded on the ref)."""
    from storage.trajectory_blob import DTYPE_INT64, encode_trajectory_blob

    rng = np.random.default_rng(seed)
    scale = 2.0 ** -QUANT_STEP_LOG2
    ports_rows = [[round(v * scale) for v in row] for row in rng.normal(size=(n_steps, 32)).tolist()]

    file_bytes, _header = encode_trajectory_blob(ports_rows, dimensions=32, dtype=DTYPE_INT64)
    ports_sha = hashlib.sha256(file_bytes).hexdigest()
    resp = client.put(f"/v3/blobs/{ports_sha}", content=file_bytes, headers={"Content-Type": "application/octet-stream"})
    assert resp.status_code == 201, resp.text
    causal_sha, causal_n_steps = _make_causal_log_blob(client, rng, n_steps)

    space_resp = client.post("/v3/space-id", json=SPACE)
    assert space_resp.status_code == 200, space_resp.text
    computed_space_id = space_id or space_resp.json()["space_id"]

    payload = {
        "schema": "hekb.experience/2",
        "raw_input": f"boardrecord seed={seed}",
        "resolved_meaning": None, "action": None, "title": f"br-{seed}", "tags": ["trj"],
        "ext": {"hekbv3": {
            "kind": "trajectory", "origin": "experience",
            "space": {**SPACE, "space_id": computed_space_id},
            "ports": {"blob_sha256": ports_sha, "n_steps": n_steps, "dim": 32,
                      "port_ids": list(CORE_32), "quant_step_log2": QUANT_STEP_LOG2},
            "causal_log": {"blob_sha256": causal_sha, "n_steps": causal_n_steps},
            "labels": {"category": category, "split": "dev", "group": group},
            "outcome": {"result": "n/a"},
        }},
    }
    sig = client.sign(payload)
    resp = client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": sig})
    return resp, computed_space_id


def test_position_less_boardrecord_compose_returns_r3_only(client: TestClient):
    space_id = None
    for i in range(6):
        resp, space_id = _post_trajectory_no_position(client, seed=i, category=f"cat{i % 2}", group=f"g{i}")
        assert resp.status_code == 201, resp.text

    build_resp = client.post(f"/v3/index/{space_id}/build", json={"L": 8})
    assert build_resp.status_code == 202, build_resp.text
    assert build_resp.json()["n_entries"] > 0

    resp0, _ = _post_trajectory_no_position(client, seed=0, category="cat0", group="g0")
    query_object_id = resp0.json()["object_id"]

    compose_resp = client.post("/v3/compose", json={
        "space_id": space_id, "query_object_id": query_object_id, "query_token_index": 10,
        "top_m": 64, "exclude": {"groups": ["g0"]},
    })
    assert compose_resp.status_code == 200, compose_resp.text
    body = compose_resp.json()
    assert abs(sum(body["weights"]) - 1.0) < 1e-9
    assert body["category_posterior"]
    assert set(body["value_by_horizon"].keys()) == {"1", "4", "8"}
    assert all(v is None for v in body["value_by_horizon"].values()), \
        "position-less Trajectory must return R3 only (no R1 value_by_horizon)"


def test_end_to_end_store_build_compose(client: TestClient):
    space_id = None
    for i in range(6):
        resp, space_id = _post_trajectory(client, seed=i, category=f"cat{i % 2}", group=f"g{i}")
        assert resp.status_code == 201, resp.text

    build_resp = client.post(f"/v3/index/{space_id}/build", json={"L": 8})
    assert build_resp.status_code == 202, build_resp.text
    build_body = build_resp.json()
    assert build_body["n_entries"] > 0

    # reuse one of the posted experiences as the query, excluding its own group
    resp0, _ = _post_trajectory(client, seed=0, category="cat0", group="g0")  # idempotent repost -> same object_id
    query_object_id = resp0.json()["object_id"]

    compose_resp = client.post("/v3/compose", json={
        "space_id": space_id, "query_object_id": query_object_id, "query_token_index": 10,
        "top_m": 64, "exclude": {"groups": ["g0"]},
    })
    assert compose_resp.status_code == 200, compose_resp.text
    body = compose_resp.json()
    assert abs(sum(body["weights"]) - 1.0) < 1e-9
    assert all(w >= 0 for w in body["weights"])
    assert body["category_posterior"]
    assert set(body["value_by_horizon"].keys()) == {"1", "4", "8"}


def test_k5_cross_space_compose_is_409(client: TestClient):
    _, space_id_a = _post_trajectory(client, seed=10, category="catA", group="gA")
    client.post(f"/v3/index/{space_id_a}/build", json={"L": 8})

    # a second trajectory in a DIFFERENT space (different model_ref)
    other_space = dict(SPACE, model_ref="other-model")
    rng = np.random.default_rng(11)
    positions = [list(v / np.linalg.norm(v)) for v in rng.normal(size=(20, 64))]
    ports_rows = _make_ports_rows(rng, 20)
    pos_sha = _put_blob(client, positions, dim=64)
    ports_sha = _put_blob(client, ports_rows, dim=38)
    causal_sha, causal_n_steps = _make_causal_log_blob(client, rng, 20)
    space_id_b = client.post("/v3/space-id", json=other_space).json()["space_id"]
    payload = {
        "schema": "hekb.experience/2", "raw_input": "other-space", "resolved_meaning": None,
        "action": None, "title": "other", "tags": [],
        "ext": {"hekbv3": {
            "kind": "trajectory", "origin": "experience",
            "space": {**other_space, "space_id": space_id_b},
            "trajectory": {"blob_sha256": pos_sha, "n_steps": 20, "dim": 64},
            "ports": {"blob_sha256": ports_sha, "n_steps": 20, "dim": 38, "port_ids": list(ALL_38)},
            "causal_log": {"blob_sha256": causal_sha, "n_steps": causal_n_steps},
            "labels": {"category": "catB", "split": "dev", "group": "gB"},
            "outcome": {"result": "n/a"},
        }},
    }
    sig = client.sign(payload)
    resp_b = client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": sig})
    assert resp_b.status_code == 201, resp_b.text
    query_object_id_b = resp_b.json()["object_id"]

    # querying space A's index with an object that lives in space B must be 409
    compose_resp = client.post("/v3/compose", json={
        "space_id": space_id_a, "query_object_id": query_object_id_b, "query_token_index": 10,
    })
    assert compose_resp.status_code == 409, compose_resp.text


def test_k7_compose_has_no_side_effects(client: TestClient, tmp_path):
    space_id = None
    for i in range(5):
        resp, space_id = _post_trajectory(client, seed=20 + i, category=f"cat{i % 2}", group=f"g{i}")
        assert resp.status_code == 201
    client.post(f"/v3/index/{space_id}/build", json={"L": 8})
    resp0, _ = _post_trajectory(client, seed=20, category="cat0", group="g0")
    query_object_id = resp0.json()["object_id"]

    def _snapshot() -> dict[str, str]:
        data_dir = tmp_path / "experience"
        snap = {}
        for sub in ("objects", "relations", "lineages", "meta", "blobs"):
            d = data_dir / sub
            if not d.exists():
                continue
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    snap[str(p.relative_to(data_dir))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return snap

    before = _snapshot()
    compose_resp = client.post("/v3/compose", json={
        "space_id": space_id, "query_object_id": query_object_id, "query_token_index": 10,
        "exclude": {"groups": ["g0"]},
    })
    assert compose_resp.status_code == 200, compose_resp.text
    after = _snapshot()
    assert before == after, "compose() must not mutate objects/relations/lineages/meta/blobs (K7, H7)"


def test_k8_no_floats_in_payload(client: TestClient):
    resp, _ = _post_trajectory(client, seed=30, category="catX", group="gX")
    assert resp.status_code == 201, resp.text
    payload = resp.json()["payload"]

    def _assert_no_float(value, path="payload"):
        if isinstance(value, float):
            raise AssertionError(f"float found at {path}: {value!r} (H6 violation)")
        if isinstance(value, dict):
            for k, v in value.items():
                _assert_no_float(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _assert_no_float(v, f"{path}[{i}]")

    _assert_no_float(payload)
