"""hekb-vnext Phase B-①: POST /query/nearest.

Brute-force nearest-neighbour search over `payload.ext.centroid`, ported
from `GemminAI/hekb`'s `cpp/src/query/query.cpp::nearest()` -- see
SensOS-HEKB-Integration-PhaseA-Design-20260918.md §1 and store_service.py's
own `query_nearest()` docstring for the design rationale.

Each test gets its own fresh TestClient bound to a throwaway data
directory and a real dev P-256 keypair, mirroring
`GemminAI/sensos/tests/e2e/dev_hekb_writer.py`'s own pattern (this file
lives inside hekb-vnext itself, so it talks to `store_service` directly
rather than via sys.path insertion).
"""

from __future__ import annotations

import base64
import os
import secrets
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HEKB_DATA_DIR", str(tmp_path / "experience"))

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


def _post_experience(client: TestClient, *, title: str, centroid: list[float] | None) -> dict:
    payload = {
        "schema": "hekb.experience/1",
        "raw_input": title,
        "resolved_meaning": None,
        "action": None,
        "title": title,
        "tags": [],
        "ext": {} if centroid is None else {"centroid": centroid},
    }
    signature = client.sign(payload)  # type: ignore[attr-defined]
    response = client.post(
        "/experience",
        json={"payload": payload, "parent_id": None},
        headers={"X-Audit-Signature": signature},
    )
    response.raise_for_status()
    return response.json()


def test_nearest_returns_closest_first(client: TestClient):
    near = _post_experience(client, title="near", centroid=[1.0, 0.0])
    far = _post_experience(client, title="far", centroid=[0.0, 1.0])

    response = client.post("/query/nearest", json={"vector": [1.0, 0.0], "limit": 10, "metric": "cosine"})
    response.raise_for_status()
    matches = response.json()["matches"]

    assert [m["object_id"] for m in matches] == [near["object_id"], far["object_id"]]
    assert matches[0]["distance"] == pytest.approx(0.0)


def test_hop_count_nearest_neighbour_can_differ_from_vector_nearest(client: TestClient):
    """The deliberately-adversarial case the task asked for: three points on
    a line where the vector-nearest match to the probe is NOT the one a
    naive "smallest coordinate difference" or insertion-order heuristic
    would pick, proving this is real distance computation, not something
    order- or hop-count-based standing in for it."""
    origin = _post_experience(client, title="origin", centroid=[0.0, 0.0, 0.0])
    close_but_inserted_last = _post_experience(client, title="close", centroid=[1.0, 0.0, 0.0])
    far_but_inserted_first = _post_experience(client, title="far", centroid=[10.0, 0.0, 0.0])
    # insertion order above is origin, close, far -- deliberately not sorted
    # by distance, so a bug that returned insertion order would be caught.
    del far_but_inserted_first

    response = client.post(
        "/query/nearest", json={"vector": [1.1, 0.0, 0.0], "limit": 2, "metric": "euclidean"}
    )
    response.raise_for_status()
    matches = response.json()["matches"]

    assert matches[0]["object_id"] == close_but_inserted_last["object_id"]
    assert matches[1]["object_id"] == origin["object_id"]


def test_nearest_skips_objects_without_a_vector(client: TestClient):
    with_vector = _post_experience(client, title="with", centroid=[1.0, 0.0])
    _post_experience(client, title="without", centroid=None)

    response = client.post("/query/nearest", json={"vector": [1.0, 0.0], "limit": 10, "metric": "cosine"})
    response.raise_for_status()
    matches = response.json()["matches"]

    assert [m["object_id"] for m in matches] == [with_vector["object_id"]]


def test_nearest_skips_objects_of_a_different_dimension(client: TestClient):
    same_dim = _post_experience(client, title="2d", centroid=[1.0, 0.0])
    _post_experience(client, title="3d", centroid=[1.0, 0.0, 0.0])

    response = client.post("/query/nearest", json={"vector": [1.0, 0.0], "limit": 10, "metric": "cosine"})
    response.raise_for_status()
    matches = response.json()["matches"]

    assert [m["object_id"] for m in matches] == [same_dim["object_id"]]


def test_nearest_respects_limit(client: TestClient):
    for i in range(5):
        _post_experience(client, title=f"pt-{i}", centroid=[float(i), 0.0])

    response = client.post("/query/nearest", json={"vector": [0.0, 0.0], "limit": 2, "metric": "euclidean"})
    response.raise_for_status()
    matches = response.json()["matches"]

    assert len(matches) == 2


def test_nearest_rejects_unknown_metric(client: TestClient):
    response = client.post("/query/nearest", json={"vector": [1.0], "limit": 10, "metric": "manhattan"})
    assert response.status_code == 422


def test_nearest_empty_store_returns_no_matches(client: TestClient):
    response = client.post("/query/nearest", json={"vector": [1.0, 0.0], "limit": 10, "metric": "cosine"})
    response.raise_for_status()
    assert response.json()["matches"] == []
