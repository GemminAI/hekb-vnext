"""hekb-vnext Phase B-③: POST /morphisms + GET /graph/geodesic.

Explicit Morphism creation between *existing* objects (unlike branch(),
which always creates a new one), and weighted shortest-path search --
ported from `GemminAI/hekb`'s `POST /v1/morphisms` / `GET /v1/query/
geodesic` (`cpp/src/query/query.cpp::geodesic()`, real Dijkstra).
`weight` lives in `label` by convention -- `core/morphism.py` itself is
never modified (see store_service.py's `_morphism_weight()` docstring).

Same fixture pattern as `test_query_nearest.py` / `test_experience_closure.py`.
"""

from __future__ import annotations

import base64
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


def _post_experience(client: TestClient, *, title: str) -> dict:
    payload = {
        "schema": "hekb.experience/1",
        "raw_input": title,
        "resolved_meaning": None,
        "action": None,
        "title": title,
        "tags": [],
        "ext": {},
    }
    signature = client.sign(payload)  # type: ignore[attr-defined]
    response = client.post(
        "/experience",
        json={"payload": payload, "parent_id": None},
        headers={"X-Audit-Signature": signature},
    )
    response.raise_for_status()
    return response.json()


def _post_morphism(client: TestClient, *, source_id: str, target_id: str, weight: float = 1.0) -> dict:
    response = client.post(
        "/morphisms", json={"source_id": source_id, "target_id": target_id, "weight": weight}
    )
    response.raise_for_status()
    return response.json()


def test_create_morphism_between_two_existing_objects(client: TestClient):
    a = _post_experience(client, title="a")
    b = _post_experience(client, title="b")

    morphism = _post_morphism(client, source_id=a["object_id"], target_id=b["object_id"], weight=2.5)

    assert morphism["source_id"] == a["object_id"]
    assert morphism["target_id"] == b["object_id"]
    assert morphism["weight"] == 2.5


def test_create_morphism_rejects_unknown_source(client: TestClient):
    b = _post_experience(client, title="b")
    response = client.post("/morphisms", json={"source_id": "does-not-exist", "target_id": b["object_id"]})
    assert response.status_code == 404


def test_create_morphism_rejects_unknown_target(client: TestClient):
    a = _post_experience(client, title="a")
    response = client.post("/morphisms", json={"source_id": a["object_id"], "target_id": "does-not-exist"})
    assert response.status_code == 404


def test_geodesic_hop_count_and_weight_minimum_disagree(client: TestClient):
    """The deliberately adversarial case the task specified: a 1-hop
    direct edge with a high weight vs. a 2-hop detour whose total weight
    is lower. The correct geodesic must take the detour, proving this is
    real weighted Dijkstra, not a hop-count shortest path standing in for
    it."""
    a = _post_experience(client, title="a")
    b = _post_experience(client, title="b")
    c = _post_experience(client, title="c")

    # Direct, expensive edge: a -> c, weight 10.
    _post_morphism(client, source_id=a["object_id"], target_id=c["object_id"], weight=10.0)
    # Detour, cheap: a -> b -> c, weight 1 + 1 = 2.
    _post_morphism(client, source_id=a["object_id"], target_id=b["object_id"], weight=1.0)
    _post_morphism(client, source_id=b["object_id"], target_id=c["object_id"], weight=1.0)

    response = client.get(
        "/graph/geodesic", params={"source_id": a["object_id"], "target_id": c["object_id"]}
    )
    response.raise_for_status()
    body = response.json()

    assert body["path"] == [a["object_id"], b["object_id"], c["object_id"]]
    assert body["total_weight"] == pytest.approx(2.0)
    assert len(body["morphisms"]) == 2


def test_geodesic_defaults_missing_weight_to_one(client: TestClient):
    """Mixes an explicit-weight morphism with a default-weight (branch())
    edge in the same path, and checks the total accounts for both
    correctly."""
    root = _post_experience(client, title="root")

    # branch()-derived edge, no "weight" key at all in its label -> defaults to 1.0.
    payload = {
        "schema": "hekb.experience/1",
        "raw_input": "child",
        "resolved_meaning": None,
        "action": None,
        "title": "child",
        "tags": [],
        "ext": {},
    }
    signature = client.sign(payload)  # type: ignore[attr-defined]
    response = client.post(
        "/experience",
        json={"payload": payload, "parent_id": root["object_id"]},
        headers={"X-Audit-Signature": signature},
    )
    response.raise_for_status()
    child = response.json()

    leaf = _post_experience(client, title="leaf")
    _post_morphism(client, source_id=child["object_id"], target_id=leaf["object_id"], weight=3.0)

    response = client.get(
        "/graph/geodesic", params={"source_id": root["object_id"], "target_id": leaf["object_id"]}
    )
    response.raise_for_status()
    body = response.json()

    assert body["path"] == [root["object_id"], child["object_id"], leaf["object_id"]]
    assert body["total_weight"] == pytest.approx(1.0 + 3.0)


def test_geodesic_no_path_returns_404(client: TestClient):
    a = _post_experience(client, title="a")
    b = _post_experience(client, title="b")  # no edge between them

    response = client.get("/graph/geodesic", params={"source_id": a["object_id"], "target_id": b["object_id"]})
    assert response.status_code == 404


def test_geodesic_unknown_object_id_returns_404(client: TestClient):
    a = _post_experience(client, title="a")
    response = client.get("/graph/geodesic", params={"source_id": a["object_id"], "target_id": "does-not-exist"})
    assert response.status_code == 404


def test_morphism_persists_on_disk_and_is_replayed_by_build_category(client: TestClient, tmp_path):
    """Durability check (SensOS-HEKB-Integration-PhaseA-Design-20260918.md
    §3's requirement): this service has no in-memory Category cache --
    `_build_category()` re-reads `objects/` and `relations/` from disk on
    every single call, with no module-level state carried between
    requests. So the real thing to verify is that the explicit morphism
    actually landed in `relations/<morphism_id>.json` on disk (not held
    only in a Python object that a real process restart would lose), and
    that `_build_category()` -- called completely independently of the
    request that created it -- reconstructs it correctly from that file
    alone."""
    a = _post_experience(client, title="a")
    b = _post_experience(client, title="b")
    morphism = _post_morphism(client, source_id=a["object_id"], target_id=b["object_id"], weight=4.0)

    relation_path = tmp_path / "experience" / "relations" / f"{morphism['morphism_id']}.json"
    assert relation_path.exists()

    import store_service

    category = store_service._build_category()
    # Look up by the real stored payload, not a guessed one -- a's payload
    # is whatever create_experience actually persisted.
    a_record = store_service._read_json(store_service._object_path(a["object_id"]))
    a_obj = store_service.MemoryObject(payload=a_record["payload"])
    edges = category.morphisms_from(a_obj)

    assert any(e.cod.object_id == b["object_id"] for e in edges)
