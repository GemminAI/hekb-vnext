"""hekb-vnext Phase B-②: GET /experience/{object_id}/closure.

Semantic Closure Query, ported from `GemminAI/hekb`'s EXP-HEKB002/003
(`index/semantic_closure.py`). Not vector search: given a query object,
walks the real parent->child morphism graph outward (pullback/premises)
and inward (pushout/impact) that `_build_category()` already reconstructs
for `/experience/recall`.

Each test gets its own fresh TestClient bound to a throwaway data
directory, same fixture pattern as `test_query_nearest.py`.
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


def _post_experience(client: TestClient, *, title: str, parent_id: str | None = None) -> dict:
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
        json={"payload": payload, "parent_id": parent_id},
        headers={"X-Audit-Signature": signature},
    )
    response.raise_for_status()
    return response.json()


def test_isolated_object_has_a_trivial_closure(client: TestClient):
    solo = _post_experience(client, title="solo")

    response = client.get(f"/experience/{solo['object_id']}/closure")
    response.raise_for_status()
    body = response.json()

    assert body["query_object_id"] == solo["object_id"]
    assert [o["id"] for o in body["objects"]] == [solo["object_id"]]
    assert body["morphisms"] == []
    assert body["pullback_roots"] == []
    assert body["pushout_wavefront"] == []
    assert body["is_minimal_self_contained"] is True


def test_linear_chain_closure_from_the_middle_node(client: TestClient):
    """root -> middle -> leaf. Querying from `middle` must see both
    directions: `root` via pushout (middle depends on root) and `leaf` via
    pullback (leaf depends on middle) -- proving this is bidirectional
    closure, not a one-directional ancestor/descendant walk alone."""
    root = _post_experience(client, title="root")
    middle = _post_experience(client, title="middle", parent_id=root["object_id"])
    leaf = _post_experience(client, title="leaf", parent_id=middle["object_id"])

    response = client.get(f"/experience/{middle['object_id']}/closure")
    response.raise_for_status()
    body = response.json()

    object_ids = {o["id"] for o in body["objects"]}
    assert object_ids == {root["object_id"], middle["object_id"], leaf["object_id"]}
    assert len(body["morphisms"]) == 2
    assert all(m["kind"] == "experience" for m in body["morphisms"])
    assert body["is_minimal_self_contained"] is True
    # leaf has no outgoing edge of its own -> pullback root from middle's
    # own outgoing direction; root has no incoming edge -> pushout wavefront.
    assert leaf["object_id"] in body["pullback_roots"]
    assert root["object_id"] in body["pushout_wavefront"]


def test_derived_composition_appears_for_a_two_hop_pullback_path(client: TestClient):
    root = _post_experience(client, title="root")
    middle = _post_experience(client, title="middle", parent_id=root["object_id"])
    leaf = _post_experience(client, title="leaf", parent_id=middle["object_id"])

    response = client.get(f"/experience/{root['object_id']}/closure")
    response.raise_for_status()
    body = response.json()

    # root -> middle -> leaf is a 2-hop pullback path from root; exactly one
    # derived composition (root -> leaf) should be folded via compose().
    assert len(body["derived_compositions"]) == 1
    derived = body["derived_compositions"][0]
    assert derived["morphism"] == f"m_derived: {root['object_id']} -> {leaf['object_id']}"
    assert " o " in derived["formula"]


def test_branching_produces_multiple_pullback_roots(client: TestClient):
    root = _post_experience(client, title="root")
    child_a = _post_experience(client, title="child-a", parent_id=root["object_id"])
    child_b = _post_experience(client, title="child-b", parent_id=root["object_id"])

    response = client.get(f"/experience/{root['object_id']}/closure")
    response.raise_for_status()
    body = response.json()

    assert set(body["pullback_roots"]) == {child_a["object_id"], child_b["object_id"]}
    assert body["is_minimal_self_contained"] is True


def test_unknown_object_id_returns_404(client: TestClient):
    response = client.get("/experience/does-not-exist/closure")
    assert response.status_code == 404
