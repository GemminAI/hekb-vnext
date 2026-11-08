import copy
import hashlib
import json
from pathlib import Path

import pytest

from graph.chainlog import ChainLog, ChainLogError
from graph.ic04 import IC04Error, IC04Store
from vendor.sensos_ic00 import ids

SES = "ses:" + "1" * 64


def seal(env: dict, lookup: dict | None = None) -> dict:
    env["content_id"] = ids.content_id(env, lookup or {})
    env["artifact_id"] = ids.artifact_id(env, lookup or {})
    return env


def make(kind="observation", payload=None, parents=(), seq=1, producer="sensos-inference", obs="obs:" + "2" * 64, version="1.0.0",
         space_id="hid:" + "3" * 64, lookup=None, **kw) -> dict:
    env = {"contract": {"id": "IC-00", "version": version}, "content_id": "", "artifact_id": "", "kind": kind,
           "payload_schema": {"id": "x.test", "version": "0"}, "payload": payload if payload is not None else {"a": 1},
           "producer": {"name": producer, "version": "0", "build": "t"}, "session_id": SES, "observation_id": obs, "event_span": None,
           "space_id": space_id, "parents": [{"role": r, "ordered": False, "artifact_ids": list(i)} for r, i in parents], "policy_refs": [],
           "calibration_ref": None, "rule_refs": [], "observed_at": None, "seq": seq, "produced_at": "2026-11-08T00:00:00Z", "extensions": {}}
    env.update(kw)
    return seal(env, lookup)


def lk(*envs):
    return {e["artifact_id"]: {"content_id": e["content_id"]} for e in envs}


@pytest.fixture
def store(tmp_path: Path) -> IC04Store:
    return IC04Store(tmp_path / "d")


def test_put_get_lineage_ids_recompute_and_null_kept(store: IC04Store) -> None:
    root = make(payload={"v": None, "z": 0, "u": {"undefined": "int64_min"}, "n": [None, 0]}, seq=1)
    mid = make(kind="interpretation", producer="meaning-space-runtime", parents=[("t", [root["artifact_id"]])], lookup=lk(root), seq=2)
    top = make(kind="interpretation", producer="meaning-space-runtime", parents=[("m", [mid["artifact_id"]])], lookup=lk(mid), seq=3)
    for e in (root, mid, top):
        r = store.put(e, {})
        assert r["artifact_id"] == e["artifact_id"] and not r["already_stored"]
    got = store.get(top["artifact_id"])["envelope"]
    assert got == top
    back = store.get(root["artifact_id"])["envelope"]
    assert back == root and back["payload"]["v"] is None and back["payload"]["z"] == 0 and back["payload"]["u"] == {"undefined": "int64_min"}
    assert ids.content_id(back, {}) == back["content_id"] and ids.artifact_id(back, {}) == back["artifact_id"]
    lin = store.lineage(top["artifact_id"])
    assert [e["artifact_id"] for e in lin["entries"]] == [top["artifact_id"], mid["artifact_id"], root["artifact_id"]]
    assert [e["depth"] for e in lin["entries"]] == [0, 1, 2] and lin["missing"] == []
    assert store.get("art:" + "f" * 64) is None


def test_returned_envelope_is_a_copy(store: IC04Store) -> None:
    e = make()
    store.put(e, {})
    g = store.get(e["artifact_id"])
    g["envelope"]["payload"]["a"] = 99
    assert store.get(e["artifact_id"])["envelope"]["payload"]["a"] == 1


def test_reput_same_envelope_keeps_count_and_identity(store: IC04Store) -> None:
    e = make()
    r1 = store.put(e, {})
    n = len(store)
    r2 = store.put(copy.deepcopy(e), {})
    assert len(store) == n and len(store.log) == 1
    assert (r2["hekb_seq"], r2["hekb_hash"], r2["artifact_id"]) == (r1["hekb_seq"], r1["hekb_hash"], r1["artifact_id"]) and r2["already_stored"]


def test_same_artifact_id_with_different_body_is_a_conflict(store: IC04Store) -> None:
    e = make()
    store.put(e, {})
    e2 = copy.deepcopy(e)
    e2["seq"] = 7   # seq is outside the id preimage: ids still recompute, the body differs
    assert ids.artifact_id(e2, {}) == e2["artifact_id"]
    with pytest.raises(IC04Error) as ex:
        store.put(e2, {})
    assert ex.value.code == "conflict" and len(store) == 1


def _rejects(store: IC04Store, env: dict, code: str, blobs: dict | None = None) -> None:
    n, nl = len(store), len(store.log)
    with pytest.raises(IC04Error) as ex:
        store.put(env, blobs or {})
    assert ex.value.code == code
    assert len(store) == n and len(store.log) == nl and not store.has(env.get("artifact_id", "")) or code == "conflict"


def test_rejections_are_not_stored(store: IC04Store) -> None:
    root = make()
    store.put(root, {})
    child = make(kind="interpretation", producer="meaning-space-runtime", parents=[("t", [root["artifact_id"]])], lookup=lk(root), seq=2)
    t = copy.deepcopy(child); t["payload"]["a"] = 2
    _rejects(store, t, "id_mismatch")
    t = copy.deepcopy(child); t["artifact_id"] = "art:" + "e" * 64
    _rejects(store, t, "id_mismatch")
    t = copy.deepcopy(child); t["parents"][0]["artifact_ids"] = ["art:" + "ab" * 32]
    _rejects(store, t, "parent_unknown")
    t = copy.deepcopy(child); t["contract"]["version"] = "2.0.0"
    _rejects(store, t, "unsupported_contract_version")
    t = copy.deepcopy(child); del t["producer"]
    _rejects(store, t, "malformed")
    t = copy.deepcopy(child); t["payload"]["a"] = 1.5      # a float cannot be canonicalised
    _rejects(store, t, "id_uncomputable")
    _rejects(store, make(seq=5, payload={"b": 1}), "blob_mismatch", {"blob:sha256:" + "0" * 64: b"x"})
    assert len(store) == 1
    assert store.get(child["artifact_id"]) is None


def test_accepted_versions_are_a_parameter(tmp_path: Path) -> None:
    e11 = make(version="1.1.0")
    with pytest.raises(IC04Error):
        IC04Store(tmp_path / "a").put(e11, {})
    s = IC04Store(tmp_path / "b", accepted_versions=("1.0.0", "1.1.0"))
    s.put(e11, {})
    s.put(make(version="1.0.0", payload={"q": 1}), {})
    assert len(s) == 2


def test_hold_policy_keeps_unknown_parent_envelopes_out_of_recall(tmp_path: Path) -> None:
    s = IC04Store(tmp_path / "d", missing_parents="hold")
    ghost = make(seq=1)
    child = make(kind="interpretation", producer="meaning-space-runtime", parents=[("t", [ghost["artifact_id"]])], lookup=lk(ghost), seq=2)
    r = s.put(child, {})
    assert r["held"] == [ghost["artifact_id"]] and len(s) == 0 and s.get(child["artifact_id"]) is None
    assert s.select({"kind": "interpretation"}, 10)[0] == []
    assert len(s.pending) == 1


def ctx(seq: int) -> dict:
    return {"session_id": SES, "observation_id": "obs:" + "9" * 64, "seq": seq, "produced_at": "2026-11-08T00:00:00Z"}


def test_recall_stores_event_first_and_references_returned_ids(store: IC04Store) -> None:
    es = [make(payload={"i": i, "prompt": "P%d" % (i % 2)}, seq=i + 1) for i in range(4)]
    for e in es:
        store.put(e, {})
    seen_at_admit = []

    def admit(env, lookup):   # the caller's Gate: runs BEFORE the event is stored
        seen_at_admit.append(store.has(env["artifact_id"]))

    out = store.recall({"space_id": es[0]["space_id"], "kind": "observation", "exclude_where": [{"pointer": "/payload/prompt", "equals": "P0"}]},
                       requester="nvs-runtime", context=ctx(10), limit=5, admit=admit)
    ev = out["event"]
    assert seen_at_admit == [False] and store.has(ev["artifact_id"]) and out["event_receipt"]["hekb_seq"] == 4
    assert [r["artifact_id"] for r in out["results"]] == [es[3]["artifact_id"], es[1]["artifact_id"]]       # seq_desc, P0 excluded
    assert [x["artifact_id"] for x in ev["payload"]["results"]] == [r["artifact_id"] for r in out["results"]]
    assert ev["producer"]["name"] == "hekb-vnext" and ev["kind"] == "observation" and store.get(ev["artifact_id"])["envelope"] == ev
    assert out["results"][0]["envelope"] == es[3]
    # chain order: the event was written after what it recalls and BEFORE the caller can use it (it is already stored on return)
    assert store.log.verify is not None and list(store.log.verify())[-1]["record"]["artifact_id"] == ev["artifact_id"]


def test_recall_refused_by_admit_stores_nothing(store: IC04Store) -> None:
    store.put(make(), {})

    def admit(env, lookup):
        raise ValueError("gate says no")

    with pytest.raises(IC04Error) as ex:
        store.recall({"kind": "observation"}, requester="r", context=ctx(5), admit=admit)
    assert ex.value.code == "admit_rejected" and len(store) == 1


def test_recall_query_forms(store: IC04Store) -> None:
    a, b = make(payload={"n": 1}, seq=1), make(payload={"n": 2}, seq=2)
    store.put(a, {}); store.put(b, {})
    assert store.select({"artifact_ids": [a["artifact_id"]]}, 5)[0] == [a["artifact_id"]]
    assert store.select({"exclude_artifact_ids": [b["artifact_id"]]}, 5)[0] == [a["artifact_id"]]
    assert store.select({"ranking": "seq_asc"}, 5)[0] == [a["artifact_id"], b["artifact_id"]]
    assert store.select({"where": [{"pointer": "/payload/n", "equals": 1}]}, 5)[0] == [a["artifact_id"]]
    assert store.select({"where": [{"pointer": "/payload/n", "equals": True}]}, 5)[0] == []      # 1 is not true
    assert store.select({}, 1) == ([b["artifact_id"]], True)
    for bad in ({"nearest_to": "x"}, {"ranking": "nearest"}):
        with pytest.raises(IC04Error):
            store.select(bad, 5)


def test_chain_verifies_and_a_new_handle_reads_the_same(tmp_path: Path) -> None:
    s = IC04Store(tmp_path / "d")
    root = make(payload={"v": None}, seq=1)
    blob = b"abc" * 10
    withblob = make(payload={"b": "x"}, seq=2, parents=[("p", [root["artifact_id"]])], lookup=lk(root))
    s.put(root, {}); s.put(withblob, {"blob:sha256:" + hashlib.sha256(blob).hexdigest(): blob})
    s.recall({"kind": "observation"}, requester="r", context=ctx(3))
    lines = s.verify_all()
    h = IC04Store(tmp_path / "d")
    assert len(h) == len(s) == 3 and set(h.verify_all()) == set(lines)
    for aid in lines:
        assert h.get(aid)["envelope"] == s.get(aid)["envelope"]
    assert h.get(withblob["artifact_id"])["blobs"] == {"blob:sha256:" + hashlib.sha256(blob).hexdigest(): blob}
    path = tmp_path / "d/graph/ic00_envelopes.jsonl"
    path.write_text(path.read_text().replace('"a":1', '"a":2', 1) if '"a":1' in path.read_text() else path.read_text().replace('"v":null', '"v":0', 1))
    with pytest.raises(ChainLogError):
        IC04Store(tmp_path / "d")


def test_vendored_ids_match_their_lock() -> None:
    d = Path(__file__).resolve().parents[1] / "vendor" / "sensos_ic00"
    lock = json.loads((d / "LOCK.json").read_text())
    for name, h in lock["files"].items():
        assert hashlib.sha256((d / name).read_text().encode()).hexdigest() == h["vendored_sha256"], name
