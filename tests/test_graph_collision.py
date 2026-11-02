# ruff: noqa: E501
"""D13 / D11: the same ``node_id`` written with a different record is a collision, not a silent drop.

Both entrances -- ``GraphStore`` and the HEXT projection layer -- write through ``graph.guard.put``
under one lock on ``<data_dir>/graph``. Tests that need a HEXT document use the P0 example documents of
GemminAI/hext (test data), like tests/test_hext_projection.py, and are skipped when they are absent.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from graph import hext_projection
from graph.chainlog import ChainLog
from graph.guard import CollisionError, GraphLock, Ledger, edge_key, node_key, put, record_hash
from graph.hext_projection import HextProjectionLayer, doc_node_id
from graph.model import GraphError
from graph.store import GraphStore

ROOT = Path(__file__).resolve().parent.parent
EX = Path.home() / "GemminAI" / "hext" / "examples" / "v2"
SRC = EX / "source-three-sentences.hext2.json"
needs_hext_docs = pytest.mark.skipif(not SRC.exists(), reason="P0 example documents absent")


def _doc_id() -> str:
    return doc_node_id(json.loads(SRC.read_bytes())["final_seal"]["state_hash"])


def _lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0


# ---- GraphStore ---------------------------------------------------------------------------------------
def test_same_node_same_content_is_idempotent_and_writes_nothing(tmp_path):
    s = GraphStore(tmp_path)
    assert s.add_node("n", "trajectory", attrs={"a": 1}) is True
    head = s.nodes_log.head
    assert s.add_node("n", "trajectory", attrs={"a": 1}) is False
    assert (len(s.nodes_log), s.nodes_log.head, _lines(s.nodes_log.path)) == (1, head, 1)


def test_same_node_different_content_is_a_collision_and_the_first_record_stays(tmp_path):
    s = GraphStore(tmp_path)
    s.add_node("n", "trajectory", attrs={"a": 1})
    head = s.nodes_log.head
    with pytest.raises(CollisionError) as caught:
        s.add_node("n", "trajectory", attrs={"a": 2})
    e = caught.value
    first = {"node_id": "n", "node_type": "trajectory", "kind": "observation", "verification": "unverified", "attrs": {"a": 1}}
    second = {**first, "attrs": {"a": 2}}
    assert isinstance(e, GraphError)
    assert (e.node_id, e.existing_hash, e.new_hash) == ("n", record_hash(first), record_hash(second))
    assert e.existing_hash in str(e) and e.new_hash in str(e) and "'n'" in str(e)
    assert (len(s.nodes_log), s.nodes_log.head) == (1, head)
    assert s.nodes()[0]["attrs"] == {"a": 1}


def test_same_node_only_kind_differs_is_a_collision(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    s.add_node("n", "trajectory", kind="observation", verification="unverified")
    with pytest.raises(CollisionError):
        s.add_node("n", "trajectory", kind="evaluation", verification="unverified")
    with pytest.raises(CollisionError):  # and so does the verification column
        s.add_node("n", "trajectory", kind="observation", verification="machine_verified")
    assert len(s.nodes_log) == 1


def test_a_collision_is_found_by_a_new_store_on_the_same_dir(tmp_path):
    GraphStore(tmp_path).add_node("n", "trajectory", attrs={"a": 1})
    with pytest.raises(CollisionError):
        GraphStore(tmp_path).add_node("n", "trajectory", attrs={"a": 2})
    assert GraphStore(tmp_path).add_node("n", "trajectory", attrs={"a": 1}) is False


def test_a_node_another_store_object_wrote_meanwhile_is_seen(tmp_path):
    a, b = GraphStore(tmp_path), GraphStore(tmp_path)
    a.add_node("n", "trajectory", attrs={"a": 1})
    assert b.has_node("n")
    with pytest.raises(CollisionError):
        b.add_node("n", "trajectory", attrs={"a": 2})
    b.add_node("m", "trajectory")  # b continues a's chain instead of forking it
    assert [r["node_id"] for r in a.nodes_log.records()] == ["n", "m"]
    list(a.nodes_log.verify())


# ---- edges --------------------------------------------------------------------------------------------
def test_same_edge_again_is_an_idempotent_success(tmp_path):
    s = GraphStore(tmp_path)
    assert s.add_edge("measured_under", "a", "b", {"x": 1, "y": "z"}) is True
    assert s.add_edge("measured_under", "a", "b", {"y": "z", "x": 1}) is False
    assert s.add_edge("measured_under", "a", "b", {"x": 2, "y": "z"}) is True  # other attrs: another edge
    assert len(s.edges_log) == 2


@pytest.mark.parametrize("a", [{}, {"x": 1}, {"x": [1, 2]}, {"k": "é", "n": None}])
def test_an_edge_identity_determines_the_record(a):
    """The id material [type, src, dst, attrs] is the record itself, so one id cannot carry two contents."""
    r1 = {"type": "measured_under", "src": "a", "dst": "b", "attrs": dict(a)}
    r2 = {"type": "measured_under", "src": "a", "dst": "b", "attrs": dict(reversed(list(a.items())))}
    assert edge_key(r1) == edge_key(r2) and record_hash(r1) == record_hash(r2)
    no_attrs = {"type": "measured_under", "src": "a", "dst": "b"}
    assert edge_key(no_attrs) == edge_key({**no_attrs, "attrs": {}}) and record_hash(no_attrs) != record_hash({**no_attrs, "attrs": {}})


def test_the_edge_check_would_catch_a_record_that_does_not_follow_that_rule(tmp_path):
    """add_edge cannot produce one; a forged record with the same identity but another shape is refused."""
    log = ChainLog(tmp_path / "graph" / "edges.jsonl")
    ledger, lock = Ledger(log, edge_key, "edge"), GraphLock.of(tmp_path / "graph")
    assert put(lock, ledger, {"type": "t", "src": "a", "dst": "b", "attrs": {}}) is True
    with pytest.raises(CollisionError):
        put(lock, ledger, {"type": "t", "src": "a", "dst": "b", "attrs": {}, "extra": 1})
    assert len(log) == 1


# ---- HEXT projection layer ----------------------------------------------------------------------------
@needs_hext_docs
def test_projection_collides_with_a_node_the_store_wrote_under_the_same_id(tmp_path):
    GraphStore(tmp_path, v01=False).add_node(_doc_id(), "hext_document", attrs={"who": "store"})
    layer = HextProjectionLayer(tmp_path)
    with pytest.raises(CollisionError) as caught:
        layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    assert caught.value.node_id == _doc_id()
    assert not layer.root.exists()  # nothing was swapped in
    assert GraphStore(tmp_path, v01=False).nodes()[0]["attrs"] == {"who": "store"}


@needs_hext_docs
def test_store_collides_with_a_node_the_projection_wrote_under_the_same_id(tmp_path):
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    store = GraphStore(tmp_path, v01=False)
    with pytest.raises(CollisionError) as caught:
        store.add_node(_doc_id(), "hext_document", attrs={"who": "store"})
    assert caught.value.where.endswith("hext_projection/nodes.jsonl")
    assert len(store.nodes_log) == 0
    same = next(n for n in layer.nodes() if n["node_id"] == _doc_id())
    assert store.add_node(_doc_id(), same["node_type"], kind=same["kind"], verification=same["verification"], attrs=same["attrs"])  # same content: allowed


@needs_hext_docs
def test_projection_with_two_contents_under_one_node_id_is_a_collision(tmp_path, monkeypatch):
    real = hext_projection.project_document

    def two_contents(raw, **kw):
        out = real(raw, **kw)
        n = out["nodes"][0]
        out["nodes"] = [n, {**n, "attrs": {**n["attrs"], "lifecycle": "other"}}]
        return out

    monkeypatch.setattr(hext_projection, "project_document", two_contents)
    layer = HextProjectionLayer(tmp_path)
    with pytest.raises(CollisionError):
        layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    assert not layer.root.exists()


@needs_hext_docs
def test_projection_output_is_unchanged_by_the_guard(tmp_path):
    """Heads of a rebuild: the same value the test for two new processes already pins as reproducible."""
    docs = [(SRC.read_bytes(), {}), ((EX / "annotation-three-sentences.hext2.json").read_bytes(), {})]
    m = HextProjectionLayer(tmp_path / "a").rebuild(docs, resolve_session=lambda s: None)
    assert m["nodes"] > 0 and m["edges"] > 0
    assert m == HextProjectionLayer(tmp_path / "b").rebuild(list(reversed(docs)), resolve_session=lambda s: None)
    assert not [p for p in (tmp_path / "a").rglob("*") if p.is_file() and p.suffix not in (".jsonl", ".json")]  # no lock file


# ---- two processes at the same time -------------------------------------------------------------------
CHILD = r"""
import json, os, sys, time
sys.path.insert(0, {root!r})
from pathlib import Path
from graph.guard import CollisionError
data, role, ready, start = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
nid, doc = sys.argv[5], sys.argv[6]
def do_store(tag):
    from graph.store import GraphStore
    s = GraphStore(data, v01=False)
    ready.touch()
    while not start.exists(): pass
    return s.add_node(nid, "hext_document", attrs={{"who": tag}})
def do_layer():
    from graph.hext_projection import HextProjectionLayer
    l = HextProjectionLayer(data)
    ready.touch()
    while not start.exists(): pass
    l.rebuild([(Path(doc).read_bytes(), {{}})], resolve_session=lambda s: None)
    return True
try:
    ok = do_layer() if role == "layer" else do_store(role)
    print(json.dumps({{"role": role, "ok": ok}}))
except CollisionError as e:
    print(json.dumps({{"role": role, "collision": [e.node_id, e.existing_hash, e.new_hash]}}))
"""

MANY = r"""
import json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from graph.store import GraphStore
data, tag, ready, start = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
s = GraphStore(data)
ready.touch()
while not start.exists(): pass
for i in range(40):
    s.add_node(f"{{tag}}-{{i}}", "trajectory", attrs={{"i": i}})
    s.add_edge("measured_under", f"{{tag}}-{{i}}", "ps", {{}})
print("done")
"""


def _race(tmp_path: Path, code: str, argv_per_child: list[list[str]]) -> list[str]:
    start = tmp_path / "start"
    procs, readies = [], []
    for i, argv in enumerate(argv_per_child):
        ready = tmp_path / f"ready{i}"
        readies.append(ready)
        procs.append(subprocess.Popen([sys.executable, "-c", code.format(root=str(ROOT)), str(tmp_path / "data"), *argv[:1], str(ready), str(start), *argv[1:]],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    deadline = time.time() + 30
    while not all(r.exists() for r in readies):
        assert time.time() < deadline, "children did not start"
        time.sleep(0.01)
    start.touch()
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=60)
        assert p.returncode == 0, err
        outs.append(out.strip())
    return outs


@needs_hext_docs
@pytest.mark.parametrize("round_", range(12))
def test_store_and_projection_race_for_one_node_id_exactly_one_wins(tmp_path, round_):
    outs = [json.loads(o) for o in _race(tmp_path, CHILD, [["store", _doc_id(), str(SRC)], ["layer", _doc_id(), str(SRC)]])]
    won = [o for o in outs if o.get("ok")]
    lost = [o for o in outs if "collision" in o]
    assert len(won) == 1 and len(lost) == 1, outs
    data = tmp_path / "data"
    in_store = [n for n in GraphStore(data, v01=False).nodes() if n["node_id"] == _doc_id()]
    layer_nodes = [n for n in HextProjectionLayer(data).nodes()] if (data / "graph" / "hext_projection" / "nodes.jsonl").exists() else []
    in_layer = [n for n in layer_nodes if n["node_id"] == _doc_id()]
    if won[0]["role"] == "layer":
        assert not in_store and len(in_layer) == 1
    else:
        assert len(in_store) == 1 and not in_layer and in_store[0]["attrs"] == {"who": "store"}
    assert lost[0]["collision"][0] == _doc_id() and lost[0]["collision"][1] != lost[0]["collision"][2]


@pytest.mark.parametrize("round_", range(6))
def test_two_stores_race_for_one_node_id_exactly_one_wins(tmp_path, round_):
    outs = [json.loads(o) for o in _race(tmp_path, CHILD, [["store-a", "same", "-"], ["store-b", "same", "-"]])]
    assert sum(1 for o in outs if o.get("ok")) == 1 and sum(1 for o in outs if "collision" in o) == 1, outs
    nodes = GraphStore(tmp_path / "data", v01=False).nodes()
    assert len(nodes) == 1 and nodes[0]["attrs"]["who"] == next(o["role"] for o in outs if o.get("ok"))


def test_two_processes_appending_different_nodes_keep_one_unbroken_chain(tmp_path):
    outs = _race(tmp_path, MANY, [["a"], ["b"]])
    assert outs == ["done", "done"]
    s = GraphStore(tmp_path / "data")
    list(s.nodes_log.verify())
    list(s.edges_log.verify())
    assert len(s.nodes_log) == 80 and len(s.edges_log) == 80
    assert {n["node_id"] for n in s.nodes()} == {f"{t}-{i}" for t in "ab" for i in range(40)}
