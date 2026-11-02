# ruff: noqa: E501
"""axis_verification (CTP data model v0.3 section 5, v0.3.1 section 3, v0.3.2 sections 2-4).

A record per (target, axis, method/evaluator, value); a ``verifies`` edge to the target; ``supersedes`` /
``corrects`` between records of the same target and axis; the current state computed from the records nobody
replaced. Tests that need a HEXT document use the P0 examples of GemminAI/hext and are skipped when absent.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # plain `pytest` does not put the repo root on sys.path

from graph import axis  # noqa: E402
from graph.axis import AxisVerificationError, EdgeRuleError  # noqa: E402
from graph.hext_projection import HextProjectionLayer, doc_node_id  # noqa: E402
from graph.jcs import canonicalize  # noqa: E402
from graph.kindmap import d70_class  # noqa: E402
from graph.model import EDGE_TYPES, KINDS, NODE_TYPES, V01_ALLOWED, GraphError  # noqa: E402
from graph.store import GraphStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EX = Path.home() / "GemminAI" / "hext" / "examples" / "v2"
SRC = EX / "source-three-sentences.hext2.json"
needs_hext_docs = pytest.mark.skipif(not SRC.exists(), reason="P0 example documents absent")


def producer(tag: str = "a") -> dict[str, str]:
    return {"component": f"verifier-{tag}", "version": "1", "config_sha256": "c" * 64}


@pytest.fixture
def store(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    s.add_node("doc", "hext_document")  # an evidence node to verify
    s.add_node("doc2", "hext_document")
    s.add_node("traj", "trajectory")  # an observation: not verifiable
    return s


def write(store, value, tag, *, axis_name="semantic", target="doc", method="m"):
    return store.write_axis_verification(axis_name=axis_name, value=value, method=method, target=target, producer=producer(tag))


# ---- types, edges, vocabulary ---------------------------------------------------------------------
def test_the_new_type_kind_and_edges_exist_and_v01_is_unchanged():
    assert "axis_verification" in NODE_TYPES and "interpretation" in KINDS
    assert {"verifies", "supersedes", "corrects"} <= set(EDGE_TYPES)
    assert V01_ALLOWED == {("observation", "machine_verified"), ("observation", "unverified")}
    assert d70_class({"node_type": "axis_verification", "kind": "interpretation"}) == "interpretation"


def test_axis_verification_is_refused_in_the_v01_mode(tmp_path):
    s = GraphStore(tmp_path)  # v01=True
    s.add_node("doc", "hext_document")
    with pytest.raises(GraphError, match="v0.1 writes only"):
        s.write_axis_verification(axis_name="semantic", value="verified", method="m", target="doc", producer=producer())


def test_the_record_has_the_five_fields_and_the_chosen_values(store):
    nid = write(store, "verified", "a")
    rec = next(n for n in store.nodes() if n["node_id"] == nid)
    assert set(rec) == {"node_id", "node_type", "kind", "verification", "attrs"}
    assert (rec["node_type"], rec["kind"], rec["verification"]) == ("axis_verification", "interpretation", "unverified")
    assert set(rec["attrs"]) == {"axis", "value", "method", "detail_ref", "target", "input_hash", "producer"}
    assert nid.startswith("axis_verification:") and len(nid) == len("axis_verification:") + 64
    assert any(e["type"] == "verifies" and (e["src"], e["dst"]) == (nid, "doc") for e in store.edges())


@pytest.mark.parametrize(("axis_name", "value"), [("semantic", "sealed"), ("integrity", "verified"), ("structural", "verified"), ("independent", "passed"), ("mystery", "passed")])
def test_a_value_outside_the_axis_vocabulary_is_refused(store, axis_name, value):
    with pytest.raises(AxisVerificationError):
        write(store, value, "a", axis_name=axis_name)
    assert not [n for n in store.nodes() if n["node_type"] == "axis_verification"]


@pytest.mark.parametrize("axis_name", list(axis.AXIS_VALUES))
def test_every_value_of_every_axis_is_accepted(store, axis_name):
    for i, value in enumerate(axis.AXIS_VALUES[axis_name]):
        write(store, value, f"p{i}", axis_name=axis_name)
    assert store.axis_state("doc", axis_name).state == "conflict"


# ---- the three-step conflict of v0.3.1 section 3 ---------------------------------------------------
def test_conflict_and_how_it_is_resolved(store):
    a, b = write(store, "verified", "a"), write(store, "contradicted", "b")
    s = store.axis_state("doc", "semantic")
    assert s.state == "conflict" and [r["node_id"] for r in s.valid] == sorted([a, b])
    c = write(store, "verified", "c")
    store.replace_axis_verification(c, a)  # C supersedes A only
    s = store.axis_state("doc", "semantic")
    assert s.state == "conflict" and {r["node_id"] for r in s.valid} == {b, c}
    d = write(store, "verified", "d")
    store.replace_axis_verification(d, b)
    store.replace_axis_verification(d, c)  # D supersedes B and C
    s = store.axis_state("doc", "semantic")
    assert s.state == "verified" and [r["node_id"] for r in s.valid] == [d]


def test_no_record_is_pending_equal_records_give_the_value_and_corrects_removes_a_record(store):
    assert store.axis_state("doc", "semantic").state == "pending"
    assert store.axis_state("doc", "semantic").valid == ()
    a, b = write(store, "verified", "a"), write(store, "verified", "b")
    assert store.axis_state("doc", "semantic").state == "verified"
    wrong = write(store, "contradicted", "w")
    assert store.axis_state("doc", "semantic").state == "conflict"
    store.replace_axis_verification(a, wrong, correction=True)  # a corrects the mistake
    s = store.axis_state("doc", "semantic")
    assert s.state == "verified" and {r["node_id"] for r in s.valid} == {a, b}
    assert any(e["type"] == "corrects" for e in store.edges())


def test_other_axes_and_targets_do_not_mix(store):
    write(store, "verified", "a")
    write(store, "passed", "a", axis_name="structural")
    write(store, "contradicted", "a", target="doc2")
    assert store.axis_state("doc", "semantic").state == "verified"
    assert store.axis_state("doc", "structural").state == "passed"
    assert store.axis_state("doc2", "semantic").state == "contradicted"
    assert store.axis_state("doc", "integrity").state == "pending"


def test_no_clock_decides_which_record_is_newer(store):
    first, second = write(store, "verified", "a"), write(store, "contradicted", "b")
    assert store.axis_state("doc", "semantic").state == "conflict"  # the later write does not win
    store.replace_axis_verification(second, first)
    assert store.axis_state("doc", "semantic").state == "contradicted"  # only an edge decides


def test_state_is_read_by_another_store_object_on_the_same_dir(store, tmp_path):
    a = write(store, "verified", "a")
    other = GraphStore(tmp_path, v01=False)
    assert other.axis_state("doc", "semantic").state == "verified"
    write(store, "contradicted", "b")
    assert other.axis_state("doc", "semantic").state == "conflict"
    assert a in {r["node_id"] for r in other.axis_state("doc", "semantic").valid}


# ---- edge rules -------------------------------------------------------------------------------------
def test_supersedes_between_other_axes_or_targets_or_non_records_is_refused(store):
    a = write(store, "verified", "a")
    other_axis = write(store, "passed", "a", axis_name="structural")
    other_target = write(store, "verified", "a", target="doc2")
    for dst in (other_axis, other_target):
        for kind in ("supersedes", "corrects"):
            with pytest.raises(EdgeRuleError, match="different (axis|target)"):
                store.add_edge(kind, a, dst)
    with pytest.raises(EdgeRuleError, match="end must be a stored axis_verification"):
        store.add_edge("supersedes", a, "doc")
    with pytest.raises(EdgeRuleError, match="start must be a stored axis_verification"):
        store.add_edge("supersedes", "doc", a)
    with pytest.raises(EdgeRuleError, match="cannot replace itself"):
        store.add_edge("corrects", a, a)
    assert store.axis_state("doc", "semantic").state == "verified"


def test_verifies_needs_an_axis_start_a_matching_target_and_a_verifiable_class(store):
    a = write(store, "verified", "a")
    with pytest.raises(EdgeRuleError, match="start must be a stored axis_verification"):
        store.add_edge("verifies", "doc", "doc2")
    with pytest.raises(EdgeRuleError, match="not the end"):
        store.add_edge("verifies", a, "doc2")  # an evidence node, but not the record's target
    with pytest.raises(EdgeRuleError, match="not a stored node"):
        store.add_edge("verifies", a, "missing")
    # a record whose target is an observation: the node exists, its verifies edge is refused
    node = axis.build_node(axis="semantic", value="verified", method="m", detail_ref=None, target="traj",
                           target_record_hash=store._record_hash_of("traj"), producer=producer())
    store.add_node(node["node_id"], node["node_type"], kind=node["kind"], verification=node["verification"], attrs=node["attrs"])
    with pytest.raises(EdgeRuleError, match="classified 'observation'"):
        store.add_edge("verifies", node["node_id"], "traj")
    with pytest.raises(AxisVerificationError, match="classified 'observation'"):
        write(store, "verified", "a", target="traj")


def test_verifies_accepts_interpretation_targets(store):
    store.add_node("ver", "verification", kind="evaluation")
    nid = write(store, "undetermined", "a", target="ver")
    assert store.axis_state("ver", "semantic").state == "undetermined" and nid


def test_rewriting_the_same_edge_is_idempotent(store):
    a, b = write(store, "verified", "a"), write(store, "verified", "b")
    assert store.replace_axis_verification(a, b) is True
    assert store.replace_axis_verification(a, b) is False
    assert write(store, "verified", "a") == a  # the node and its verifies edge too
    assert len([e for e in store.edges() if e["type"] == "verifies" and e["src"] == a]) == 1


# ---- attrs, JCS, node_id ----------------------------------------------------------------------------
def _put(store, node):
    return store.add_node(node["node_id"], node["node_type"], kind=node["kind"], verification=node["verification"], attrs=node["attrs"])


def _node(store, **over):
    args = {"axis": "semantic", "value": "verified", "method": "m", "detail_ref": None, "target": "doc", "target_record_hash": store._record_hash_of("doc"), "producer": producer()}
    return axis.build_node(**{**args, **over})


@pytest.mark.parametrize("method", [{"score": 0.5}, {"m": ["a", {"b": 1.0}]}, {"n": 2**53}, {"n": -(2**60)}])
def test_a_float_or_an_out_of_range_integer_in_attrs_is_refused(store, method):
    from graph.jcs import JcsError

    with pytest.raises(JcsError):  # the builder cannot even hash it
        _node(store, method=method)
    forged = _node(store)
    forged["attrs"] = {**forged["attrs"], "method": method}  # a record written by hand
    with pytest.raises(AxisVerificationError, match="floating-point|outside"):
        _put(store, forged)
    with pytest.raises(JcsError):
        write(store, "verified", "a", method=method)
    assert not [n for n in store.nodes() if n["node_type"] == "axis_verification"]
    assert canonicalize({"n": 2**53 - 1})  # the boundary is fine
    assert _put(store, _node(store, method={"n": 2**53 - 1})) is True


def test_other_attrs_shapes_are_refused(store):
    good = _node(store)
    for attrs in ({**good["attrs"], "detail_ref": 5}, {**good["attrs"], "extra": 1}, {**good["attrs"], "producer": {"component": "x"}},
                  {k: v for k, v in good["attrs"].items() if k != "method"}, {**good["attrs"], "target": ""}):
        with pytest.raises(AxisVerificationError):
            store.add_node(good["node_id"], good["node_type"], kind=good["kind"], verification=good["verification"], attrs=attrs)
    with pytest.raises(AxisVerificationError, match="not a stored node"):
        _put(store, axis.build_node(axis="semantic", value="verified", method="m", detail_ref=None, target="nope", target_record_hash="0" * 64, producer=producer()))
    with pytest.raises(AxisVerificationError, match="verification column"):
        store.add_node(good["node_id"], good["node_type"], kind=good["kind"], verification="machine_verified", attrs=good["attrs"])


def test_a_node_id_that_is_not_the_derived_one_is_refused(store):
    good = _node(store)
    with pytest.raises(AxisVerificationError, match="not the one derived"):
        store.add_node("axis_verification:" + "0" * 64, good["node_type"], kind=good["kind"], verification=good["verification"], attrs=good["attrs"])
    stale = _node(store, target_record_hash="1" * 64)  # computed from a target record that is not the stored one
    with pytest.raises(AxisVerificationError, match="not the one derived"):
        _put(store, stale)


def test_the_node_id_is_the_documented_function_of_its_inputs(store):
    nid = write(store, "verified", "a")
    t = store._record_hash_of("doc")
    in_h = axis.input_hash("doc", t)
    res_h = axis.result_hash("verified", "m", None)
    fp = axis.producer_fingerprint(producer("a"))
    assert nid == axis.axis_node_id("semantic", "doc", in_h, res_h, fp)
    import hashlib

    assert in_h == hashlib.sha256(canonicalize([{"node_id": "doc", "record_hash": t}]).encode()).hexdigest()
    assert res_h == hashlib.sha256(canonicalize({"value": "verified", "method": "m", "detail_ref": None}).encode()).hexdigest()
    assert fp == hashlib.sha256(canonicalize(producer("a")).encode()).hexdigest()


def test_same_inputs_same_evaluator_but_another_value_are_two_records_and_both_stay(store):
    a = write(store, "verified", "same")
    b = write(store, "contradicted", "same")  # same evaluator, same target, same method, value differs
    assert a != b
    ids = [n["node_id"] for n in store.nodes() if n["node_type"] == "axis_verification"]
    assert sorted(ids) == sorted([a, b])
    assert store.axis_state("doc", "semantic").state == "conflict"
    again = write(store, "verified", "same")  # and the first one is still idempotent
    assert again == a and len([n for n in store.nodes() if n["node_type"] == "axis_verification"]) == 2


def test_a_different_producer_or_method_is_a_different_record(store):
    ids = {write(store, "verified", "a"), write(store, "verified", "b"), write(store, "verified", "a", method="m2")}
    assert len(ids) == 3


CHILD = r"""
import json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from graph.store import GraphStore
s = GraphStore(Path(sys.argv[1]), v01=False)
s.add_node("doc", "hext_document", attrs={{"x": 1}})
nid = s.write_axis_verification(axis_name="integrity", value="unverified", method={{"recompute": "seal"}}, target="doc",
                                producer={{"component": "v", "version": "1", "config_sha256": "c" * 64}}, detail_ref="blob:abc")
print(json.dumps({{"id": nid, "nodes": Path(sys.argv[1], "graph", "nodes.jsonl").read_text(), "edges": Path(sys.argv[1], "graph", "edges.jsonl").read_text()}}))
"""


def test_the_node_id_and_the_logs_are_identical_in_two_new_processes(tmp_path):
    outs = []
    for i, seed in enumerate(("1", "2")):
        done = subprocess.run([sys.executable, "-c", CHILD.format(root=str(ROOT)), str(tmp_path / f"p{i}")], capture_output=True, text=True, check=True,
                              env={**os.environ, "PYTHONHASHSEED": seed})
        outs.append(json.loads(done.stdout))
    assert outs[0]["id"] == outs[1]["id"] and outs[0]["id"].startswith("axis_verification:")
    assert outs[0] == outs[1]  # the whole log content, chain hashes included


# ---- a projected hext_document -----------------------------------------------------------------------
@needs_hext_docs
def test_a_projected_hext_documents_integrity_is_pending_until_a_sealed_record_is_valid(tmp_path):
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    heads = layer.manifest()["heads"]
    doc = doc_node_id(json.loads(SRC.read_bytes())["final_seal"]["state_hash"])
    s = GraphStore(tmp_path, v01=False)
    assert s.axis_state(doc, "integrity").state == "pending"
    assert d70_class(next(n for n in layer.nodes() if n["node_id"] == doc)) == "evidence"
    unver = s.write_axis_verification(axis_name="integrity", value="unverified", method="sha256 only; the signature is not verified", target=doc, producer=producer())
    assert s.axis_state(doc, "integrity").state == "unverified"  # SHA-256 equal is not "sealed"
    sealed = s.write_axis_verification(axis_name="integrity", value="sealed", method={"seal_recompute": True, "signature_verified": True}, target=doc, producer=producer("b"))
    assert s.axis_state(doc, "integrity").state == "conflict"
    s.replace_axis_verification(sealed, unver)
    assert s.axis_state(doc, "integrity").state == "sealed"
    assert HextProjectionLayer(tmp_path).manifest()["heads"] == heads  # the projection layer was not touched
    assert not [e for e in layer.edges() if e["type"] == "verifies"]
