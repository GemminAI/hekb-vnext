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


SEALED_METHOD = {
    "seal_recompute": {"ref": "blob:recomputed-seal"},
    "signature_verify": {"ref": "blob:signature-check", "key_ref": "key:hext-signing-1"},
}


@pytest.fixture
def sealed_enabled(monkeypatch):
    """The state after signature verification exists: sealed writes are checked by the method shape again."""
    monkeypatch.setattr(axis, "SEALED_WRITES_ENABLED", True)


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
def test_every_value_of_every_axis_is_accepted(store, axis_name, sealed_enabled):
    for i, value in enumerate(axis.AXIS_VALUES[axis_name]):
        method = SEALED_METHOD if (axis_name, value) == ("integrity", "sealed") else "m"
        write(store, value, f"p{i}", axis_name=axis_name, method=method)
    assert store.axis_state("doc", axis_name).state == "conflict"


# ---- the three-step conflict of v0.3.1 section 3 ---------------------------------------------------
def test_conflict_and_how_it_is_resolved(store):
    a, b = write(store, "verified", "a"), write(store, "contradicted", "b")
    s = store.axis_state("doc", "semantic")
    assert s.state == "conflict" and [r["node_id"] for r in s.records] == sorted([a, b])
    c = write(store, "verified", "c")
    store.replace_axis_verification(c, a)  # C supersedes A only
    s = store.axis_state("doc", "semantic")
    assert s.state == "conflict" and {r["node_id"] for r in s.records} == {b, c}
    d = write(store, "verified", "d")
    store.replace_axis_verification(d, b)
    store.replace_axis_verification(d, c)  # D supersedes B and C
    s = store.axis_state("doc", "semantic")
    assert s.state == "verified" and [r["node_id"] for r in s.records] == [d]


def test_no_record_is_pending_equal_records_give_the_value_and_corrects_removes_a_record(store):
    assert store.axis_state("doc", "semantic").state == "pending"
    assert store.axis_state("doc", "semantic").records == ()
    assert store.axis_state("doc", "semantic").basis == "no_record"
    a, b = write(store, "verified", "a"), write(store, "verified", "b")
    assert store.axis_state("doc", "semantic").state == "verified"
    wrong = write(store, "contradicted", "w")
    assert store.axis_state("doc", "semantic").state == "conflict"
    store.replace_axis_verification(a, wrong, correction=True)  # a corrects the mistake
    s = store.axis_state("doc", "semantic")
    assert s.state == "verified" and {r["node_id"] for r in s.records} == {a, b}
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
    assert a in {r["node_id"] for r in other.axis_state("doc", "semantic").records}


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
def test_a_projected_hext_documents_integrity_is_pending_until_a_sealed_record_is_valid(tmp_path, sealed_enabled):
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    heads = layer.manifest()["heads"]
    doc = doc_node_id(json.loads(SRC.read_bytes())["final_seal"]["state_hash"])
    s = GraphStore(tmp_path, v01=False)
    assert s.axis_state(doc, "integrity").state == "pending"
    assert d70_class(next(n for n in layer.nodes() if n["node_id"] == doc)) == "evidence"
    unver = s.write_axis_verification(axis_name="integrity", value="unverified", method="sha256 only; the signature is not verified", target=doc, producer=producer())
    assert s.axis_state(doc, "integrity").state == "unverified"  # SHA-256 equal is not "sealed"
    sealed = s.write_axis_verification(axis_name="integrity", value="sealed", method=SEALED_METHOD, target=doc, producer=producer("b"))
    assert s.axis_state(doc, "integrity").state == "conflict"
    s.replace_axis_verification(sealed, unver)
    assert s.axis_state(doc, "integrity").state == "sealed"
    assert HextProjectionLayer(tmp_path).manifest()["heads"] == heads  # the projection layer was not touched
    assert not [e for e in layer.edges() if e["type"] == "verifies"]


# ---- v0.3.4 section 4: the four corrections -----------------------------------------------------------
def test_integrity_sealed_needs_both_references(store, sealed_enabled):
    ok = write(store, "sealed", "a", axis_name="integrity", method=SEALED_METHOD)
    s = store.axis_state("doc", "integrity")
    assert s.state == "sealed" and s.basis == "recorded" and [r["node_id"] for r in s.records] == [ok]
    bad_methods = [
        "seal recomputed and signature verified",  # a string
        {},
        {"seal_recompute": {"ref": "r"}},  # one reference only
        {"signature_verify": {"ref": "r", "key_ref": "k"}},  # the other one only
        {"seal_recompute": {"ref": "r"}, "signature_verify": {"ref": "r"}},  # no key_ref
        {"seal_recompute": {"ref": "r"}, "signature_verify": {"key_ref": "k"}},  # no ref
        {"seal_recompute": {"ref": ""}, "signature_verify": {"ref": "r", "key_ref": "k"}},
        {"seal_recompute": True, "signature_verify": True},
        {"seal_recompute": {"ref": 1}, "signature_verify": {"ref": "r", "key_ref": "k"}},
    ]
    for i, method in enumerate(bad_methods):
        with pytest.raises(AxisVerificationError, match="integrity=sealed|method must be"):
            write(store, "sealed", f"b{i}", axis_name="integrity", method=method)
    assert len([n for n in store.nodes() if n["node_type"] == "axis_verification"]) == 1


def test_other_axes_and_values_keep_their_method_rules(store, sealed_enabled):
    write(store, "unverified", "a", axis_name="integrity", method="sha256 only")
    write(store, "invalid", "b", axis_name="integrity", method="m")
    write(store, "verified", "c", axis_name="semantic", method={"anything": 1})
    assert write(store, "sealed", "d", axis_name="integrity", method=SEALED_METHOD)  # but sealed alone is checked


def test_a_replacement_cycle_is_refused_and_others_pass(store):
    a, b, c = (write(store, "verified", t) for t in "abc")
    assert store.replace_axis_verification(a, b) is True  # a replaces b
    with pytest.raises(EdgeRuleError, match="cycle"):
        store.replace_axis_verification(b, a)  # b replaces a: mutual
    with pytest.raises(EdgeRuleError, match="cycle"):
        store.replace_axis_verification(b, a, correction=True)  # corrects counts the same
    assert store.replace_axis_verification(b, c) is True  # b replaces c: a -> b -> c
    with pytest.raises(EdgeRuleError, match="cycle"):
        store.replace_axis_verification(c, a)  # c replaces a: a -> b -> c -> a
    d = write(store, "contradicted", "d")
    assert store.replace_axis_verification(d, a) is True  # d -> a -> b -> c: still acyclic
    assert store.replace_axis_verification(d, c) is True  # a second path to c, no cycle
    with pytest.raises(EdgeRuleError, match="itself"):
        store.replace_axis_verification(a, a)
    assert store.axis_consistency().cyclic == ()
    assert store.axis_state("doc", "semantic").state == "contradicted"  # only d is left


def test_a_three_record_cycle_across_supersedes_and_corrects_is_refused(store):
    a, b, c = (write(store, "verified", t) for t in "abc")
    store.replace_axis_verification(a, b)
    store.replace_axis_verification(b, c, correction=True)
    with pytest.raises(EdgeRuleError, match="cycle"):
        store.replace_axis_verification(c, a, correction=True)


def test_an_existing_cycle_is_reported_not_raised(tmp_path):
    """Logs written before the rule can hold a cycle: reading must not break, the report names the records."""
    s = GraphStore(tmp_path, v01=False)
    s.add_node("doc", "hext_document")
    a, b = write(s, "verified", "a"), write(s, "contradicted", "b")
    s.edges_log.append({"type": "supersedes", "src": a, "dst": b, "attrs": {}})  # as an old writer could have
    s.edges_log.append({"type": "supersedes", "src": b, "dst": a, "attrs": {}})
    fresh = GraphStore(tmp_path, v01=False)
    state = fresh.axis_state("doc", "semantic")  # no exception
    assert (state.state, state.basis, state.records) == ("pending", "no_record", ())
    assert fresh.axis_consistency().cyclic == tuple(sorted([a, b])) and not fresh.axis_consistency().ok


def test_basis_tells_no_record_from_a_recorded_pending(store):
    none = store.axis_state("doc", "semantic")
    assert (none.state, none.basis) == ("pending", "no_record")
    pending = write(store, "pending", "a")
    s = store.axis_state("doc", "semantic")
    assert (s.state, s.basis) == ("pending", "recorded") and [r["node_id"] for r in s.records] == [pending]
    write(store, "verified", "b")
    assert store.axis_state("doc", "semantic").basis == "recorded"  # conflict is recorded too
    assert store.axis_state("doc", "semantic").state == "conflict"
    assert store.axis_state("doc", "integrity").basis == "no_record"  # integrity has no value "pending": only no_record reads so


def test_a_record_without_its_verifies_edge_is_read_and_reported_until_the_write_is_repeated(store):
    node = _node(store)  # the node alone: what is left if the process stops between the two writes
    _put(store, node)
    s = store.axis_state("doc", "semantic")
    assert s.state == "verified" and s.basis == "recorded" and [r["node_id"] for r in s.records] == [node["node_id"]]
    report = store.axis_consistency()
    assert report.missing_verifies == (node["node_id"],) and not report.ok
    assert not [e for e in store.edges() if e["type"] == "verifies"]
    again = store.write_axis_verification(axis_name="semantic", value="verified", method="m", target="doc", producer=producer())
    assert again == node["node_id"]
    assert store.axis_consistency().ok and store.axis_consistency().missing_verifies == ()
    assert len([e for e in store.edges() if e["type"] == "verifies"]) == 1
    assert store.axis_state("doc", "semantic").state == "verified"


def test_the_state_follows_attrs_target_not_the_edge(store):
    """A verifies edge is an index: the record is read for the target in its attrs whatever edges exist."""
    nid = write(store, "contradicted", "a", target="doc")
    assert store.axis_state("doc", "semantic").state == "contradicted"
    assert store.axis_state("doc2", "semantic").state == "pending"
    assert nid in {r["node_id"] for r in store.axis_state("doc", "semantic").records}


# ---- sealed writes are refused until signatures can be verified ---------------------------------------
def test_the_switch_is_off_and_is_one_constant():
    assert axis.SEALED_WRITES_ENABLED is False
    assert axis.SEALED_REFUSED_CODE == "SIGNATURE_VERIFICATION_NOT_IMPLEMENTED"


@pytest.mark.parametrize("method", [SEALED_METHOD, "m", {}, {"seal_recompute": {"ref": "r"}}])
def test_a_sealed_write_is_refused_whatever_the_method_says(store, method):
    with pytest.raises(AxisVerificationError, match="SIGNATURE_VERIFICATION_NOT_IMPLEMENTED") as caught:
        write(store, "sealed", "a", axis_name="integrity", method=method)
    assert caught.value.reason_code == "SIGNATURE_VERIFICATION_NOT_IMPLEMENTED"
    assert not [n for n in store.nodes() if n["node_type"] == "axis_verification"]
    assert store.axis_state("doc", "integrity").state == "pending"


def test_the_node_check_itself_refuses_a_hand_written_sealed_record(store, sealed_enabled, monkeypatch):
    node = _node(store, axis="integrity", value="sealed", method=SEALED_METHOD)
    monkeypatch.setattr(axis, "SEALED_WRITES_ENABLED", False)
    with pytest.raises(AxisVerificationError, match="SIGNATURE_VERIFICATION_NOT_IMPLEMENTED"):
        _put(store, node)


def test_other_integrity_values_and_other_axes_are_not_affected(store):
    for value in ("unverified", "invalid", "not_applicable"):
        write(store, value, value, axis_name="integrity")
    write(store, "verified", "s", axis_name="semantic")
    assert store.axis_state("doc", "integrity").state == "conflict"
    assert store.axis_state("doc", "semantic").state == "verified"


def test_with_the_switch_on_the_method_shape_check_works_as_before(store, sealed_enabled):
    assert write(store, "sealed", "a", axis_name="integrity", method=SEALED_METHOD)
    assert store.axis_state("doc", "integrity").state == "sealed"
    for method in ("m", {}, {"seal_recompute": {"ref": "r"}}, {"seal_recompute": {"ref": "r"}, "signature_verify": {"ref": "r"}}):
        with pytest.raises(AxisVerificationError, match="integrity=sealed|method must be") as caught:
            write(store, "sealed", "b", axis_name="integrity", method=method)
        assert caught.value.reason_code is None  # the shape refusals have no code; the switch does
