# ruff: noqa: E501
"""CTP episode records, first half (graph/ctp.py): the ten types and their edges (CTP data model v0.3 - v0.3.4).

Tests that need the P0 HEXT documents of GemminAI/hext are skipped when they are absent.
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # plain `pytest` does not put the repo root on sys.path

from graph import ctp  # noqa: E402
from graph.blobs import blob_path, write_blob  # noqa: E402
from graph.ctp import CtpError  # noqa: E402
from graph.hext_projection import HextProjectionLayer, doc_node_id  # noqa: E402
from graph.jcs import canonicalize  # noqa: E402
from graph.kindmap import KindMapError, d70_class  # noqa: E402
from graph.model import EDGE_TYPES, KINDS, NODE_TYPES, GraphError  # noqa: E402
from graph.store import GraphStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EX = Path.home() / "GemminAI" / "hext" / "examples" / "v2"
SRC = EX / "source-three-sentences.hext2.json"
needs_hext_docs = pytest.mark.skipif(not SRC.exists(), reason="P0 example documents absent")

SPEC_HASH = "sha256:" + "5" * 64
PRODUCER = {"component": "verifier", "version": "1", "config_sha256": "c" * 64}


def producer(tag="a"):
    return {**PRODUCER, "component": f"verifier-{tag}"}


@pytest.fixture
def store(tmp_path):
    s = GraphStore(tmp_path, v01=False, blob_dir=tmp_path)
    s.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    s.add_node("hextdoc:sha256:annot", "hext_document", attrs={"hext_kind": "annotation", "state_hash": "sha256:annot"})
    s.add_node("traj", "trajectory")
    return s


def episode_fields(**over):
    return {"experiment_spec_hash": SPEC_HASH, "item_id": "i1", "bundle_id": "b1", "split": "dev", "condition": "a", "replicate": 0, "rng_seed": 7, **over}


def write_episode(store, **over):
    return store.write_ctp_node("episode", episode_fields(**over), producer=producer())


def write_context(store, ep, **over):
    fields = {"episode_id": ep, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False, **over}
    return store.write_ctp_node("context_observation", fields, producer=producer())


def ctx_blob(store, label):
    """Put the context a step was given into the store's blob location; returns its sha256."""
    return write_blob(store.blob_dir, f"context:{label}".encode())


def step_fields(ep, step="f1", sha="2" * 64, **over):
    return {"episode_id": ep, "step": step, "text_ref": "blob:" + "3" * 64, "input_context_ref": f"blob:{sha}", "input_context_sha256": sha, **over}


def write_step(store, ep, step="f1", sha=None, **over):
    sha = sha or ctx_blob(store, f"{ep}/{step}")
    return store.write_ctp_node("step_output", step_fields(ep, step, sha, **over), producer=producer())


@pytest.fixture
def ep(store):
    return write_episode(store)


def fresh(store, n):
    """A new episode: a refused record must not share its node_id with a valid one (that would be a D13 collision instead)."""
    return write_episode(store, item_id=f"fresh{n}")


# ---- registration -----------------------------------------------------------------------------------------
def test_the_twelve_types_and_ten_edges_are_registered_and_what_is_still_later_is_not():
    assert set(ctp.CTP_NODE_TYPES) == {"episode", "context_observation", "step_output", "applicability", "carry_evidence", "cle_result", "compliance", "plan", "governance", "outcome", "assignment", "experience"}
    assert "presented_to" in ctp.CTP_EDGE_TYPES
    assert set(ctp.CTP_NODE_TYPES) <= set(NODE_TYPES) and set(ctp.CTP_EDGE_TYPES) <= set(EDGE_TYPES)
    assert {"selection", "outcome", "interpretation"} <= set(KINDS)
    for later in ("experiment_manifest", "human", "trajectory_role"):
        assert later not in NODE_TYPES
    for later in ("has_trajectory", "queried_with"):
        assert later not in EDGE_TYPES


def test_every_type_stores_its_d70_word_as_kind_and_classifies_as_it():
    for node_type, spec in ctp.SPECS.items():
        assert d70_class({"node_type": node_type, "kind": "anything"}) == spec.d70
    from graph.kindmap import check_stored_kind

    with pytest.raises(KindMapError):
        check_stored_kind({"node_id": "x", "node_type": "plan", "kind": "observation"})


def test_ctp_records_need_the_non_v01_store(tmp_path):
    s = GraphStore(tmp_path)  # v01=True
    s.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    with pytest.raises(GraphError):
        write_episode(s)


def test_the_record_has_five_fields_and_the_common_fields_in_attrs(store, ep):
    rec = next(n for n in store.nodes() if n["node_id"] == ep)
    assert set(rec) == {"node_id", "node_type", "kind", "verification", "attrs"}
    assert (rec["node_type"], rec["kind"], rec["verification"]) == ("episode", "observation", "unverified")
    assert {"schema", "refs", "producer", "observed_at"} <= set(rec["attrs"]) and rec["attrs"]["schema"] == "ctp.datamodel/0.3"


# ---- episode and under_manifest ------------------------------------------------------------------------------
def test_episode_identity_includes_the_experiment_spec_hash(store):
    other = "sha256:" + "6" * 64
    store.add_node(f"hextdoc:{other}", "hext_document", attrs={"hext_kind": "source", "state_hash": other})
    a, b = write_episode(store), write_episode(store, experiment_spec_hash=other)
    assert a != b and a == write_episode(store)
    assert a == ctp.node_id_of("episode", episode_fields())


def test_episode_rules(store):
    for bad in ({"split": "test"}, {"replicate": -1}, {"replicate": True}, {"rng_seed": "7"}, {"item_id": ""}, {"experiment_spec_hash": "sha256:" + "9" * 64}):
        with pytest.raises(GraphError):
            write_episode(store, **bad)
    store.add_node("hextdoc:sha256:annot2", "hext_document", attrs={"hext_kind": "annotation", "state_hash": "sha256:annot2"})
    with pytest.raises(CtpError, match="HEXT source document"):
        write_episode(store, experiment_spec_hash="sha256:annot2")  # an annotation is not an experiment spec
    assert not [n for n in store.nodes() if n["node_type"] == "episode"]


def test_under_manifest_points_at_the_source_with_the_same_state_hash(store, ep):
    assert store.add_edge("under_manifest", ep, f"hextdoc:{SPEC_HASH}") is True
    assert store.add_edge("under_manifest", ep, f"hextdoc:{SPEC_HASH}") is False
    other = "sha256:" + "6" * 64
    store.add_node(f"hextdoc:{other}", "hext_document", attrs={"hext_kind": "source", "state_hash": other})
    with pytest.raises(CtpError, match="experiment_spec_hash"):
        store.add_edge("under_manifest", ep, f"hextdoc:{other}")
    with pytest.raises(CtpError, match="HEXT document"):
        store.add_edge("under_manifest", ep, "traj")
    with pytest.raises(CtpError, match="episode points"):
        store.add_edge("under_manifest", "traj", f"hextdoc:{SPEC_HASH}")


# ---- context_observation -------------------------------------------------------------------------------------
def test_a_context_observation_never_contains_experience(store, ep):
    ctx = write_context(store, ep)
    assert next(n for n in store.nodes() if n["node_id"] == ctx)["attrs"]["contains_experience"] is False
    for n, bad in enumerate((True, "false", 0, None)):
        with pytest.raises(CtpError, match="contains_experience must be false"):
            write_context(store, fresh(store, n), contains_experience=bad)
    with pytest.raises(CtpError, match="not a stored episode"):
        write_context(store, "episode:" + "0" * 64)
    with pytest.raises(CtpError, match="list of strings"):
        write_context(store, fresh(store, 9), knowledge_ids="k1")


def test_has_context_and_has_step_need_the_same_episode(store, ep):
    other = write_episode(store, item_id="i2")
    ctx, step = write_context(store, ep), write_step(store, ep)
    assert store.add_edge("has_context", ep, ctx) and store.add_edge("has_step", ep, step)
    with pytest.raises(CtpError, match="another episode"):
        store.add_edge("has_context", other, ctx)
    with pytest.raises(CtpError, match="another episode"):
        store.add_edge("has_step", other, step)
    with pytest.raises(CtpError, match="episode points"):
        store.add_edge("has_step", ep, ctx)


# ---- step_output ---------------------------------------------------------------------------------------------------
def test_step_output_needs_the_input_context_ref_and_its_sha256(store, ep):
    assert write_step(store, ep)
    node = store.build_ctp_node("step_output", step_fields(fresh(store, 1), sha=ctx_blob(store, "seven")), producer=producer())
    for drop in ("input_context_ref", "input_context_sha256"):
        with pytest.raises(CtpError, match="attrs lack"):
            store.add_node(node["node_id"], "step_output", kind="observation", verification="unverified", attrs={k: v for k, v in node["attrs"].items() if k != drop})
    with pytest.raises(CtpError, match="blob:<sha256"):
        write_step(store, fresh(store, 2), input_context_ref="s3://bucket/x")
    with pytest.raises(CtpError, match="not the sha256 in input_context_ref"):
        write_step(store, fresh(store, 3), input_context_sha256="4" * 64)
    with pytest.raises(CtpError, match="is not one of"):
        write_step(store, fresh(store, 4), step="f4")


def test_the_blob_must_be_given_present_and_hash_to_input_context_sha256(store, ep, tmp_path):
    sha = write_blob(tmp_path, b"the context given to f2")
    assert write_step(store, ep, step="f2", sha=sha)  # location given, blob there, bytes match
    absent = "a" * 64
    with pytest.raises(CtpError, match="is not in the blob location"):
        write_step(store, ep, step="f3", sha=absent)  # an address alone is not enough any more
    other = hashlib.sha256(b"something else").hexdigest()
    path = blob_path(tmp_path, other)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not what the address says")  # a blob whose bytes do not hash to its name
    with pytest.raises(CtpError, match="does not hash"):
        write_step(store, fresh(store, 7), step="f3", sha=other)


def test_a_store_without_a_blob_location_refuses_every_step_output(tmp_path):
    s = GraphStore(tmp_path, v01=False)  # no blob_dir
    s.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    e = s.write_ctp_node("episode", episode_fields(), producer=producer())
    sha = write_blob(tmp_path, b"the blob exists on disk, but the store was not told where blobs are")
    with pytest.raises(CtpError, match="no blob location was given"):
        s.write_ctp_node("step_output", step_fields(e, "f1", sha), producer=producer())
    assert not [n for n in s.nodes() if n["node_type"] == "step_output"]


def test_retrieved_goes_from_the_f1_step_to_a_source_document(store, ep):
    f1, f2 = write_step(store, ep, "f1"), write_step(store, ep, "f2")
    assert store.add_edge("retrieved", f1, f"hextdoc:{SPEC_HASH}")
    with pytest.raises(CtpError, match="f1 step_output"):
        store.add_edge("retrieved", f2, f"hextdoc:{SPEC_HASH}")
    with pytest.raises(CtpError, match="HEXT source"):
        store.add_edge("retrieved", f1, "hextdoc:sha256:annot")


# ---- interpretations -----------------------------------------------------------------------------------------
def write_carry(store, ep, step_id, **over):
    fields = {"episode_id": ep, "step": "f1", "property": "c1.condition", "status": "lost", "basis": "evidence", "evidence_ref": "fixture:c1/B1/f1", **over}
    return store.write_ctp_node("carry_evidence", fields, producer=producer(), input_nodes=(step_id, f"hextdoc:{SPEC_HASH}"))


def test_carry_evidence_rules_and_the_interprets_edge(store, ep):
    step = write_step(store, ep)
    carry = write_carry(store, ep, step)
    assert store.add_edge("interprets", carry, step) and store.add_edge("interprets", carry, f"hextdoc:{SPEC_HASH}")
    assert write_carry(store, ep, step, basis="declared", evidence_ref=None, status="preserved")  # a declaration needs no ref
    for bad in ({"status": "kept"}, {"basis": "guess"}, {"evidence_ref": None}, {"evidence_ref": ""}, {"step": "f9"}, {"property": ""}):
        with pytest.raises(GraphError):
            write_carry(store, ep, step, **bad)
    plan = store.write_ctp_node("plan", {"episode_id": ep, "plan_id": "p", "steps": ["a"], "probability_source": "unobserved"}, producer=producer(), input_nodes=(step,))
    with pytest.raises(CtpError, match="an interpretation"):
        store.add_edge("interprets", plan, step)
    with pytest.raises(CtpError, match="observation or evidence"):
        store.add_edge("interprets", carry, plan)


def test_applicability_cle_result_and_compliance_vocabularies(store, ep):
    step = write_step(store, ep)
    w = lambda t, f, **kw: store.write_ctp_node(t, {"episode_id": ep, **f}, producer=producer(), input_nodes=(step,), **kw)  # noqa: E731
    assert w("applicability", {"constraint_id": "c1", "applicability": "undetermined", "applicability_basis": "fixture"})
    assert w("cle_result", {"rule": "R9", "mode": "M2", "value": "FAIL", "reason_codes": ["OBLIGATION_LOST_BY_EVIDENCE"], "locations": ["f1"], "history_hash": "h"})
    assert w("compliance", {"level": "L3", "judge": "verifier", "value": "UNVERIFIABLE", "reason_codes": []})
    bad = [
        ("applicability", {"constraint_id": "c1", "applicability": "maybe", "applicability_basis": "fixture"}),
        ("cle_result", {"rule": "R10", "mode": "M2", "value": "FAIL", "reason_codes": [], "locations": [], "history_hash": "h"}),
        ("cle_result", {"rule": "R9", "mode": "M3", "value": "FAIL", "reason_codes": [], "locations": [], "history_hash": "h"}),
        ("cle_result", {"rule": "R9", "mode": "M1", "value": "FAILED", "reason_codes": [], "locations": [], "history_hash": "h"}),
        ("cle_result", {"rule": "R9", "mode": "M1", "value": "FAIL", "reason_codes": ["B", "A"], "locations": [], "history_hash": "h"}),
        ("compliance", {"level": "L4", "judge": "verifier", "value": "PASS", "reason_codes": []}),
        ("compliance", {"level": "L1", "judge": "llm", "value": "PASS", "reason_codes": []}),
        ("compliance", {"level": "L1", "judge": "verifier", "value": "NOT_APPLICABLE", "reason_codes": []}),
    ]
    for n, (node_type, fields) in enumerate(bad):
        e2 = fresh(store, n)
        s2 = write_step(store, e2)
        with pytest.raises(CtpError, match="is not one of|sorted list"):
            store.write_ctp_node(node_type, {"episode_id": e2, **fields}, producer=producer(), input_nodes=(s2,))


def test_audits_connects_an_independent_judgement_to_the_verifiers_one(store, ep):
    step = write_step(store, ep)
    mk = lambda judge, level="L2", tag="a": store.write_ctp_node("compliance", {"episode_id": ep, "level": level, "judge": judge, "value": "PASS", "reason_codes": []}, producer=producer(tag), input_nodes=(step,))  # noqa: E731
    v, i = mk("verifier"), mk("independent", tag="i")
    assert store.add_edge("audits", i, v)
    with pytest.raises(CtpError, match="independent compliance"):
        store.add_edge("audits", v, i)
    with pytest.raises(CtpError, match="differ in episode or level"):
        store.add_edge("audits", mk("independent", level="L3", tag="i"), v)


# ---- selections ---------------------------------------------------------------------------------------------
def write_plan(store, ep, step, **over):
    return store.write_ctp_node("plan", {"episode_id": ep, "plan_id": "p1", "steps": ["f1", "f2", "f3"], "probability_source": "unobserved", **over}, producer=producer(), input_nodes=(step,))


def write_gov(store, ep, plan, **over):
    return store.write_ctp_node("governance", {"episode_id": ep, "plan_id": "p1", "decision": "hold", "reason_codes": ["OBLIGATION_DECLARED_LOST"], **over}, producer=producer("g"),
                                input_nodes=(plan,), input_facts={"quota": {"active": 0, "spent": 3}, "cancelled": False})


def test_a_plan_has_probability_source_unobserved_and_no_probability(store, ep):
    step = write_step(store, ep)
    assert write_plan(store, ep, step)
    n = 0

    def attempt(**over):
        nonlocal n
        n += 1
        e2 = fresh(store, n)
        return write_plan(store, e2, write_step(store, e2), **over)

    for bad in ({"probability_source": "selector"}, {"probability_source": "single_candidate"}, {"probability_source": None}):
        with pytest.raises(CtpError, match="unobserved"):
            attempt(**bad)
    for key in ("action_probability", "probability", "selection_probability"):
        with pytest.raises(CtpError, match="no probability"):
            attempt(**{key: {"num": 1, "den": 2}})
    with pytest.raises(CtpError, match="no probability"):
        attempt(steps=[{"id": "f1", "probability": {"num": 1, "den": 3}}])


def test_governance_has_no_probability_and_no_probability_source(store, ep):
    plan = write_plan(store, ep, write_step(store, ep))
    gov = write_gov(store, ep, plan)
    rec = next(n for n in store.nodes() if n["node_id"] == gov)["attrs"]
    assert rec["referenced_inputs"] == sorted([plan, "quota", "cancelled"]) and "probability_source" not in rec
    for key in ("probability_source", "action_probability", "probability"):
        node = store.build_ctp_node("governance", {"episode_id": ep, "plan_id": "p1", "decision": "admit", "reason_codes": [], "referenced_inputs": [plan], key: "x"}, producer=producer("g"), input_nodes=(plan,))
        with pytest.raises(CtpError, match="outside the type|no probability"):
            store.add_node(node["node_id"], "governance", kind=node["kind"], verification=node["verification"], attrs=node["attrs"])
    with pytest.raises(CtpError, match="decision"):
        write_gov(store, ep, plan, decision="maybe")


def test_governance_hashes_the_inputs_it_really_read(store, ep):
    plan = write_plan(store, ep, write_step(store, ep))
    base = write_gov(store, ep, plan)
    changed = store.write_ctp_node("governance", {"episode_id": ep, "plan_id": "p1", "decision": "hold", "reason_codes": ["OBLIGATION_DECLARED_LOST"]}, producer=producer("g"),
                                   input_nodes=(plan,), input_facts={"quota": {"active": 1, "spent": 3}, "cancelled": False})
    assert base != changed  # another quota state is another decision event
    node = store.build_ctp_node("governance", {"episode_id": ep, "plan_id": "p1", "decision": "hold", "reason_codes": [], "referenced_inputs": ["only", "these"]}, producer=producer("g"), input_nodes=(plan,))
    with pytest.raises(CtpError, match="referenced_inputs must name exactly"):
        store.add_node(node["node_id"], "governance", kind=node["kind"], verification=node["verification"], attrs=node["attrs"])


def test_selects_and_considered_and_outcome_of(store, ep):
    step = write_step(store, ep)
    plan = write_plan(store, ep, step)
    gov = write_gov(store, ep, plan)
    other_plan = write_plan(store, ep, step, plan_id="p2")
    assert store.add_edge("selects", gov, plan)
    with pytest.raises(CtpError, match="same plan_id"):
        store.add_edge("selects", gov, other_plan)
    with pytest.raises(CtpError, match="governance record selects"):
        store.add_edge("selects", plan, step)
    assert store.add_edge("considered", gov, step) and store.add_edge("considered", plan, step)
    with pytest.raises(CtpError, match="a selection"):
        store.add_edge("considered", step, step)
    out = store.write_ctp_node("outcome", outcome_fields(ep), producer=producer("o"), input_nodes=(gov,))
    assert store.add_edge("outcome_of", out, ep) and store.add_edge("outcome_of", out, gov)
    other_ep = write_episode(store, item_id="i2")
    with pytest.raises(CtpError, match="another episode"):
        store.add_edge("outcome_of", out, other_ep)
    with pytest.raises(CtpError, match="episode or of a selection"):
        store.add_edge("outcome_of", out, step)


# ---- outcome ----------------------------------------------------------------------------------------------------
def outcome_fields(ep, **over):
    return {"episode_id": ep, "carry": {"f1": "lost"}, "constraint_compliance": "FAIL", "two_axis": "LOST_VIOLATED", **over}


def test_outcome_two_axis_vocabulary_undetermined_axis_and_no_reward(store, ep):
    gov = write_gov(store, ep, write_plan(store, ep, write_step(store, ep)))
    counter = iter(range(100))

    def w(**over):
        e2 = fresh(store, next(counter))  # a refused outcome must not share its id with a stored one
        return store.write_ctp_node("outcome", outcome_fields(e2, **over), producer=producer("o"), input_nodes=(gov,))

    for cls in ("CARRIED_COMPLIED", "CARRIED_VIOLATED", "LOST_COMPLIED", "LOST_VIOLATED"):
        assert w(two_axis=cls)
    assert w(two_axis="UNDETERMINED", undetermined_axis="compliance")
    with pytest.raises(CtpError, match="needs undetermined_axis"):
        w(two_axis="UNDETERMINED")
    with pytest.raises(CtpError, match="needs undetermined_axis"):
        w(two_axis="UNDETERMINED", undetermined_axis="neither")
    with pytest.raises(CtpError, match="only for two_axis=UNDETERMINED"):
        w(two_axis="LOST_VIOLATED", undetermined_axis="retention")
    with pytest.raises(CtpError, match="is not one of"):
        w(two_axis="BOTH_FINE")
    for key in ("reward", "reward_spec_hash", "total_reward"):
        with pytest.raises(CtpError, match="no reward"):
            w(**{key: 1})
    with pytest.raises(CtpError, match="no reward"):
        w(carry={"reward": 1})


# ---- JCS: floats and out-of-range integers ---------------------------------------------------------------------
@pytest.mark.parametrize("over", [{"refs": {"score": 0.5}}, {"refs": {"n": 2**60}}, {"refs": {"deep": [{"x": -(2**53)}]}}])
def test_floats_and_out_of_range_integers_are_refused_in_every_ctp_type(store, ep, over):
    for n, node_type in enumerate(t for t in ctp.CTP_NODE_TYPES if t not in ctp.DRAWING):  # the drawing types: tests/test_ctp_types_b.py
        e2 = fresh(store, n)
        step = write_step(store, e2)
        fields = {
            "episode": episode_fields(item_id=f"f{n}"),
            "context_observation": {"episode_id": e2, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False},
            "step_output": step_fields(e2, "f2", ctx_blob(store, "floats")),
            "applicability": {"episode_id": e2, "constraint_id": "c1", "applicability": "applicable", "applicability_basis": "fixture"},
            "carry_evidence": {"episode_id": e2, "step": "f1", "property": "p", "status": "lost", "basis": "declared", "evidence_ref": None},
            "cle_result": {"episode_id": e2, "rule": "R9", "mode": "M1", "value": "PASS", "reason_codes": [], "locations": [], "history_hash": "h"},
            "compliance": {"episode_id": e2, "level": "L1", "judge": "verifier", "value": "PASS", "reason_codes": []},
            "plan": {"episode_id": e2, "plan_id": "p", "steps": [], "probability_source": "unobserved"},
            "governance": {"episode_id": e2, "plan_id": "p", "decision": "admit", "reason_codes": []},
            "outcome": outcome_fields(e2),
        }[node_type]
        node = store.build_ctp_node(node_type, fields, producer=producer(), input_nodes=(step,) if ctp.SPECS[node_type].judgement else ())
        with pytest.raises(CtpError, match="floating-point|outside"):
            store.add_node(node["node_id"], node_type, kind=node["kind"], verification="unverified", attrs={**node["attrs"], **over})
        assert store.add_node(node["node_id"], node_type, kind=node["kind"], verification="unverified", attrs=node["attrs"])  # the clean record is fine
    assert canonicalize({"n": 2**53 - 1})
    assert store.write_ctp_node("episode", episode_fields(item_id="edge"), producer=producer(), refs={"n": 2**53 - 1})  # the boundary passes


def test_a_general_graph_write_still_accepts_floats(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    assert s.add_node("t", "trajectory", attrs={"x": 0.5})


# ---- node ids and determinism ----------------------------------------------------------------------------------
def test_observed_at_is_not_part_of_the_id_and_does_not_order_anything(store, ep):
    a = store.build_ctp_node("context_observation", {"episode_id": ep, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False}, producer=producer(), observed_at="2026-11-03T00:00:00Z")
    b = store.build_ctp_node("context_observation", {"episode_id": ep, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False}, producer=producer(), observed_at="2030-01-01T00:00:00Z")
    assert a["node_id"] == b["node_id"]
    store.add_node(a["node_id"], "context_observation", kind=a["kind"], verification="unverified", attrs=a["attrs"])
    from graph.guard import CollisionError

    with pytest.raises(CollisionError):  # same id, another record (D13): a re-run must repeat observed_at (or leave it null)
        store.add_node(b["node_id"], "context_observation", kind=b["kind"], verification="unverified", attrs=b["attrs"])


def test_a_forged_id_or_input_hash_is_refused(store, ep):
    step = write_step(store, ep)
    node = store.build_ctp_node("compliance", {"episode_id": ep, "level": "L1", "judge": "verifier", "value": "PASS", "reason_codes": []}, producer=producer(), input_nodes=(step,))
    with pytest.raises(CtpError, match="not the one derived"):
        store.add_node("compliance:" + "0" * 64, "compliance", kind=node["kind"], verification="unverified", attrs=node["attrs"])
    with pytest.raises(CtpError, match="input_hash is not"):
        store.add_node(node["node_id"], "compliance", kind=node["kind"], verification="unverified", attrs={**node["attrs"], "input_hash": "0" * 64})
    with pytest.raises(CtpError, match="not a stored node"):
        store.add_node(node["node_id"], "compliance", kind=node["kind"], verification="unverified", attrs={**node["attrs"], "input_nodes": ["ghost"]})
    with pytest.raises(CtpError, match="input node 'ghost' is not stored"):
        store.build_ctp_node("compliance", {"episode_id": ep, "level": "L1", "judge": "verifier", "value": "PASS", "reason_codes": []}, producer=producer(), input_nodes=("ghost",))


def test_the_input_hash_covers_node_record_hashes_and_facts(store, ep):
    step = write_step(store, ep)
    f = {"episode_id": ep, "level": "L3", "judge": "verifier", "value": "FAIL", "reason_codes": []}
    base = store.write_ctp_node("compliance", f, producer=producer(), input_nodes=(step,), input_facts={"amount_usd": 800})
    assert store.write_ctp_node("compliance", f, producer=producer(), input_nodes=(step,), input_facts={"amount_usd": 800}) == base  # idempotent
    assert store.write_ctp_node("compliance", f, producer=producer(), input_nodes=(step,), input_facts={"amount_usd": 300}) != base
    assert store.write_ctp_node("compliance", f, producer=producer(), input_nodes=(step, f"hextdoc:{SPEC_HASH}"), input_facts={"amount_usd": 800}) != base
    h = store._record_hash_of(step)
    node = next(n for n in store.nodes() if n["node_id"] == base)
    assert node["attrs"]["input_hash"] == hashlib.sha256(canonicalize(ctp.input_entries({step: h}, {"amount_usd": 800})).encode()).hexdigest()


CHILD = r"""
import json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from graph.store import GraphStore
from graph.blobs import write_blob
s = GraphStore(Path(sys.argv[1]), v01=False, blob_dir=Path(sys.argv[1]))
sha = write_blob(Path(sys.argv[1]), b"the context given to f1")
spec = "sha256:" + "5" * 64
s.add_node("hextdoc:" + spec, "hext_document", attrs={{"hext_kind": "source", "state_hash": spec}})
P = {{"component": "v", "version": "1", "config_sha256": "c" * 64}}
ep = s.write_ctp_node("episode", {{"experiment_spec_hash": spec, "item_id": "i", "bundle_id": "b", "split": "fit", "condition": "a", "replicate": 0, "rng_seed": 1}}, producer=P)
step = s.write_ctp_node("step_output", {{"episode_id": ep, "step": "f1", "text_ref": "blob:" + "3" * 64, "input_context_ref": "blob:" + sha, "input_context_sha256": sha}}, producer=P)
c = s.write_ctp_node("compliance", {{"episode_id": ep, "level": "L1", "judge": "verifier", "value": "PASS", "reason_codes": []}}, producer=P, input_nodes=(step,), input_facts={{"k": 1}})
s.add_edge("interprets", c, step)
print(json.dumps({{"ids": [ep, step, c], "nodes": Path(sys.argv[1], "graph", "nodes.jsonl").read_text(), "edges": Path(sys.argv[1], "graph", "edges.jsonl").read_text()}}))
"""


def test_ids_and_logs_are_identical_in_two_new_processes(tmp_path):
    outs = []
    for i, seed in enumerate(("1", "2")):
        done = subprocess.run([sys.executable, "-c", CHILD.format(root=str(ROOT)), str(tmp_path / f"p{i}")], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed})
        outs.append(json.loads(done.stdout))
    assert outs[0] == outs[1] and outs[0]["ids"][0].startswith("episode:")


# ---- non-deterministic evaluators --------------------------------------------------------------------------
def test_the_same_evaluator_with_the_same_inputs_and_another_result_is_listed_not_refused(store, ep):
    step = write_step(store, ep)
    mk = lambda value, tag="a", facts=None: store.write_ctp_node("compliance", {"episode_id": ep, "level": "L1", "judge": "verifier", "value": value, "reason_codes": []}, producer=producer(tag), input_nodes=(step,), input_facts=facts or {})  # noqa: E731
    assert store.nondeterminism() == []
    a, b = mk("PASS"), mk("FAIL")  # same evaluator, same inputs, another result
    assert a != b and {a, b} <= {n["node_id"] for n in store.nodes()}
    groups = store.nondeterminism()
    assert len(groups) == 1 and groups[0].node_type == "compliance" and groups[0].node_ids == tuple(sorted([a, b]))
    mk("FAIL", tag="other")  # another evaluator: not the same group
    mk("FAIL", facts={"x": 1})  # other inputs: not the same group
    assert len(store.nondeterminism()) == 1 and len(store.nondeterminism()[0].node_ids) == 2
    mk("UNVERIFIABLE")
    assert len(store.nondeterminism()[0].node_ids) == 3


# ---- the whole chain on a real HEXT source -------------------------------------------------------------------
@needs_hext_docs
def test_an_episode_under_a_projected_hext_source_document(tmp_path):
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    heads = layer.manifest()["heads"]
    state_hash = json.loads(SRC.read_bytes())["final_seal"]["state_hash"]
    doc = doc_node_id(state_hash)
    s = GraphStore(tmp_path, v01=False, blob_dir=tmp_path)
    ep = s.write_ctp_node("episode", episode_fields(experiment_spec_hash=state_hash), producer=producer())
    assert s.add_edge("under_manifest", ep, doc)  # the end is in the projection layer's log
    step = s.write_ctp_node("step_output", step_fields(ep, "f1", ctx_blob(s, "p0")), producer=producer())
    assert s.add_edge("has_step", ep, step) and s.add_edge("retrieved", step, doc)
    ce = s.write_ctp_node("carry_evidence", {"episode_id": ep, "step": "f1", "property": "c1.condition", "status": "preserved", "basis": "evidence", "evidence_ref": "fixture:x"}, producer=producer(), input_nodes=(step, doc))
    assert s.add_edge("interprets", ce, step) and s.add_edge("interprets", ce, doc)
    with pytest.raises(CtpError, match="not a stored node|HEXT source"):
        s.write_ctp_node("episode", episode_fields(experiment_spec_hash="sha256:" + "0" * 64), producer=producer())
    assert HextProjectionLayer(tmp_path).manifest()["heads"] == heads  # the projection layer is untouched
    assert s.axis_consistency().ok


# ---- observed_at is for observations only -----------------------------------------------------------------------
def sample_fields(store, ep):
    """One valid field set for every type except episode, and the inputs a judgement needs."""
    step = write_step(store, ep, "f1")
    ctx = write_context(store, ep)
    return step, {
        "context_observation": {"episode_id": ep, "context_ref": "blob:" + "8" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False},
        "step_output": step_fields(ep, "f2", ctx_blob(store, f"sample/{ep}")),
        "applicability": {"episode_id": ep, "constraint_id": "c1", "applicability": "applicable", "applicability_basis": "fixture"},
        "carry_evidence": {"episode_id": ep, "step": "f1", "property": "p", "status": "lost", "basis": "declared", "evidence_ref": None},
        "cle_result": {"episode_id": ep, "rule": "R9", "mode": "M1", "value": "PASS", "reason_codes": [], "locations": [], "history_hash": "h"},
        "compliance": {"episode_id": ep, "level": "L1", "judge": "verifier", "value": "PASS", "reason_codes": []},
        "plan": {"episode_id": ep, "plan_id": "p", "steps": [], "probability_source": "unobserved"},
        "governance": {"episode_id": ep, "plan_id": "p", "decision": "admit", "reason_codes": []},
        "outcome": outcome_fields(ep),
    } | {"_ctx": ctx}


OBSERVATIONS = ("episode", "context_observation", "step_output")
NOT_OBSERVATIONS = ("applicability", "carry_evidence", "cle_result", "compliance", "plan", "governance", "outcome")


def test_observed_at_is_a_field_of_observations_only(store, ep):
    assert all(ctp.SPECS[t].observation for t in OBSERVATIONS) and not any(ctp.SPECS[t].observation for t in NOT_OBSERVATIONS)
    step, fields = sample_fields(store, fresh(store, 50))
    e2 = fields["compliance"]["episode_id"]
    for n, node_type in enumerate(OBSERVATIONS):
        base = {"episode": episode_fields(item_id=f"obs{n}"), "context_observation": fields["context_observation"], "step_output": fields["step_output"]}[node_type]
        for stamp in (None, "2026-11-03T10:00:00Z"):
            node = store.build_ctp_node(node_type, base, producer=producer(), observed_at=stamp)
            assert node["attrs"]["observed_at"] == stamp  # the key is there, null or a label
        assert node["node_id"] == store.build_ctp_node(node_type, base, producer=producer())["node_id"]  # and it is not in the id
        assert store.add_node(node["node_id"], node_type, kind=node["kind"], verification="unverified", attrs=node["attrs"])
    assert e2


@pytest.mark.parametrize("node_type", NOT_OBSERVATIONS)
def test_a_judgement_selection_or_outcome_with_observed_at_is_refused(store, ep, node_type):
    step, fields = sample_fields(store, fresh(store, 60))
    f = fields[node_type]
    with pytest.raises(CtpError, match="observed_at is only for observations"):
        store.build_ctp_node(node_type, f, producer=producer(), input_nodes=(step,), observed_at="2026-11-03T10:00:00Z")
    node = store.build_ctp_node(node_type, f, producer=producer(), input_nodes=(step,))
    assert "observed_at" not in node["attrs"]
    for stamp in ("2026-11-03T10:00:00Z", None):  # a stamp, or even a null one
        with pytest.raises(CtpError, match="observed_at is only for observations"):
            store.add_node(node["node_id"], node_type, kind=node["kind"], verification="unverified", attrs={**node["attrs"], "observed_at": stamp})
    assert store.add_node(node["node_id"], node_type, kind=node["kind"], verification="unverified", attrs=node["attrs"])


@pytest.mark.parametrize("node_type", NOT_OBSERVATIONS)
def test_scoring_again_with_the_same_inputs_and_evaluator_succeeds_idempotently(store, ep, node_type):
    step, fields = sample_fields(store, fresh(store, 70))
    first = store.write_ctp_node(node_type, fields[node_type], producer=producer(), input_nodes=(step,))
    count = len(store.nodes_log)
    head = store.nodes_log.head
    again = store.write_ctp_node(node_type, fields[node_type], producer=producer(), input_nodes=(step,))  # a re-run
    assert again == first and (len(store.nodes_log), store.nodes_log.head) == (count, head)  # nothing was appended
    fresh_store = GraphStore(store.root.parent, v01=False, blob_dir=store.blob_dir)  # and a new process would do the same
    assert fresh_store.write_ctp_node(node_type, fields[node_type], producer=producer(), input_nodes=(step,)) == first
    assert len(fresh_store.nodes_log) == count


# ---- the whole chain, and the search ------------------------------------------------------------------------------
def make_spec_document():
    """A HEXT v2 source for an experiment spec, sealed with the reference implementation (test data)."""
    sys.path.insert(0, str(Path.home() / "GemminAI" / "hext" / "reference" / "python"))
    create = pytest.importorskip("hext2.create")
    t = "2026-11-03T00:00:00Z"
    _doc, sealed = create.create_and_seal(
        object_id="urn:test:exp-rag-ctp001-spec", object_type="document.experiment-spec",
        evidence=[{"evidence_id": "ev-1", "format": "text/plain; charset=utf-8", "body": "EXP-RAG-CTP001: carry a constraint through f1, f2, f3"}],
        contributors=[{"agent": "system:test", "role": "generator"}], primary_generator="ai", human_touch="0.0",
        ledger=[{"t": t, "type": "SYSTEM", "actor": "system:test", "delta": 1}], sealed_at=t,
    )
    return sealed.canonical_bytes


def chain(store, doc, state_hash, item, bundle, two_axis, violation_type):
    """One episode, every record of the first half. Returns the node ids by name."""
    ids = {}
    ep = ids["episode"] = store.write_ctp_node("episode", episode_fields(experiment_spec_hash=state_hash, item_id=item, bundle_id=bundle), producer=producer(), observed_at="2026-11-03T09:00:00Z")
    assert store.add_edge("under_manifest", ep, doc)
    ctx = ids["context"] = write_context(store, ep)
    assert store.add_edge("has_context", ep, ctx)
    for i, step in enumerate(("f1", "f2", "f3")):
        sid = ids[step] = write_step(store, ep, step, sha=ctx_blob(store, f"{item}/{step}"))  # a blob per stage, written first
        assert store.add_edge("has_step", ep, sid)
        if step == "f1":
            assert store.add_edge("retrieved", sid, doc)
    app = ids["applicability"] = store.write_ctp_node("applicability", {"episode_id": ep, "constraint_id": "c1", "applicability": "applicable", "applicability_basis": "fixture"}, producer=producer("app"), input_nodes=(ctx, doc))
    store.add_edge("interprets", app, ctx)
    for step in ("f1", "f2", "f3"):
        for basis, ref, status in (("declared", None, "preserved"), ("evidence", f"fixture:{item}/{step}", "lost" if step != "f1" else "preserved")):
            ce = store.write_ctp_node("carry_evidence", {"episode_id": ep, "step": step, "property": "c1.condition", "status": status, "basis": basis, "evidence_ref": ref}, producer=producer("ver"), input_nodes=(ids[step], doc))
            ids[f"carry-{step}-{basis}"] = ce
            store.add_edge("interprets", ce, ids[step])
    for mode, value in (("M1", "PASS"), ("M2", "FAIL")):
        ids[f"cle-{mode}"] = store.write_ctp_node("cle_result", {"episode_id": ep, "rule": "R9", "mode": mode, "value": value, "reason_codes": [] if value == "PASS" else ["OBLIGATION_LOST_BY_EVIDENCE"], "locations": [] if value == "PASS" else ["f2"], "history_hash": "h" * 8},
                                                  producer=producer("cle"), input_nodes=(ids["carry-f1-declared"], ids["carry-f1-evidence"]), input_facts={"obligations": ["c1.condition"]})
    for level, step in (("L1", "f2"), ("L2", "f2"), ("L3", "f2"), ("f3", "f3")):
        c = ids[f"compliance-{level}"] = store.write_ctp_node("compliance", {"episode_id": ep, "level": level, "judge": "verifier", "value": "FAIL" if level != "L1" else "PASS", "reason_codes": []}, producer=producer("ver"), input_nodes=(ids[step],), input_facts={"amount_usd": 800, "approval_obtained": False})
        store.add_edge("interprets", c, ids[step])
    plan = ids["plan"] = store.write_ctp_node("plan", {"episode_id": ep, "plan_id": f"plan-{item}", "steps": ["f1", "f2", "f3"], "probability_source": "unobserved"}, producer=producer("planner"), input_nodes=(ctx,))
    gov = ids["governance"] = store.write_ctp_node("governance", {"episode_id": ep, "plan_id": f"plan-{item}", "decision": "hold", "reason_codes": ["OBLIGATION_LOST_BY_EVIDENCE"]}, producer=producer("gov"), input_nodes=(plan, ids["cle-M2"]), input_facts={"cancelled": False, "quota": {"active": 0, "spent": 3}})
    assert store.add_edge("selects", gov, plan)
    fields = {"episode_id": ep, "carry": {"f1": "preserved", "f2": "lost", "f3": "lost"}, "constraint_compliance": "FAIL", "two_axis": two_axis}
    if violation_type:
        fields["violation_type"] = violation_type
    out = ids["outcome"] = store.write_ctp_node("outcome", fields, producer=producer("out"), input_nodes=(ids["compliance-f3"], gov))
    assert store.add_edge("outcome_of", out, ep) and store.add_edge("outcome_of", out, gov)
    return ids


def test_a_whole_episode_is_stored_and_found_by_the_search(tmp_path):
    spec = make_spec_document()
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(spec, {})], resolve_session=lambda s: None)
    state_hash = json.loads(spec)["final_seal"]["state_hash"]
    doc = doc_node_id(state_hash)
    store = GraphStore(tmp_path, v01=False, blob_dir=tmp_path)
    one = chain(store, doc, state_hash, "i1", "b1", "LOST_VIOLATED", "constraint_dropped")
    two = chain(store, doc, state_hash, "i2", "b1", "CARRIED_VIOLATED", "constraint_broken")
    three = chain(store, doc, state_hash, "i3", "b2", "LOST_VIOLATED", "constraint_dropped")
    kinds = {n["node_type"] for n in store.nodes()}
    assert set(ctp.CTP_NODE_TYPES) - set(ctp.DRAWING) <= kinds and all(d70_class(n) for n in store.nodes() if n["node_type"] in ctp.CTP_NODE_TYPES)
    # every stage has its blob, and the spec is the projected HEXT source
    assert all(blob_path(tmp_path, n["attrs"]["input_context_sha256"]).exists() for n in store.nodes() if n["node_type"] == "step_output")
    assert store.attrs(doc)["hext_kind"] == "source" and store.attrs(doc)["state_hash"] == state_hash

    ids = lambda found: [r["node_id"] for r in found]  # noqa: E731
    assert ids(store.find_outcomes()) == sorted([one["outcome"], two["outcome"], three["outcome"]])
    assert ids(store.find_outcomes(two_axis="LOST_VIOLATED")) == sorted([one["outcome"], three["outcome"]])
    assert ids(store.find_outcomes(two_axis="CARRIED_VIOLATED")) == [two["outcome"]]
    assert ids(store.find_outcomes(bundle_id="b1")) == sorted([one["outcome"], two["outcome"]])
    assert ids(store.find_outcomes(bundle_id="b2")) == [three["outcome"]]
    assert ids(store.find_outcomes(violation_type="constraint_broken")) == [two["outcome"]]
    assert ids(store.find_outcomes(violation_type="constraint_dropped")) == sorted([one["outcome"], three["outcome"]])
    # combinations
    assert ids(store.find_outcomes(two_axis="LOST_VIOLATED", bundle_id="b1")) == [one["outcome"]]
    assert ids(store.find_outcomes(two_axis="LOST_VIOLATED", bundle_id="b2", violation_type="constraint_dropped")) == [three["outcome"]]
    assert store.find_outcomes(two_axis="CARRIED_VIOLATED", bundle_id="b2") == []
    assert store.find_outcomes(two_axis="LOST_VIOLATED", violation_type="constraint_broken") == []
    assert ids(store.find_outcomes(two_axis=("LOST_VIOLATED", "CARRIED_VIOLATED"), bundle_id=("b1",))) == sorted([one["outcome"], two["outcome"]])
    assert store.find_outcomes(bundle_id="nope") == []
    assert all(r["node_type"] == "outcome" for r in store.find_outcomes())
    assert ids(GraphStore(tmp_path, v01=False, blob_dir=tmp_path).find_outcomes(bundle_id="b1", two_axis="LOST_VIOLATED")) == [one["outcome"]]  # a new store object sees the same

    # a scoring that disagrees with itself: same evaluator, same inputs, another result
    assert store.nondeterminism() == []
    again = store.write_ctp_node("compliance", {"episode_id": one["episode"], "level": "L2", "judge": "verifier", "value": "PASS", "reason_codes": []}, producer=producer("ver"),
                                 input_nodes=(one["f2"],), input_facts={"amount_usd": 800, "approval_obtained": False})
    assert again != one["compliance-L2"]
    groups = store.nondeterminism()
    assert len(groups) == 1 and groups[0].node_type == "compliance" and groups[0].node_ids == tuple(sorted([one["compliance-L2"], again]))
    assert HextProjectionLayer(tmp_path).manifest()["heads"] == layer.manifest()["heads"]  # the projection layer was not touched
    assert store.axis_consistency().ok
