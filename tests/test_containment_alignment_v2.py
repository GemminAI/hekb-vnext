import pytest

from graph.alignment_v2 import SCHEMA, diff_v1_v2, to_v2
from graph.containment import containing_documents, intervals, range_verification

H = "d" * 64


def _edge(doc, traj, lo, hi, ref="r"):
    return {"type": "observation_ref", "src": doc, "dst": traj,
            "attrs": {"seq_from": lo, "seq_to": hi, "ref": ref, "session_id": "s"}}


def test_inclusive_zero_based_interval_and_union_per_document():
    ivs = intervals([_edge("d1", "t", 0, 4), _edge("d1", "t", 3, 9), _edge("d2", "t", 5, 5)])
    assert [c["document"] for c in containing_documents(ivs, "t", 0)] == ["d1"]
    assert [c["document"] for c in containing_documents(ivs, "t", 4)] == ["d1"]  # both ends inclusive
    c = containing_documents(ivs, "t", 3)
    assert len(c) == 1 and c[0]["document"] == "d1" and len(c[0]["intervals"]) == 2  # one fact per (document, token)
    assert [c["document"] for c in containing_documents(ivs, "t", 5)] == ["d1", "d2"]  # several documents, sorted
    assert containing_documents(ivs, "t", 10) == []
    assert containing_documents(ivs, "other", 1) == []


def test_reference_without_seq_names_no_interval():
    e = {"type": "observation_ref", "src": "d1", "dst": "t", "attrs": {"seq_from": None, "seq_to": None, "ref": "r"}}
    assert intervals([e]) == {}


@pytest.mark.parametrize("lo,hi,n,status", [(0, 9, 10, "VERIFIED"), (0, 10, 10, "FAILED"), (-1, 3, 10, "FAILED"),
                                            (5, 4, 10, "FAILED"), (None, None, 10, "NOT_EVALUATED")])
def test_range_verification_statuses(lo, hi, n, status):
    v = range_verification(doc_node="d", trajectory="t", n_steps=n, seq_from=lo, seq_to=hi, ref="r",
                           doc_sha256=H, causal_log_sha256=H)
    assert v["status"] == status and v["rule"]["id"] == "R-VER-DOC-TOKEN-01"


def _v1(i=3):
    return {"alignment_id": "old", "session": "s", "object_id": "t", "space_id": "sp", "token_index": i,
            "hext_ref": {"hext_object_id": "k1", "schema_version": "hext/1.0", "source": "cycle_raw.hext", "document": "ABSENT"},
            "parties_present": {"trajectory": True, "causal_log": True, "hext_reference": True, "hext_document": False},
            "identity_status": "VERIFIED", "temporal_status": "VERIFIED", "semantic_status": "NOT_EVALUATED",
            "rules_used": ["R-ID-01"], "basis": "b", "findings": []}


def test_v2_record_only_renames_and_adds():
    v1 = _v1()
    cont = [{"document": "hextdoc:x", "intervals": [(0, 9, "r")]}]
    v2 = to_v2(v1, containing=cont, resolution="UNRESOLVED", resolution_verification="ver:1",
               doc_meta={"hextdoc:x": {"state_hash": "sha256:x", "hext_object_id": "doc-1"}})
    assert v2["schema"] == SCHEMA and "hext_ref" not in v2 and v2["alignment_id_v1"] == "old"
    assert v2["hext10_reference"]["kind"] == "kernel_wire_object" and v2["hext10_reference"]["document"] == "NOT_STORED"
    assert v2["hext10_reference"]["resolution"] == "UNRESOLVED"
    assert v2["parties_present"]["hext_v2_document"] is True and v2["parties_present"]["hext10_reference"] is True
    assert diff_v1_v2(v1, v2) == []
    v2["identity_status"] = "CONFLICT"
    assert diff_v1_v2(v1, v2) == ["changed:identity_status"]


def test_uncontained_token_has_no_v2_document():
    v2 = to_v2(_v1(), containing=[], resolution="UNRESOLVED", resolution_verification=None, doc_meta={})
    assert v2["hext_v2_documents"] == [] and v2["parties_present"]["hext_v2_document"] is False
