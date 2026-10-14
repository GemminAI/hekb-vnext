import json

import pytest

from graph.model import GraphError
from graph.store import GraphStore
from graph.verification import (
    epistemic_status,
    make_verification,
    rule_ref,
    unresolved_reverse_index,
    write_verification,
)

H = "a" * 64
H2 = "b" * 64


def _store(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    s.add_node("cl", "causal_log", kind="observation", verification="unverified", attrs={})
    s.add_node("hext:x", "hext_kernel", kind="observation", verification="unverified", attrs={})
    s.add_node("doc", "hext_document", kind="observation", verification="unverified", attrs={"epistemic_status": "derived"})
    return s


def _v(status="VERIFIED", **kw):
    base = dict(rule_id="R-VER-ID-HEXT10-01", subject_a="cl", subject_b="hext:x", status=status,
                verified_means="identifier match", inputs=[{"role": "causal_log", "sha256": H}])
    base.update(kw)
    return make_verification(**base)


def test_status_requirements():
    assert _v()["epistemic_status"] == "verified_relation"
    with pytest.raises(GraphError):
        _v(status="UNRESOLVED")  # needs a reason
    with pytest.raises(GraphError):
        _v(status="NOT_EVALUATED", verified_means="")
    with pytest.raises(GraphError):
        _v(verified_means=" ")  # VERIFIED needs what was confirmed
    with pytest.raises(GraphError):
        _v(inputs=[])
    with pytest.raises(GraphError):
        _v(inputs=[{"role": "x", "sha256": "zz"}])
    with pytest.raises(GraphError):
        _v(status="MAYBE")


def test_id_is_deterministic_and_binds_rule_inputs_and_status():
    a, b = _v(), _v()
    assert a == b
    assert _v(inputs=[{"role": "causal_log", "sha256": H2}])["verification_id"] != a["verification_id"]
    assert _v(status="FAILED", reason="x")["verification_id"] != a["verification_id"]
    assert rule_ref("R-VER-ID-HEXT10-01")["sha256"] == rule_ref("R-VER-ID-HEXT10-01")["sha256"]


def test_existing_nodes_are_not_modified_and_subjects_must_exist(tmp_path):
    s = _store(tmp_path)
    before = [json.dumps(n, sort_keys=True) for n in s.nodes()]
    write_verification(s, _v())
    after = [json.dumps(n, sort_keys=True) for n in s.nodes()]
    assert after[: len(before)] == before  # appended only
    with pytest.raises(GraphError):
        write_verification(s, _v(subject_b="hext:missing"))


def test_derived_stays_derived_when_verified(tmp_path):
    s = _store(tmp_path)
    ver = make_verification(rule_id="R-VER-DOC-TRAJ-01", subject_a="doc", subject_b="cl", status="VERIFIED",
                            verified_means="session id equal", inputs=[{"role": "hext_document", "sha256": H}])
    write_verification(s, ver)
    nodes = {n["node_id"]: n for n in s.nodes()}
    assert epistemic_status(nodes["doc"]) == "derived"
    assert epistemic_status(nodes[ver["verification_id"]]) == "verified_relation"
    assert epistemic_status(nodes["cl"]) == "observed"


def test_identity_verified_and_reference_unresolved_are_separate_records(tmp_path):
    s = _store(tmp_path)
    ident = _v()
    res = make_verification(rule_id="R-VER-REF-RESOLVE-01", subject_a="cl", subject_b="hext:x", status="UNRESOLVED",
                            verified_means="", inputs=[{"role": "causal_log", "sha256": H}], reason="not confirmed in scope",
                            detail={"session": "s1", "token_index": 3, "target_id": "x", "search_scope": {"ids": 1}})
    write_verification(s, ident)
    write_verification(s, res)
    assert ident["verification_id"] != res["verification_id"]
    rev = unresolved_reverse_index(s)
    assert list(rev) == ["x"] and rev["x"][0]["session"] == "s1" and rev["x"][0]["token_index"] == 3
    # rebuilding from the same log gives the same index
    assert unresolved_reverse_index(GraphStore(tmp_path, v01=False)) == rev
