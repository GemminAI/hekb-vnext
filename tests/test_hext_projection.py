# ruff: noqa: E501
"""HEXT v2 -> graph projection. Test data: the P0 example documents of GemminAI/hext (v2-draft) and a minimal
synthetic source document built with the reference implementation. These are TEST DATA, not experimental data."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HEXT = Path.home() / "GemminAI" / "hext"
EX = HEXT / "examples" / "v2"
sys.path.insert(0, str(HEXT / "reference" / "python"))
pytest.importorskip("hext2")

from hext2.create import create_and_seal  # noqa: E402

from graph.hext_projection import HextProjectionLayer, doc_node_id, project_document  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from graph.verification import make_verification, write_verification  # noqa: E402

pytestmark = pytest.mark.skipif(not (EX / "source-three-sentences.hext2.json").exists(), reason="P0 examples absent")
H = "c" * 64


def _src() -> bytes:
    return (EX / "source-three-sentences.hext2.json").read_bytes()


def _ann() -> bytes:
    return (EX / "annotation-three-sentences.hext2.json").read_bytes()


def _obs_doc(session: str = "s1", rng: str = "0-9") -> bytes:
    t = "2026-10-15T00:00:00Z"
    _d, s = create_and_seal(
        object_id=f"urn:test:{session}", object_type="document.test",
        evidence=[{"evidence_id": "ev-1", "format": "text/plain; charset=utf-8", "body": "x"}],
        contributors=[{"agent": "system:test", "role": "generator"}], primary_generator="ai", human_touch="0.0",
        ledger=[{"t": t, "type": "SYSTEM", "actor": "system:test", "delta": 1}], sealed_at=t,
        measurement={"observation_refs": [f"urn:gmm:nvs:session:{session}#seq={rng}"], "space_id": "sp"},
    )
    return s.canonical_bytes


def test_source_projection_and_separate_identity_attributes():
    p = project_document(_src(), resolve_session=lambda s: None, hekb_object_id="hekb-obj-1", validated_level=2)
    sh = json.loads(_src())["final_seal"]["state_hash"]
    doc = next(n for n in p["nodes"] if n["node_type"] == "hext_document")
    assert doc["node_id"] == doc_node_id(sh) == "hextdoc:" + sh
    a = doc["attrs"]
    assert a["hext_object_id"] == "hext2-three-sentences" and a["state_hash"] == sh and a["hekb_object_id"] == "hekb-obj-1"
    assert len({a["hext_object_id"], a["state_hash"], a["hekb_object_id"]}) == 3
    claims = [n for n in p["nodes"] if n["node_type"] == "hext_claim"]
    assert len(claims) == 3 and all(c["node_id"].startswith(f"hextclaim:{sh}#t") for c in claims)
    assert sum(e["type"] == "has_claim" for e in p["edges"]) == 3


def test_only_hext_derived_node_types_are_created():
    for raw in (_src(), _ann(), _obs_doc()):
        p = project_document(raw, resolve_session=lambda s: "traj-1", known_doc_hashes=[])
        assert {n["node_type"] for n in p["nodes"]} <= {"hext_document", "hext_claim"}
        assert {e["type"] for e in p["edges"]} <= {"has_claim", "time_relation", "observation_ref", "derived_from"}


def test_annotation_relations_and_derived_from():
    src_hash = json.loads(_src())["final_seal"]["state_hash"]
    p = project_document(_ann(), resolve_session=lambda s: None, known_doc_hashes=[src_hash],
                         known_claims={src_hash: {"t1", "t2", "t3"}})
    rel = [e for e in p["edges"] if e["type"] == "time_relation"]
    assert len(rel) == 2 and rel[0]["src"].startswith(f"hextclaim:{src_hash}#")
    assert [e["type"] for e in p["edges"] if e["type"] == "derived_from"] == ["derived_from"] and not p["unresolved"]
    q = project_document(_ann(), resolve_session=lambda s: None, known_doc_hashes=[])
    assert not any(e["type"] == "derived_from" for e in q["edges"])
    assert {u["kind"] for u in q["unresolved"]} == {"derived_from", "relation_endpoint"}
    assert not any(e["type"] == "time_relation" for e in q["edges"])


def test_unresolved_observation_ref_makes_no_edge():
    ok = project_document(_obs_doc("s1"), resolve_session={"s1": "traj-1"}.get)
    e = [x for x in ok["edges"] if x["type"] == "observation_ref"]
    assert len(e) == 1 and e[0]["dst"] == "traj-1" and e[0]["attrs"]["seq_from"] == 0 and e[0]["attrs"]["seq_to"] == 9
    bad = project_document(_obs_doc("zz"), resolve_session={"s1": "traj-1"}.get)
    assert not any(x["type"] == "observation_ref" for x in bad["edges"])
    assert bad["unresolved"] == [{"kind": "observation_ref", "document": bad["document"], "ref": "urn:gmm:nvs:session:zz#seq=0-9",
                                  "session_id": "zz", "reason": "no stored trajectory with this session id in the graph"}]


def test_rebuild_is_identical_and_keeps_non_hext_records(tmp_path):
    docs = [(_src(), {}), (_ann(), {}), (_obs_doc("s1"), {"hekb_object_id": "o1"})]
    layer = HextProjectionLayer(tmp_path)
    m1 = layer.rebuild(docs, resolve_session={"s1": "traj-1"}.get)
    # a Verification lives in the main store; rebuilding the projection must not touch it
    store = GraphStore(tmp_path, v01=False)
    store.add_node("traj-1", "trajectory", kind="observation", verification="unverified", attrs={})
    doc_id = next(n["node_id"] for n in layer.nodes() if n["node_type"] == "hext_document" and n["attrs"]["hext_object_id"] == "urn:test:s1")
    ver = make_verification(rule_id="R-VER-DOC-TRAJ-01", subject_a=doc_id, subject_b="traj-1", status="VERIFIED",
                            verified_means="session id equal", inputs=[{"role": "hext_document", "sha256": H}])
    write_verification(store, ver, known_ids={doc_id})
    nodes_before = [json.dumps(n, sort_keys=True) for n in store.nodes()]
    shutil_rmtree = __import__("shutil").rmtree
    shutil_rmtree(layer.root)  # delete the projection ...
    m2 = layer.rebuild(list(reversed(docs)), resolve_session={"s1": "traj-1"}.get)  # ... and rebuild (input order differs)
    assert m1 == m2
    assert [json.dumps(n, sort_keys=True) for n in GraphStore(tmp_path, v01=False).nodes()] == nodes_before
    assert any(n["node_type"] == "verification" for n in GraphStore(tmp_path, v01=False).nodes())


def test_projection_is_identical_in_two_new_processes(tmp_path):
    code = (
        "import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from graph.hext_projection import HextProjectionLayer\n"
        "d=Path(%r); docs=[(Path(%r).read_bytes(),{}),(Path(%r).read_bytes(),{})]\n"
        "m=HextProjectionLayer(d).rebuild(docs, resolve_session=lambda s: None)\n"
        "print(json.dumps(m['heads'], sort_keys=True))\n"
    )
    outs = []
    for i in range(2):
        r = subprocess.run(
            [sys.executable, "-c", code % (str(Path(__file__).resolve().parent.parent), str(HEXT / "reference" / "python"),
                                           str(tmp_path / f"p{i}"), str(EX / "source-three-sentences.hext2.json"),
                                           str(EX / "annotation-three-sentences.hext2.json"))],
            capture_output=True, text=True, check=True, env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
        outs.append(r.stdout.strip())
    assert outs[0] == outs[1] and json.loads(outs[0])["nodes"]


# ---- full-content determinism and input-order independence --------------------------------------------
def _dump(layer: HextProjectionLayer) -> dict:
    """Every stored record, in full (node id, type, kind, verification and ALL attrs; edge type, src, dst and ALL
    attrs; unresolved records), plus the manifest. Not only the chain heads."""
    return {"nodes": layer.nodes(), "edges": layer.edges(), "unresolved": layer.unresolved(), "manifest": layer.manifest()}


def _canon(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def test_source_and_annotation_order_does_not_change_the_final_projection(tmp_path):
    resolve = {"s1": "traj-1"}.get
    a, b = (_src(), {"hekb_object_id": "o-src"}), (_ann(), {"hekb_object_id": "o-ann"})
    third = (_obs_doc("s1"), {"hekb_object_id": "o-obs"})
    layers = {}
    for name, order in {"src_first": [a, b, third], "ann_first": [b, a, third], "reversed": [third, b, a]}.items():
        layer = HextProjectionLayer(tmp_path / name)
        layer.rebuild(order, resolve_session=resolve)
        layers[name] = _dump(layer)
    base = _canon(layers["src_first"])
    assert _canon(layers["ann_first"]) == base and _canon(layers["reversed"]) == base
    d = layers["ann_first"]
    # the annotation's relations resolved to the SOURCE's claims, whatever the order of arrival
    src_hash = json.loads(_src())["final_seal"]["state_hash"]
    rel = [e for e in d["edges"] if e["type"] == "time_relation"]
    assert len(rel) == 2 and all(e["src"].startswith(f"hextclaim:{src_hash}#") and e["dst"].startswith(f"hextclaim:{src_hash}#") for e in rel)
    assert any(e["type"] == "derived_from" and e["dst"] == f"hextdoc:{src_hash}" for e in d["edges"])
    assert d["unresolved"] == []
    # same content as unordered sets as well
    for kind in ("nodes", "edges"):
        assert {_canon(x) for x in layers["src_first"][kind]} == {_canon(x) for x in layers["reversed"][kind]}


def test_late_arrival_of_the_source_gives_the_same_projection_as_having_it_from_the_start(tmp_path):
    resolve = {"s1": "traj-1"}.get
    both = [(_src(), {}), (_ann(), {})]
    only_ann = HextProjectionLayer(tmp_path / "early")
    only_ann.rebuild([(_ann(), {})], resolve_session=resolve)
    partial = _dump(only_ann)
    assert {u["kind"] for u in partial["unresolved"]} == {"derived_from", "relation_endpoint"}
    assert not any(e["type"] in ("time_relation", "derived_from") for e in partial["edges"])
    only_ann.rebuild(both, resolve_session=resolve)  # the source arrives later; the projection is rebuilt
    fresh = HextProjectionLayer(tmp_path / "fresh")
    fresh.rebuild(list(reversed(both)), resolve_session=resolve)
    assert _canon(_dump(only_ann)) == _canon(_dump(fresh))


def test_full_content_is_identical_in_two_new_processes(tmp_path):
    code = (
        "import sys, json; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from graph.hext_projection import HextProjectionLayer\n"
        "docs=[(Path(%r).read_bytes(),{'hekb_object_id':'o1'}),(Path(%r).read_bytes(),{'hekb_object_id':'o2'})]\n"
        "l=HextProjectionLayer(Path(%r)); l.rebuild(docs, resolve_session=lambda s: None)\n"
        "print(json.dumps({'nodes':l.nodes(),'edges':l.edges(),'unresolved':l.unresolved(),'manifest':l.manifest()},"
        " sort_keys=True, ensure_ascii=False))\n"
    )
    outs = []
    for i in range(2):
        r = subprocess.run(
            [sys.executable, "-c", code % (str(Path(__file__).resolve().parent.parent), str(HEXT / "reference" / "python"),
                                           str(EX / "annotation-three-sentences.hext2.json"),
                                           str(EX / "source-three-sentences.hext2.json"), str(tmp_path / f"q{i}"))],
            capture_output=True, text=True, check=True, env={"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin"})
        outs.append(r.stdout)
    assert outs[0] == outs[1]
    full = json.loads(outs[0])
    assert full["nodes"] and full["edges"]
    # attributes and provenance are part of what was compared
    doc = next(n for n in full["nodes"] if n["node_type"] == "hext_document" and n["attrs"]["hext_kind"] == "annotation")
    assert doc["attrs"]["contributors"] and doc["attrs"]["agency"] and doc["attrs"]["hekb_object_id"] == "o1"


# ---- what blob_sha256 covers, and what happens with duplicates ----------------------------------------------
import hashlib  # noqa: E402

from graph.hext_projection import ProjectionConflict, sealed_bytes  # noqa: E402
from graph.model import GraphError  # noqa: E402


def test_blob_sha256_is_over_the_sealed_canonical_bytes():
    raw = _src()  # the P0 file carries one trailing newline that is not part of the sealed bytes
    assert raw.endswith(b"\n") and not sealed_bytes(raw).endswith(b"\n")
    p = project_document(raw, resolve_session=lambda s: None)
    expect = hashlib.sha256(raw[:-1]).hexdigest()
    assert p["nodes"][0]["attrs"]["blob_sha256"] == expect
    # with or without the newline: the same identity
    q = project_document(raw[:-1], resolve_session=lambda s: None)
    assert q["nodes"][0]["attrs"]["blob_sha256"] == expect


def test_non_canonical_or_tampered_input_is_refused_not_reserialised():
    doc = json.loads(_src())
    pretty = json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")
    with pytest.raises(GraphError, match="not the sealed canonical form"):
        project_document(pretty, resolve_session=lambda s: None)
    bad = json.loads(_src())
    bad["final_seal"]["state_hash"] = "sha256:" + "0" * 64
    from graph.hext_projection import _enc

    with pytest.raises(GraphError, match="state_hash does not match"):
        project_document(_enc(bad).encode("utf-8"), resolve_session=lambda s: None)


def test_identical_duplicates_are_merged_and_counted(tmp_path):
    layer = HextProjectionLayer(tmp_path)
    one = layer.rebuild([(_src(), {"hekb_object_id": "o"})], resolve_session=lambda s: None)
    layer2 = HextProjectionLayer(tmp_path / "dup")
    two = layer2.rebuild([(_src(), {"hekb_object_id": "o"}), (_src()[:-1], {"hekb_object_id": "o"})], resolve_session=lambda s: None)
    assert two["duplicates_merged"] == 1 and one["duplicates_merged"] == 0
    assert (two["nodes"], two["edges"], two["heads"]) == (one["nodes"], one["edges"], one["heads"])


def test_same_bytes_with_different_metadata_is_a_conflict_and_leaves_the_layer_intact(tmp_path):
    layer = HextProjectionLayer(tmp_path)
    good = layer.rebuild([(_src(), {"hekb_object_id": "o1"})], resolve_session=lambda s: None)
    with pytest.raises(ProjectionConflict, match="different metadata"):
        layer.rebuild([(_src(), {"hekb_object_id": "o1"}), (_src(), {"hekb_object_id": "o2"})], resolve_session=lambda s: None)
    assert layer.manifest() == good  # the previous projection was not replaced


def test_same_state_hash_with_different_bytes_is_a_conflict(tmp_path):
    from graph.hext_projection import _enc

    signed = json.loads(_src())
    signed["final_seal"]["signature"] = {"alg": "ecdsa-p256-sha256", "key_id": "k", "value": "v"}  # not part of state_hash
    variant = _enc(signed).encode("utf-8")
    assert json.loads(variant)["final_seal"]["state_hash"] == json.loads(_src())["final_seal"]["state_hash"]
    with pytest.raises(ProjectionConflict, match="same state_hash with different bytes"):
        HextProjectionLayer(tmp_path).rebuild([(_src(), {}), (variant, {})], resolve_session=lambda s: None)


def test_same_object_id_with_different_state_hash_are_two_versions_not_a_conflict(tmp_path):
    from graph.hext_projection import _enc

    v2 = json.loads(_src())
    v2["evidence"][0]["body"] = v2["evidence"][0]["body"] + "。"  # a different document with the same object.id
    v2["final_seal"]["state_hash"] = "sha256:" + "0" * 64
    from graph.hext_projection import _state_hash

    v2["final_seal"]["state_hash"] = _state_hash(v2)
    m = HextProjectionLayer(tmp_path).rebuild([(_src(), {}), (_enc(v2).encode("utf-8"), {})], resolve_session=lambda s: None)
    assert m["documents"] == 2
    assert list(m["same_object_id_versions"]) == ["hext2-three-sentences"]
    assert len(m["same_object_id_versions"]["hext2-three-sentences"]) == 2
