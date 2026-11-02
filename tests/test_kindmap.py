# ruff: noqa: E501
"""CTP data model v0.3.3 section 1: the stored-kind rule and the D-70 classification of every node_type.

Writes are checked against the ``stored kind`` column of ``graph.kindmap.KIND_MAP`` on both entrances
(``GraphStore`` and the HEXT projection layer); reads classify from ``node_type`` only. Tests that need a
HEXT document use the P0 examples of GemminAI/hext (test data) and are skipped when they are absent; the
real-data regression is skipped when the Mac's data dirs are absent.
"""

import json
import re
from pathlib import Path

import pytest

from graph import hext_projection, model
from graph.chainlog import ChainLog
from graph.guard import CollisionError
from graph.hext_projection import HextProjectionLayer, project_document
from graph.kindmap import D70_CLASSES, KIND_MAP, KindMapError, check_stored_kind, d70_class
from graph.model import KINDS, NODE_TYPES, GraphError
from graph.store import GraphStore
from graph.verification import make_verification, write_verification

ROOT = Path(__file__).resolve().parent.parent
EX = Path.home() / "GemminAI" / "hext" / "examples" / "v2"
SRC = EX / "source-three-sentences.hext2.json"
ANN = EX / "annotation-three-sentences.hext2.json"
needs_hext_docs = pytest.mark.skipif(not SRC.exists(), reason="P0 example documents absent")

# v0.3.3 section 1 plus the four rows Tomo decided (2026-11-03, option A), written out independently of the module.
EXPECTED = {
    "trajectory": ("observation", "observation"),
    "hext_document": ("observation", "evidence"),
    "hext_claim": ("observation", "evidence"),
    "hext_kernel": ("observation", "evidence"),
    "verification": ("evaluation", "interpretation"),
    "causal_log": ("observation", "observation"),
    "source_text": ("observation", "observation"),
    "policy_set": ("observation", "observation"),
    "raw_log": ("observation", "observation"),
}


def test_the_table_is_the_v033_table_and_covers_every_node_type():
    assert {t: (r.stored_kind, r.d70_class) for t, r in KIND_MAP.items()} == EXPECTED
    assert set(KIND_MAP) == set(NODE_TYPES)
    assert all(r.d70_class in D70_CLASSES and r.stored_kind in KINDS for r in KIND_MAP.values())


@pytest.mark.parametrize("node_type", ["causal_log", "raw_log", "source_text", "policy_set"])
def test_the_four_rows_decided_by_tomo_are_observations(tmp_path, node_type):
    """Tomo, 2026-11-03, option A: D-70 Evidence is HEXT only; these four are not listed in v0.3.3 s1."""
    rule = KIND_MAP[node_type]
    assert (rule.stored_kind, rule.d70_class) == ("observation", "observation")
    assert rule.source.startswith("Tomo decision 2026-11-03 (option A: D-70 Evidence is HEXT only")
    assert "to confirm" not in rule.source
    s = GraphStore(tmp_path, v01=False)
    assert s.add_node("n", node_type, kind="observation") is True
    assert d70_class(s.nodes()[0]) == "observation"
    with pytest.raises(KindMapError):
        s.add_node("m", node_type, kind="evaluation")
    assert {t for t, r in KIND_MAP.items() if r.d70_class == "evidence"} == {"hext_document", "hext_claim", "hext_kernel"}


@pytest.mark.parametrize("node_type", sorted(EXPECTED))
def test_a_write_with_the_stored_kind_passes_and_classifies(tmp_path, node_type):
    stored, cls = EXPECTED[node_type]
    s = GraphStore(tmp_path, v01=False)
    assert s.add_node("n", node_type, kind=stored, verification="unverified") is True
    assert d70_class(s.nodes()[0]) == cls


@pytest.mark.parametrize("node_type", sorted(EXPECTED))
def test_a_write_with_another_kind_is_refused_and_nothing_is_written(tmp_path, node_type):
    stored = EXPECTED[node_type][0]
    s = GraphStore(tmp_path, v01=False)
    for kind in (k for k in KINDS if k != stored):
        with pytest.raises(KindMapError) as caught:
            s.add_node("n", node_type, kind=kind, verification="unverified")
        assert (caught.value.node_id, caught.value.node_type, caught.value.kind) == ("n", node_type, kind)
        assert isinstance(caught.value, GraphError)
    assert len(s.nodes_log) == 0


def test_an_unknown_node_type_is_refused(tmp_path, monkeypatch):
    s = GraphStore(tmp_path, v01=False)
    with pytest.raises(GraphError):  # validate_node: not in NODE_TYPES
        s.add_node("n", "mystery")
    monkeypatch.setattr(model, "NODE_TYPES", (*NODE_TYPES, "mystery"))  # ... and if it were added there, the table still refuses
    with pytest.raises(KindMapError, match="not in the kind map"):
        s.add_node("n", "mystery")
    with pytest.raises(KindMapError):
        d70_class({"node_id": "n", "node_type": "mystery", "kind": "observation"})
    assert len(s.nodes_log) == 0


def test_the_v01_refusals_are_unchanged(tmp_path):
    s = GraphStore(tmp_path)  # v01=True
    with pytest.raises(GraphError, match="v0.1 writes only"):
        s.add_node("n", "trajectory", kind="claim", verification="human_verified")
    assert s.add_node("n", "trajectory", verification="machine_verified") is True


def test_a_stored_node_is_not_a_new_write_so_a_changed_kind_is_a_collision(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    s.add_node("n", "trajectory")
    with pytest.raises(CollisionError):  # the id exists with another content: reported as D13 does
        s.add_node("n", "trajectory", kind="evaluation")
    with pytest.raises(KindMapError):  # a new id with the wrong kind is the kind map's refusal
        s.add_node("m", "trajectory", kind="evaluation")


# ---- classification of records written before D-70 -----------------------------------------------------
@pytest.mark.parametrize(
    ("node", "cls"),
    [
        ({"node_id": "a", "node_type": "verification", "kind": "evaluation"}, "interpretation"),
        ({"node_id": "b", "node_type": "hext_document", "kind": "observation"}, "evidence"),
        ({"node_id": "c", "node_type": "hext_claim", "kind": "observation"}, "evidence"),
        ({"node_id": "d", "node_type": "hext_kernel", "kind": "observation"}, "evidence"),
        ({"node_id": "e", "node_type": "trajectory", "kind": "observation"}, "observation"),
        ({"node_id": "f", "node_type": "trajectory", "kind": "claim"}, "observation"),  # the kind is not read
    ],
)
def test_existing_records_classify_by_node_type(node, cls):
    assert d70_class(node) == cls


def test_a_real_verification_record_is_an_interpretation(tmp_path):
    s = GraphStore(tmp_path, v01=False)
    s.add_node("t", "trajectory")
    s.add_node("d", "hext_document")
    ver = make_verification(rule_id="R-VER-DOC-TRAJ-01", subject_a="d", subject_b="t", status="VERIFIED",
                            verified_means="session id equal", inputs=[{"role": "hext_document", "sha256": "c" * 64}])
    write_verification(s, ver)
    node = next(n for n in s.nodes() if n["node_type"] == "verification")
    assert node["kind"] == "evaluation" and d70_class(node) == "interpretation"


def test_no_writer_uses_claim():
    pattern = re.compile(r"""kind\s*=\s*["']claim["']|["']kind["']\s*:\s*["']claim["']""")
    found = [str(p.relative_to(ROOT)) for d in ("graph", "tools", "scripts") for p in (ROOT / d).rglob("*.py") if pattern.search(p.read_text())]
    assert found == []


# ---- the HEXT projection entrance ----------------------------------------------------------------------
@needs_hext_docs
def test_what_the_projection_writes_passes_the_kind_map(tmp_path):
    for doc in (SRC, ANN):
        for node in project_document(doc.read_bytes(), resolve_session=lambda s: None)["nodes"]:
            check_stored_kind(node)
            assert d70_class(node) == "evidence"
    layer = HextProjectionLayer(tmp_path)
    layer.rebuild([(SRC.read_bytes(), {}), (ANN.read_bytes(), {})], resolve_session=lambda s: None)
    assert {d70_class(n) for n in layer.nodes()} == {"evidence"}


@needs_hext_docs
def test_the_projection_refuses_a_node_with_another_kind(tmp_path, monkeypatch):
    real = hext_projection.project_document

    def wrong_kind(raw, **kw):
        out = real(raw, **kw)
        out["nodes"][0] = {**out["nodes"][0], "kind": "evidence"}
        return out

    monkeypatch.setattr(hext_projection, "project_document", wrong_kind)
    layer = HextProjectionLayer(tmp_path)
    with pytest.raises(KindMapError):
        layer.rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)
    assert not layer.root.exists() and not layer.root.with_name("hext_projection.tmp").exists()


@needs_hext_docs
def test_the_projection_refuses_an_unknown_node_type(tmp_path, monkeypatch):
    real = hext_projection.project_document

    def unknown(raw, **kw):
        out = real(raw, **kw)
        out["nodes"][0] = {**out["nodes"][0], "node_type": "mystery"}
        return out

    monkeypatch.setattr(hext_projection, "project_document", unknown)
    with pytest.raises(KindMapError, match="not in the kind map"):
        HextProjectionLayer(tmp_path).rebuild([(SRC.read_bytes(), {})], resolve_session=lambda s: None)


# ---- regression on the real data dirs (copies in tmp_path; the originals are only read) -----------------
EV = Path.home() / "vaults" / "20260124" / "experiment-evidence" / "GRAPH-ALIGN001"
REAL = [EV / "graph_data", EV / "graph_data_v2", Path("/Volumes/SSD1TB/graph002_work/data"),
        Path("/Volumes/SSD1TB/graph002_work/data_samecontent"), Path("/Volumes/SSD1TB/d71_work/hekb_lib")]


@pytest.mark.parametrize("src", REAL, ids=lambda p: p.name)
def test_real_data_rewrites_byte_identically_through_the_kind_map(tmp_path, src):
    if not (src / "graph" / "nodes.jsonl").exists():
        pytest.skip("real data dir absent")
    out = GraphStore(tmp_path / "copy", v01=False)
    for line in (src / "graph" / "nodes.jsonl").open():
        r = json.loads(line)["record"]
        out.add_node(r["node_id"], r["node_type"], kind=r["kind"], verification=r["verification"], attrs=r["attrs"])
    for line in (src / "graph" / "edges.jsonl").open():
        r = json.loads(line)["record"]
        out.add_edge(r["type"], r["src"], r["dst"], r.get("attrs", {}))
    for name in ("nodes.jsonl", "edges.jsonl"):
        assert (tmp_path / "copy" / "graph" / name).read_bytes() == (src / "graph" / name).read_bytes()
    layer = src / "graph" / "hext_projection" / "nodes.jsonl"
    if layer.exists():
        for rec in ChainLog(layer).records():
            check_stored_kind(rec)
