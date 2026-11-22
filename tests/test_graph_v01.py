import json
from pathlib import Path

import pytest

from graph import catalog
from graph.chainlog import ChainLog, ChainLogError
from graph.content import content_id
from graph.model import GraphError
from graph.store import GraphStore
from graph.trace import port_scalar


def _rec(t: int, run: str) -> dict:
    return {
        "token_index": t,
        "seq": t,
        "trajectory_hash": run,
        "ports": {"P18": {"result": {"v_norm": 0.5, "n": t + 1}, "annotations": {"x": run}}},
        "cycle_raw": {"hext": {"object_id": f"h{t}"}},
    }


def test_content_id_ignores_run_identity() -> None:
    a = [_rec(0, "run-A"), _rec(1, "run-A")]
    b = [_rec(0, "run-B"), _rec(1, "run-B")]
    assert content_id("s", a) == content_id("s", b)
    assert content_id("s", a) != content_id("t", a)
    b[1]["ports"]["P18"]["result"]["v_norm"] = 0.6
    assert content_id("s", a) != content_id("s", b)


def test_chain_log_detects_tampering(tmp_path: Path) -> None:
    log = ChainLog(tmp_path / "l.jsonl")
    log.append({"a": 1})
    log.append({"a": 2})
    assert [r["a"] for r in log.records()] == [1, 2]
    text = (tmp_path / "l.jsonl").read_text().replace('"a":1', '"a":9')
    (tmp_path / "l.jsonl").write_text(text)
    with pytest.raises(ChainLogError):
        ChainLog(tmp_path / "l.jsonl")


def test_store_rules_and_catalog_rebuild(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    attrs = {"session_id": "s1", "space_id": "sp", "content_id": "c"}
    assert store.add_node("n1", "trajectory", verification="machine_verified", attrs=attrs)
    assert not store.add_node("n1", "trajectory", verification="machine_verified", attrs=attrs)
    with pytest.raises(GraphError):
        store.add_node("n2", "trajectory", kind="claim", verification="human_verified")
    store.add_node("ps", "policy_set")
    assert store.add_edge("measured_under", "n1", "ps")
    assert not store.add_edge("measured_under", "n1", "ps")
    first = catalog.write(store, tmp_path)
    catalog.path_of(tmp_path).unlink()
    assert catalog.load(GraphStore(tmp_path), tmp_path) == first
    assert json.loads(catalog.path_of(tmp_path).read_text()) == first


def test_port_scalar_rules() -> None:
    assert port_scalar("P18", {"v_norm": 0.25}) == 0.25
    assert port_scalar("P28", {"mean_vector": [3.0, 4.0]}) == 5.0
    assert port_scalar("P15", {"x": 1}) != port_scalar("P15", {"x": 1})  # None scalar -> NaN (as TRACE001)
    assert port_scalar("P05", {"a": [3.0, 4.0]}) == 5.0  # not in the table -> L2 fallback
