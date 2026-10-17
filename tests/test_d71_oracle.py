"""ORACLE001 harness: no GPU, mock backend (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from d71 import hf_provider, oracle, stage3

OBS = "Run oracle-aaaaaaaa (node 0123456789ab, summarize_v1).\n- mean_step (P03): median 1.0, max 1.2 at token 3 (\"the\").\n"


class MockBackend:
    def __init__(self, ids: list[int], eos: int = 128009) -> None:
        self._ids, self._eos = ids, eos

    def eos_ids(self) -> list[int]:
        return [self._eos, 128001]

    def describe(self) -> dict[str, str]:
        return {"backend": "mock"}

    def generate(self, chat, *, seed, max_new_tokens, min_new_tokens):  # type: ignore[no-untyped-def]
        n = len(self._ids)
        return SimpleNamespace(hidden=np.full((n, 1, 4), seed % 7, dtype=np.float32), ids=np.array(self._ids),
                               logprob=np.zeros(n, np.float32), entropy=np.ones(n, np.float32), text="x")


def _provider(ids: list[int]) -> hf_provider.Llama31HFProvider:
    b = MockBackend(ids)
    return hf_provider.Llama31HFProvider(backend_factory=lambda m, r: b)


ARMS = ("O_same_correct", "O_kind", "Cprime_r")


def _rows(items: list[str], block: str, version: str = "deficient") -> list[dict[str, object]]:
    rows = []
    for pid in items:
        for arm in ARMS:
            for s in (0, 1, 2):
                user = f"{pid}|{arm}|{version}|{s}"
                rows.append({"arm": arm, "version": version, "prompt_id": pid, "sample": s, "sources": ["a", "b", "c"],
                             "obs_tokens": 10, "obs_tokens_Cprime": 10, "context_tokens": 100,
                             "context_sha256": stage3.sha(user), "user": user, "block": block, "kind": "pow2",
                             "difficulty": "easy", "removed": "n", "expected": None, "leak_E": "1024", "leak_R": "7",
                             "obs_texts": [OBS, OBS, OBS]})
    return rows


def _bundle(tmp: Path, n1: int = 2, n2: int = 1) -> Path:
    b = tmp / "bundle"
    b.mkdir()
    files = {}
    for block, items in (("block1", [f"i{n}" for n in range(n1)]), ("block2", [f"j{n}" for n in range(n2)])):
        p = b / f"targets_{block}.jsonl"
        p.write_text("".join(json.dumps(r) + "\n" for r in _rows(items, block)))
        files[p.name] = oracle._sha_file(p)
    (b / "system_prompt.txt").write_text("sys")
    files["system_prompt.txt"] = oracle._sha_file(b / "system_prompt.txt")
    (b / "manifest.json").write_text(json.dumps({"files": files, "leak_n_residual": 0}))
    return b


def test_seed_is_stage3_formula_and_design_is_deterministic(tmp_path: Path) -> None:
    rows = _rows(["i0", "i1"], "block1")
    d1, d2 = oracle.build_design(rows), oracle.build_design(rows)
    assert d1 == d2 and [d["idx"] for d in d1] == list(range(len(d1))) == list(range(18))
    assert all(d["seed"] == stage3.seed_for(d["prompt_id"], d["version"], d["sample"]) for d in d1)
    assert {d["seed"] for d in d1 if d["prompt_id"] == "i0" and d["sample"] == 1} == {stage3.seed_for("i0", "deficient", 1)}  # same seed across arms
    assert {d["cond"] for d in d1} == set(ARMS)


def test_load_bundle_detects_sha_leak_and_file_tamper(tmp_path: Path) -> None:
    b = _bundle(tmp_path)
    assert set(oracle.load_bundle(b)) == {"block1", "block2"}
    p = b / "targets_block1.jsonl"
    p.write_text(p.read_text().replace("i0|O_kind", "i0|O_kinD", 1))
    with pytest.raises(RuntimeError, match="differs from manifest"):
        oracle.load_bundle(b)
    # user text changed AND manifest re-pinned -> per-target sha256 still catches it
    man = json.loads((b / "manifest.json").read_text())
    man["files"][p.name] = oracle._sha_file(p)
    (b / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match="context sha256 mismatch"):
        oracle.load_bundle(b)
    # leak: an observation whose fragment overlaps the expected digits
    b2 = _bundle(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path)
    leaky = OBS.replace('("the")', '("1024")')
    rows = _rows(["i0"], "block1")
    for r in rows:
        r["obs_texts"] = [leaky] * 3
    p2 = b2 / "targets_block1.jsonl"
    p2.write_text("".join(json.dumps(r) + "\n" for r in rows))
    man = json.loads((b2 / "manifest.json").read_text())
    man["files"][p2.name] = oracle._sha_file(p2)
    (b2 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(RuntimeError, match="leak residual"):
        oracle.load_bundle(b2)


def test_leak_rules_a_b_c() -> None:
    assert oracle.leak_hits(OBS.replace('("the")', '("a399")'), "2939", "5", "nth_prime") == ["a"]
    assert oracle.leak_hits(OBS.replace("token 3", "token 19"), "19", "5", "remainder") == ["b"]
    assert oracle.leak_hits(OBS.replace('("the")', '("Mon")'), "x", "y", "day_of_week") == ["c"]
    assert oracle.leak_hits(OBS, "1024", "7", "pow2") == []


def test_dry_run_counts_and_expected_totals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(oracle, "_version_fingerprint", lambda: {"version": "t"})
    b = _bundle(tmp_path, n1=2, n2=1)
    monkeypatch.setitem(oracle.EXPECTED_TARGETS, "block1", 18)
    monkeypatch.setitem(oracle.EXPECTED_TARGETS, "block2", 9)
    s = oracle.dry_run(b, tmp_path / "out")
    assert s["block1"]["n"] == 18 and s["block2"]["n"] == 9 and (tmp_path / "out" / "dry_run.json").exists()
    monkeypatch.setitem(oracle.EXPECTED_TARGETS, "block1", 684)
    with pytest.raises(RuntimeError, match="expected 684"):
        oracle.dry_run(b)


def test_gates_block_order_and_status(tmp_path: Path) -> None:
    rows = _rows(["i0"], "block1")
    gates = tmp_path / "g"
    out = gates / "block1"
    p = _provider([5, 128009])
    assert oracle.run_block(rows, "block1", p, out, gates, "sys") == oracle.EXIT_GATE  # nothing passed yet
    (gates / "b0").mkdir(parents=True)
    (gates / "b0" / "b0_result.json").write_text(json.dumps({"pass": True}))
    assert oracle.run_block(rows, "block1", p, out, gates, "sys") == oracle.EXIT_GATE  # bridge missing
    (gates / "bridge").mkdir()
    (gates / "bridge" / "bridge_result.json").write_text(json.dumps({"pass": False}))
    assert oracle.run_block(rows, "block1", p, out, gates, "sys") == oracle.EXIT_GATE  # bridge failed
    (gates / "bridge" / "bridge_result.json").write_text(json.dumps({"pass": True}))
    assert oracle.run_block(rows, "block1", p, out, gates, "sys") == 0
    gen = [json.loads(x) for x in (out / "generations.jsonl").read_text().splitlines()]
    assert len(gen) == 9 and all("determinism_warning_count" in g and "determinism_warning_ops" in g for g in gen)
    assert all("user" not in g and g["cond"] in ARMS for g in gen)
    assert oracle.gate_ok(gates, "block1")
    rows2 = _rows(["j0"], "block2")
    assert oracle.run_block(rows2, "block2", p, gates / "block2", gates, "sys") == 0  # block1 finished -> allowed
    (out / "run_status.json").write_text(json.dumps({"code": 5}))
    assert oracle.run_block(rows2, "block2", p, gates / "block2b", gates, "sys") == oracle.EXIT_GATE


def test_stage3_stop_conditions_still_apply(tmp_path: Path) -> None:
    gates = tmp_path
    for g, f, v in (("b0", "b0_result.json", {"pass": True}), ("bridge", "bridge_result.json", {"pass": True})):
        (gates / g).mkdir()
        (gates / g / f).write_text(json.dumps(v))
    rows = _rows(["i0"], "block1")
    assert oracle.run_block(rows, "block1", _provider([5, 128009]), tmp_path / "m", gates, "s", mem_fraction=lambda: 0.95) == 3
    assert oracle.run_block(rows, "block1", _provider([5, 6]), tmp_path / "e", gates, "s") == 5  # 3 consecutive eos_missing
    assert not oracle.gate_ok(gates, "block1")  # a stopped block never opens block2


def _ctx(pid: str) -> dict[str, object]:
    return {"prompt_id": pid, "kind": "pow2", "difficulty": "easy", "deficient": {"removed": "n"}, "expected": "4",
            "conditions": {f"{v}:{c}": {"user": f"{pid}-{v}-{c}", "tokens": 1, "sha256": stage3.sha(f"{pid}-{v}-{c}")}
                           for v in ("sufficient", "deficient") for c in stage3.CONDS}}


def test_bridge_match_and_mismatch(tmp_path: Path) -> None:
    ids = [f"p{n}" for n in range(20)]
    ctx = {p: _ctx(p) for p in ids}
    des = oracle.bridge_design(ctx, {"main": ids})
    assert des and all(d["cond"] in ("B", "Cprime") and d["idx"] % 50 == 0 for d in des)
    full = stage3.build_design(ids, ctx, conds=stage3.CONDS, samples=stage3.SAMPLES)
    assert len(des) == sum(1 for d in full if d["idx"] % 50 == 0 and d["cond"] in ("B", "Cprime"))
    gens = tmp_path / "g.jsonl"
    gens.write_text("".join(json.dumps({"idx": d["idx"], "prompt_id": d["prompt_id"], "cond": d["cond"],
                                        "version": d["version"], "sample": d["sample"], "token_ids": [5]}) + "\n" for d in full))
    ok = oracle.run_bridge(des, gens, _provider([5, 128009]), tmp_path / "ok", "s")
    assert ok["pass"] and ok["n"] == len(des)
    bad = oracle.run_bridge(des, gens, _provider([6, 128009]), tmp_path / "bad", "s")
    assert not bad["pass"] and bad["n_mismatch"] == len(des)
    (tmp_path / "g2.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="does not reproduce"):
        oracle.run_bridge(des, tmp_path / "g2.jsonl", _provider([5, 128009]), tmp_path / "x", "s")


def test_stage3_run_generations_default_unchanged(tmp_path: Path) -> None:
    d = stage3.build_design(["a"], {"a": {"prompt_id": "a", "kind": "pow2", "difficulty": "e", "deficient": {"removed": "n"},
                                          "expected": "4", "conditions": {f"{v}:{c}": {"user": "u", "tokens": 1, "sha256": stage3.sha("u")}
                                                                          for v in ("sufficient", "deficient") for c in stage3.CONDS}}},
                            conds=("A",), samples=(0,))
    assert stage3.run_generations(d, _provider([5, 128009]), tmp_path, "s") == 0
    row = json.loads((tmp_path / "generations.jsonl").read_text().splitlines()[0])
    assert "determinism_warning_count" not in row
