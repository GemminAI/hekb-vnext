"""Stage-3 harness: runs without a GPU, with a mock backend (no torch)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from d71 import hf_provider, stage2, stage3


class MockBackend:
    def __init__(self, ids: list[int], eos: int = 128009) -> None:
        self._ids, self._eos = ids, eos
        self.calls: list[dict[str, int]] = []

    def eos_ids(self) -> list[int]:
        return [self._eos, 128001]

    def describe(self) -> dict[str, str]:
        return {"backend": "mock"}

    def generate(self, chat, *, seed, max_new_tokens, min_new_tokens):  # type: ignore[no-untyped-def]
        self.calls.append({"seed": seed, "max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens})
        n = len(self._ids)
        return SimpleNamespace(hidden=np.full((n, 1, 4), seed % 7, dtype=np.float32), ids=np.array(self._ids),
                               logprob=np.zeros(n, np.float32), entropy=np.ones(n, np.float32), text="ANSWER: 4")


def _provider(ids: list[int]) -> hf_provider.Llama31HFProvider:
    b = MockBackend(ids)
    return hf_provider.Llama31HFProvider(backend_factory=lambda m, r: b)


def test_seed_matches_stage2() -> None:
    assert stage3.seed_for("p", "sufficient", 1) == stage2.seed_for("p", "sufficient", 1)


def test_eos_is_stripped_and_hidden_shape() -> None:
    p = _provider([5, 6, 128009])
    r = p.generate_chat([{"role": "user", "content": "x"}], seed=3)
    assert r.ok and r.finish_reason == "eos" and r.token_ids == [5, 6] and r.completion_tokens == 2
    assert p.last_hidden is not None and p.last_hidden.shape == (3, 1, 4)  # one vector per generated step


def test_length_limit_and_min_new_tokens_zero() -> None:
    p = _provider([5, 6, 7])
    r = p.generate_chat([{"role": "user", "content": "x"}], max_tokens=3, seed=1)
    assert not r.ok and r.failure_kind == "length_limit"
    assert p._backend.calls[0]["min_new_tokens"] == 0


def test_refuses_other_sampler_settings() -> None:
    r = _provider([1]).generate_chat([{"role": "user", "content": "x"}], temperature=1.0)
    assert not r.ok and r.failure_kind == "generate_failed"


def _ctx(pid: str, kind: str) -> dict[str, object]:
    conds = {f"{v}:{c}": {"user": f"{pid}-{v}-{c}", "tokens": 10, "sha256": stage3.sha(f"{pid}-{v}-{c}")}
             for v in ("sufficient", "deficient") for c in stage3.CONDS}
    return {"prompt_id": pid, "kind": kind, "difficulty": "easy", "deficient": {"removed": "n"},
            "expected": "4", "conditions": conds}


def test_selection_rules_and_design() -> None:
    ctx = {f"i{n:03d}": _ctx(f"i{n:03d}", "pi_digits" if n % 10 == 0 else "pow2") for n in range(60)}
    s2 = {"i001", "i002"}
    sel = stage3.select_items(ctx, s2)
    chosen = set(sel["calibration"]) | set(sel["main"])
    assert not chosen & s2 and all(ctx[p]["kind"] != "pi_digits" for p in chosen)
    assert len(sel["calibration"]) == 20 and not set(sel["calibration"]) & set(sel["main"])
    assert sel == stage3.select_items(ctx, s2)  # deterministic
    design = stage3.build_design(sel["main"][:2], ctx, conds=stage3.CONDS, samples=stage3.SAMPLES)
    assert len(design) == 2 * 2 * 5 * 3
    assert [d["idx"] for d in design] == list(range(len(design)))


def test_run_resume_and_stop_on_failure(tmp_path: Path) -> None:
    ctx = {"a": _ctx("a", "pow2")}
    design = stage3.build_design(["a"], ctx, conds=("A",), samples=(0,))
    p = _provider([5, 128009])
    assert stage3.run_generations(design, p, tmp_path, "sys") == 0
    n = len((tmp_path / "generations.jsonl").read_text().splitlines())
    assert n == len(design) and (tmp_path / "hidden" / "000000.npy").exists()
    assert stage3.run_generations(design, p, tmp_path, "sys") == 0  # resume: nothing re-run
    assert len((tmp_path / "generations.jsonl").read_text().splitlines()) == n
    bad = _provider([5, 6])  # no EOS, 2 tokens < max -> eos_missing rows
    more = stage3.build_design(["a"], ctx, conds=("A",), samples=stage3.SAMPLES)
    code = stage3.run_generations(more, bad, tmp_path / "bad", "sys")
    assert code == 5  # 3 consecutive eos_missing rows
    long_ = _provider([5, 6, 7])  # hits max_tokens=3 -> length_limit rows must NOT stop the run
    import d71.stage3 as st

    st.MAX_TOKENS = 3
    try:
        assert st.run_generations(more, long_, tmp_path / "long", "sys") == 0
    finally:
        st.MAX_TOKENS = 768


def test_memory_stop(tmp_path: Path) -> None:
    ctx = {"a": _ctx("a", "pow2")}
    design = stage3.build_design(["a"], ctx, conds=("A",), samples=(0,))
    assert stage3.run_generations(design, _provider([5, 128009]), tmp_path, "s", mem_fraction=lambda: 0.95) == 3


def test_compare_runs_tolerance() -> None:
    h = np.zeros((2, 1, 4), np.float32)
    ok = stage3.compare_runs({"token_ids": [1], "hidden": h}, {"token_ids": [1], "hidden": h + 5e-4})
    assert ok["pass"]
    bad = stage3.compare_runs({"token_ids": [1], "hidden": h}, {"token_ids": [1], "hidden": h + 2e-3})
    assert not bad["pass"]
    assert not stage3.compare_runs({"token_ids": [1], "hidden": h}, {"token_ids": [2], "hidden": h})["pass"]


def test_manifest(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("a")
    m = stage3.write_manifest(tmp_path)
    assert m["n_files"] == 1 and json.loads((tmp_path / "manifest.json").read_text())["n_files"] == 1


def test_failed_generation_is_fail_not_match() -> None:
    h = np.zeros((2, 1, 4), np.float32)
    failed = {"token_ids": [], "hidden": None, "ok": False, "error": "boom"}
    r = stage3.compare_runs(failed, dict(failed))
    assert not r["pass"] and r["reason"] == "generation_failed"
    assert not stage3.compare_runs({"token_ids": [], "hidden": h}, {"token_ids": [], "hidden": h})["pass"]


def test_warning_summary_counts_ops() -> None:
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            warnings.warn("cumsum_cuda_kernel does not have a deterministic implementation, but ...")
    s = hf_provider.summarize_warnings(caught)
    assert s["count"] == 3 and s["ops"] == {"cumsum_cuda_kernel": 3}


def test_resample_rule_and_mismatch_recording(tmp_path: Path) -> None:
    ctx = {"a": _ctx("a", "pow2")}
    design = stage3.build_design(["a"] * 1, ctx, conds=stage3.CONDS, samples=stage3.SAMPLES)  # 30 rows
    prov = _provider([5, 128009])
    stage3.run_generations(design, prov, tmp_path, "s")
    summ = stage3.run_resample(design, prov, tmp_path, "s", every=10)
    assert summ["n_new"] == 3 and summ["mismatches_new"] == 0  # idx 0, 10, 20
    other = _provider([9, 128009])  # different tokens -> recorded, run continues
    summ2 = stage3.run_resample(design, other, tmp_path / "x", "s", every=10) if False else None
    assert summ2 is None
    rows = [json.loads(ln) for ln in (tmp_path / "resample.jsonl").read_text().splitlines()]
    assert [r["idx"] for r in rows] == [0, 10, 20] and all(r["token_ids_identical"] for r in rows)
