"""U6T-2b-a: the checks before an append, the torn last line, and the pending chain of a Formal Store
(design U6T-2b v0.1, 3.3, 3.4, 4; T-T2b-04 .. 06). The legacy store must behave exactly as before."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from gv_support import BUNDLE, decl, make_env, make_verdict

from graph.chainlog import ChainLog, ChainLogError, GuardedChainLog
from graph.ic04 import IC04Error, IC04Store

CHAIN = Path("graph") / "ic00_envelopes.jsonl"
PENDING = Path("graph") / "ic00_pending.jsonl"
QDIR = Path("graph") / "quarantine"


def new_store(tmp_path: Path) -> IC04Store:
    return IC04Store.create_formal(tmp_path / "s", decl())


def reopen(root: Path) -> IC04Store:
    return IC04Store(root, require_formal=True)


def put(store: IC04Store, env: dict[str, Any]) -> dict[str, Any]:
    return store.put(env, None, gate_verdict=make_verdict(env), checker_bundle=BUNDLE)


def tree(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def torn_file(tmp_path: Path, keep: int = 17) -> tuple[Path, bytes, bytes]:
    """A store with 2 stored envelopes whose last line is cut after ``keep`` bytes. Returns (root, good part, torn bytes)."""
    store = new_store(tmp_path)
    put(store, make_env(seq=1))
    put(store, make_env(payload={"n": 2}, seq=2))
    path = store.data_dir / CHAIN
    data = path.read_bytes()
    cut = data[:-1].rfind(b"\n") + 1
    path.write_bytes(data[: cut + keep])
    return store.data_dir, data[:cut], data[cut : cut + keep]


# ------------------------------------------------------------------------------------------------ T-T2b-04
def test_t_t2b_04_another_handle_wrote_a_line(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env(seq=1))
    other = reopen(store.data_dir)
    put(other, make_env(payload={"n": 2}, seq=2))  # a rule-breaker writes while `store` is open
    before = tree(store.data_dir)
    with pytest.raises(IC04Error) as raised:
        put(store, make_env(payload={"n": 3}, seq=3))
    assert raised.value.code == "store_changed_externally"
    assert tree(store.data_dir) == before, "nothing is written, not even a blob"
    # the handle never writes again, whatever the envelope; reads still work
    with pytest.raises(IC04Error) as again:
        put(store, make_env(payload={"n": 4}, seq=4))
    assert again.value.code == "store_changed_externally"
    assert tree(store.data_dir) == before
    assert store.get(make_env(seq=1)["artifact_id"]) is not None
    # opening again shows what the file really holds: a good chain of 3 records (declaration + 2)
    assert len(reopen(store.data_dir)) == 2
    assert not reopen(store.data_dir).quarantined


def test_t_t2b_04_the_check_sees_a_changed_size_and_a_changed_last_line(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env(seq=1))
    path = store.data_dir / CHAIN
    original = path.read_bytes()
    path.write_bytes(original + b"x")  # size differs
    with pytest.raises(IC04Error, match="store_changed_externally|not what this handle wrote"):
        put(store, make_env(payload={"n": 2}, seq=2))
    path.write_bytes(original)
    other = new_store(tmp_path / "other")  # a handle whose own writes are checked and found unchanged
    put(other, make_env(seq=1))
    put(other, make_env(payload={"n": 2}, seq=2))  # still fine: the file is what it left


def test_a_plain_second_put_after_a_good_put_is_not_blocked(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    for n in range(1, 4):
        put(store, make_env(payload={"n": n}, seq=n))
    assert len(store) == 3 and store.log.dead is None


# ------------------------------------------------------------------------------------------------ T-T2b-05
def test_t_t2b_05_a_forked_chain_does_not_open(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env(seq=1))
    a = reopen(store.data_dir)
    b = reopen(store.data_dir)
    put(a, make_env(payload={"n": 2}, seq=2))
    path = store.data_dir / CHAIN
    after_a = path.read_bytes()
    # b's memory is the old head: write its line by hand, as a handle without the check would
    plain = ChainLog.__new__(ChainLog)
    plain.path, plain._seq, plain._head = path, b.log._seq, b.log._head
    plain.append({"artifact_id": "forked"})
    assert path.read_bytes() != after_a
    with pytest.raises(ChainLogError, match="chain broken"):
        reopen(store.data_dir)
    with pytest.raises(ChainLogError):
        IC04Store(store.data_dir)  # nor does the older entry


# ------------------------------------------------------------------------------------------------ T-T2b-06
def test_t_t2b_06_a_torn_last_line_is_set_aside_and_the_store_opens(tmp_path: Path) -> None:
    root, good, torn = torn_file(tmp_path)
    reopened = reopen(root)
    assert (root / CHAIN).read_bytes() == good
    assert not reopened.quarantined and len(reopened) == 1
    [rec] = reopened.torn_tails()
    saved = root / "graph" / rec["torn_file"]
    assert saved.read_bytes() == torn
    assert rec["torn_sha256"] == hashlib.sha256(torn).hexdigest()
    assert rec["torn_length"] == len(torn) and rec["offset"] == len(good)
    assert rec["original_length"] == len(good) + len(torn) and rec["file"] == "ic00_envelopes.jsonl"
    assert saved.name == f"ic00_envelopes.jsonl.{len(good) + len(torn)}.torn"
    assert saved.parent == root / QDIR and rec["quarantined_at"].endswith("Z")
    list(ChainLog(root / QDIR / "torn_tails.jsonl").verify())  # the record is itself a chain
    put(reopened, make_env(payload={"n": 9}, seq=9))  # and the store takes the next envelope
    assert len(reopen(root)) == 2


def test_a_torn_line_that_is_a_complete_line_without_its_newline_is_also_set_aside(tmp_path: Path) -> None:
    root, good, _torn = torn_file(tmp_path, keep=10**6)  # the whole last line; only its newline is taken off below
    path = root / CHAIN
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    reopen(root)
    assert (root / CHAIN).read_bytes() == good


def test_a_line_that_ends_in_a_newline_but_is_broken_does_not_open(tmp_path: Path) -> None:
    root, good, torn = torn_file(tmp_path)
    (root / CHAIN).write_bytes(good + torn + b"\n")
    before = tree(root)
    with pytest.raises(ValueError):  # json.JSONDecodeError, as it always was
        reopen(root)
    assert tree(root) == before and not (root / QDIR).exists()


def test_a_broken_chain_before_a_torn_tail_does_not_open_and_changes_nothing(tmp_path: Path) -> None:
    root, good, torn = torn_file(tmp_path)
    path = root / CHAIN
    path.write_bytes((good + torn).replace(b'"n":', b'"m":', 1) if b'"n":' in good else good.replace(b'"a":1', b'"a":2') + torn)
    before = tree(root)
    with pytest.raises(ChainLogError, match="chain broken"):
        reopen(root)
    assert tree(root) == before


def test_a_torn_tail_on_a_file_that_is_not_a_formal_store_is_left_alone(tmp_path: Path) -> None:
    legacy = IC04Store(tmp_path / "old", accepted_versions=("1.1.0",))
    legacy.put(make_env(seq=1))
    path = tmp_path / "old" / CHAIN
    path.write_bytes(path.read_bytes() + b'{"seq":1,"pr')
    before = tree(tmp_path / "old")
    with pytest.raises(IC04Error) as raised:
        IC04Store(tmp_path / "old", require_formal=True)
    assert raised.value.code == "formal_declaration_missing"
    assert tree(tmp_path / "old") == before


# the three steps; a failure in 1 or 2 leaves the file untouched and the store closed
@pytest.mark.parametrize("step", ["_step_write_evidence", "_step_write_record"])
def test_a_failure_in_step_1_or_2_changes_nothing_and_does_not_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str) -> None:
    root, good, torn = torn_file(tmp_path)
    before = tree(root)

    def boom(self: GuardedChainLog, *a: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(GuardedChainLog, step, boom)
    with pytest.raises(IC04Error) as raised:
        reopen(root)
    assert raised.value.code == "torn_tail_not_set_aside"
    assert (root / CHAIN).read_bytes() == good + torn, "not one byte of the original changed"
    assert {k: v for k, v in tree(root).items() if k == str(CHAIN)} == {k: v for k, v in before.items() if k == str(CHAIN)}
    monkeypatch.undo()
    reopened = reopen(root)  # the next open can finish it
    assert (root / CHAIN).read_bytes() == good and len(reopened.torn_tails()) == 1


def test_a_failure_after_step_1_keeps_the_evidence_and_the_retry_writes_one_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _good, torn = torn_file(tmp_path)

    def boom(self: GuardedChainLog, *a: Any) -> None:
        raise OSError("no")

    with monkeypatch.context() as m:
        m.setattr(GuardedChainLog, "_step_write_record", boom)
        with pytest.raises(IC04Error):
            reopen(root)
    [saved] = list((root / QDIR).glob("*.torn"))
    assert saved.read_bytes() == torn
    reopen(root)
    assert len(reopen(root).torn_tails()) == 1


def test_a_failure_in_step_3_keeps_the_evidence_and_the_record_and_does_not_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, good, torn = torn_file(tmp_path)

    def boom(self: GuardedChainLog, *a: Any) -> None:
        raise OSError("read-only")

    with monkeypatch.context() as m:
        m.setattr(GuardedChainLog, "_step_cut", boom)
        with pytest.raises(IC04Error) as raised:
            reopen(root)
    assert raised.value.code == "torn_tail_not_set_aside"
    assert (root / CHAIN).read_bytes() == good + torn
    assert len(list((root / QDIR).glob("*.torn"))) == 1
    reopened = reopen(root)  # the retry finds the evidence and the record already there: one record only
    assert (root / CHAIN).read_bytes() == good and len(reopened.torn_tails()) == 1


def test_the_order_is_evidence_then_record_then_cut(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, good, torn = torn_file(tmp_path)
    seen: list[str] = []
    for name in ("_step_write_evidence", "_step_write_record", "_step_cut"):
        original = getattr(GuardedChainLog, name)

        def wrap(self: GuardedChainLog, *a: Any, _o: Any = original, _n: str = name) -> None:
            if _n == "_step_cut":
                assert (root / CHAIN).read_bytes() == good + torn, "the file is still whole when the cut starts"
                assert list((root / QDIR).glob("*.torn")) and (root / QDIR / "torn_tails.jsonl").exists()
            seen.append(_n)
            return _o(self, *a)  # type: ignore[no-any-return]

        monkeypatch.setattr(GuardedChainLog, name, wrap)
    reopen(root)
    assert seen == ["_step_write_evidence", "_step_write_record", "_step_cut"]


def test_a_handle_that_is_not_set_aside_refuses_to_append(tmp_path: Path) -> None:
    root, _good, _torn = torn_file(tmp_path)
    log = GuardedChainLog(root / CHAIN)
    assert log.torn is not None and len(log.open_lines) == 2
    with pytest.raises(ChainLogError):
        log.append({"x": 1})
    with pytest.raises(ChainLogError):
        list(log.verify())  # the file as it is does not end in a newline


# ------------------------------------------------------------------------------------------------ the pending chain
def test_a_formal_store_with_a_pending_line_is_quarantined(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env(seq=1))
    ChainLog(store.data_dir / PENDING).append({"artifact_id": "held", "envelope": {}, "missing_parents": ["x"]}, sync=True)
    again = reopen(store.data_dir)
    assert again.quarantined
    assert [v["code"] for v in again.violations] == ["formal_pending_not_empty"]
    assert again.get(make_env(seq=1)["artifact_id"]) is not None  # reads work
    with pytest.raises(IC04Error) as raised:
        put(again, make_env(payload={"n": 2}, seq=2))
    assert raised.value.code == "formal_store_quarantined"
    assert reopen(store.data_dir).quarantined  # nothing clears it


def test_an_empty_pending_chain_is_fine(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    (store.data_dir / PENDING).write_bytes(b"")
    assert not reopen(store.data_dir).quarantined


# ------------------------------------------------------------------------------------------------ the legacy store
def test_the_legacy_store_still_refuses_a_torn_tail_and_still_does_not_check_before_appending(tmp_path: Path) -> None:
    legacy = IC04Store(tmp_path / "old", accepted_versions=("1.1.0",))
    legacy.put(make_env(seq=1))
    path = tmp_path / "old" / CHAIN
    good = path.read_bytes()
    path.write_bytes(good + b'{"seq":1,"pr')
    with pytest.raises(json.JSONDecodeError):
        IC04Store(tmp_path / "old", accepted_versions=("1.1.0",))
    assert path.read_bytes() == good + b'{"seq":1,"pr' and not (tmp_path / "old" / QDIR).exists()
    path.write_bytes(good)
    a, b = (IC04Store(tmp_path / "old", accepted_versions=("1.1.0",)) for _ in range(2))
    a.put(make_env(payload={"n": 2}, seq=2))
    b.put(make_env(payload={"n": 3}, seq=3))  # the old way: no check, so the chain forks, as before
    with pytest.raises(ChainLogError):
        IC04Store(tmp_path / "old", accepted_versions=("1.1.0",))
    assert type(a.log) is ChainLog and type(ChainLog(tmp_path / "x.jsonl")) is ChainLog
