"""Value-less cells in BoardRecord ingestion (board-42-v2: P15 carries no value; wire `null`).

Default: IngestError that names the cell and the option. `--drop-valueless-columns`: the column is
left out (37 columns) and recorded in the space fingerprint and in the trajectory metadata. The
38-column board-42-v1 ingest is unchanged. What a missing value means to HEKB in general is not
decided here. Every fixture uses a throwaway tmp_path data dir (never a running service)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from index.space_id import SpaceIdError, compute_space_id

PORTS = [f"P{n:02d}" for n in range(1, 39)]
ANCHOR_POLICY = "a" * 64
SPACE_BASE = {
    "model_ref": "toy-model", "layer": 7, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": ANCHOR_POLICY, "wiring_policy_sha256": "w" * 64,
    "time_policy_sha256": "t" * 64, "quant_step_log2": -30,
}  # fmt: skip


def _record(session_id: str, t: int, values: list[int | None], board_format: str) -> dict:
    return {
        "format": board_format, "session_id": session_id, "seq": t, "token_index": t,
        "prev_hash": "0" * 64, "anchor_policy_sha256": ANCHOR_POLICY,
        "anchor_counts": {"config_bootstrap": 0, "dynamic": 0, "manual": 0},
        "causal_log_ref": "0" * 64,
        "cells": [[p, "NOT_MEASURED" if v is None else "LIVE", v] for p, v in zip(PORTS, values, strict=True)],
        "calibration": "未較正", "summary": "", "crc": 0, "record_hash": "", "hmac": "",
    }  # fmt: skip


def _session_values(seed: int, n_steps: int, p15: str) -> list[list[int | None]]:
    """`p15`: "zero" (board-42-v1: a column of 0), "null" (board-42-v2), "number" (v2 with a value)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        row: list[int | None] = [int(v) for v in rng.integers(-1_000_000, 1_000_000, size=38)]
        row[PORTS.index("P15")] = {"zero": 0, "null": None, "number": int(rng.integers(1, 99))}[p15]
        out.append(row)
    return out


def _make(tmp_path: Path, name: str, board_format: str, p15: str, *, sessions: int = 2, n_steps: int = 12,
          mutate=None) -> Path:  # type: ignore[no-untyped-def]
    root = tmp_path / name
    root.mkdir()
    entries = []
    for k in range(sessions):
        sid = f"{name}-s{k}"
        rows = _session_values(100 + k, n_steps, p15)
        records = [_record(sid, t, row, board_format) for t, row in enumerate(rows)]
        if mutate is not None:
            mutate(k, records)
        with gzip.open(root / f"{sid}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")
        entries.append({"session_id": sid, "file": f"{sid}.jsonl.gz", "n_records": n_steps,
                        "category": "c", "group": f"g{k}", "split": "dev"})  # fmt: skip
    manifest = {**SPACE_BASE, "sessions": entries, "board_format": board_format, "format_version": 1}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _ingest(root: Path, data_dir: Path, **kwargs):  # type: ignore[no-untyped-def]
    import ingest_boardrecords as ib

    return ib.run_ingest(manifest_path=root / "manifest.json", shards_root=root, data_dir=data_dir, **kwargs)


def _trajectory_ext(data_dir: Path) -> list[dict]:
    out = []
    for path in sorted((data_dir / "objects").glob("*.json")):
        ext = json.loads(path.read_text())["payload"]["ext"].get("hekbv3")
        if ext and ext.get("kind") == "trajectory":
            out.append(ext)
    return out


def test_v1_38_columns_ingest_is_unchanged(tmp_path: Path) -> None:
    root = _make(tmp_path, "v1", "board-42-v1", "zero")
    result = _ingest(root, tmp_path / "d1")
    assert result["excluded_columns"] == []
    for ext in _trajectory_ext(tmp_path / "d1"):
        assert ext["ports"]["dim"] == 38 and "P15" in ext["ports"]["port_ids"]
        assert "excluded_columns" not in ext["ports"] and "excluded_columns" not in ext["space"]
    # the space_id is the 9-key hash it always was
    assert result["space_id"] == compute_space_id({**SPACE_BASE, "board_format": "board-42-v1"})


def test_v2_valueless_cell_is_refused_by_default_with_a_message_that_names_the_option(tmp_path: Path) -> None:
    import ingest_boardrecords as ib

    root = _make(tmp_path, "v2", "board-42-v2", "null")
    with pytest.raises(ib.IngestError) as raised:
        _ingest(root, tmp_path / "d2")
    message = str(raised.value)
    assert "P15" in message and "--drop-valueless-columns" in message and "no value" in message
    objects = tmp_path / "d2" / "objects"
    assert not objects.exists() or not list(objects.glob("*.json"))  # nothing was stored


def test_the_option_drops_the_column_and_records_it_in_the_fingerprint_and_the_metadata(tmp_path: Path) -> None:
    root = _make(tmp_path, "v2o", "board-42-v2", "null")
    result = _ingest(root, tmp_path / "d3", drop_valueless_columns=True)
    assert result["excluded_columns"] == ["P15"] and result["n_sessions"] == 2
    exts = _trajectory_ext(tmp_path / "d3")
    assert len(exts) == 2
    expected_space = {**SPACE_BASE, "board_format": "board-42-v2", "excluded_columns": "P15=valueless"}
    for ext in exts:
        assert ext["ports"]["dim"] == 37 and "P15" not in ext["ports"]["port_ids"]
        assert ext["ports"]["excluded_columns"] == [{"port_id": "P15", "reason": "valueless"}]
        assert ext["space"]["excluded_columns"] == "P15=valueless"
        assert ext["space"]["space_id"] == compute_space_id(expected_space) == result["space_id"]


def test_ingesting_the_same_records_twice_is_idempotent(tmp_path: Path) -> None:
    root = _make(tmp_path, "v2i", "board-42-v2", "null")
    first = _ingest(root, tmp_path / "d4", drop_valueless_columns=True)
    second = _ingest(root, tmp_path / "d4", drop_valueless_columns=True)
    assert first["space_id"] == second["space_id"]
    assert [s["object_id"] for s in first["sessions"]] == [s["object_id"] for s in second["sessions"]]


def test_v1_and_v2_never_share_a_space_id(tmp_path: Path) -> None:
    v1 = _ingest(_make(tmp_path, "a1", "board-42-v1", "zero"), tmp_path / "d5")
    v2 = _ingest(_make(tmp_path, "a2", "board-42-v2", "null"), tmp_path / "d6", drop_valueless_columns=True)
    assert v1["space_id"] != v2["space_id"]
    # the exclusion alone separates spaces, even with the same board_format
    base = {**SPACE_BASE, "board_format": "board-42-v2"}
    assert compute_space_id(base) != compute_space_id({**base, "excluded_columns": "P15=valueless"})
    assert compute_space_id({**base, "excluded_columns": "P15=valueless"}) != compute_space_id(
        {**base, "excluded_columns": "P15=valueless,P99=valueless"}
    )


def test_a_v2_session_with_no_valueless_cell_passes_without_the_option_and_records_no_exclusion(tmp_path: Path) -> None:
    root = _make(tmp_path, "v2n", "board-42-v2", "number")
    for flag in (False, True):  # the option is a no-op when nothing is value-less
        result = _ingest(root, tmp_path / f"d7{flag}", drop_valueless_columns=flag)
        assert result["excluded_columns"] == []
        for ext in _trajectory_ext(tmp_path / f"d7{flag}"):
            assert ext["ports"]["dim"] == 38 and "excluded_columns" not in ext["space"]


def test_malformed_valueless_cells_are_ingest_errors_not_encoder_errors(tmp_path: Path) -> None:
    import ingest_boardrecords as ib

    def partly_null(k: int, records: list[dict]) -> None:  # null in one record only
        records[3]["cells"][PORTS.index("P15")][2] = None

    root = _make(tmp_path, "pn", "board-42-v2", "number", mutate=partly_null)
    with pytest.raises(ib.IngestError, match="P15"):
        _ingest(root, tmp_path / "d8")
    with pytest.raises(ib.IngestError, match="P15"):  # dropped as value-less, but 11 records have a value
        _ingest(root, tmp_path / "d9", drop_valueless_columns=True)

    def float_value(k: int, records: list[dict]) -> None:
        records[0]["cells"][0][2] = 1.5

    with pytest.raises(ib.IngestError, match="non-integer"):
        _ingest(_make(tmp_path, "fl", "board-42-v1", "zero", mutate=float_value), tmp_path / "d10")

    def second_session_has_none(k: int, records: list[dict]) -> None:
        if k == 1:
            for rec in records:
                rec["cells"][PORTS.index("P15")][2] = None

    mixed = _make(tmp_path, "mx", "board-42-v2", "number", mutate=second_session_has_none)
    with pytest.raises(ib.IngestError, match="same value-less columns"):
        _ingest(mixed, tmp_path / "d11", drop_valueless_columns=True)


def test_p15_is_not_in_the_index_feature_so_dropping_it_changes_no_key(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """The experience index reads ten named ports (P01 P03 P09 P13 P18 P19 P25 P28 P31 P35)."""
    import store_service
    from index import experience_index as ei

    def keys_of(root: Path, name: str, **kwargs):  # type: ignore[no-untyped-def]
        data = tmp_path / name
        result = _ingest(root, data, **kwargs)
        monkeypatch.setenv("HEKB_INDEX_DIR", str(tmp_path / f"{name}-idx"))
        ei.build_index(store_service._objects_dir(), store_service._blobs_dir(), tmp_path / f"{name}-idx", result["space_id"])
        keys, ids, meta = ei.load_index(tmp_path / f"{name}-idx", result["space_id"])
        return {(e["group"], e["token_index"]): keys[i] for i, e in enumerate(ids)}, meta

    # same numbers in the 37 shared columns: v1's P15 is a column of 0, v2's is null (dropped)
    k1, m1 = keys_of(_make(tmp_path, "i1", "board-42-v1", "zero"), "k1")
    k2, m2 = keys_of(_make(tmp_path, "i2", "board-42-v2", "null"), "k2", drop_valueless_columns=True)
    assert k1.keys() == k2.keys() and len(k1) > 0
    assert all(np.array_equal(k1[c], k2[c]) for c in k1)
    assert m1["kept_dims"] == m2["kept_dims"] and m1["d_kept"] == m2["d_kept"]


def test_space_id_accepts_the_optional_key_only_as_a_non_empty_string() -> None:
    base = {**SPACE_BASE, "board_format": "board-42-v2"}
    with pytest.raises(SpaceIdError):
        compute_space_id({**base, "excluded_columns": ""})
    with pytest.raises(SpaceIdError):
        compute_space_id({**base, "excluded_columns": ["P15"]})
    with pytest.raises(SpaceIdError):
        compute_space_id({**base, "not_a_key": "x"})
