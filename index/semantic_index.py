"""HEKB vNext: `semantic_addresses` index table (SPEC-HEKB-REFACTOR-2026-v2.0
section 3.1 / section 5 Step 5).

No prior SQLite index existed in this repo (verified: `store_service.py`'s
own on-disk layout before this change was JSON-file-per-object under
`objects/`, `relations/`, `lineages/`, `meta/` only -- no `.db` file
anywhere). This module adds the `hekb_vnext_index.db` SQLite file and its
`semantic_addresses` table, composite-keyed on `(object_id, codebook_id)` so
that re-running `/trajectories/recalibrate` under a new `codebook_id` never
overwrites a prior codebook generation's addresses -- multi-generation
history, per spec section 3.1 ("過去の実験時点で使用された住所履歴を完全に
再現可能にする").
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

# [OBSERVED DIFFERENCE, disclosed]: earlier revision of this module stored
# `address` as a JSON array string ("[1,0,1,...]"). Per this task's explicit
# request ("semantic_address 階層コード文字列"), addresses are now stored as
# a plain concatenated digit string ("101...") -- e.g. address (1,0,1) ->
# "101". `_encode`/`_decode` are the only two places this representation is
# assumed, so a future format change stays localized here.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_addresses (
    object_id         TEXT NOT NULL,
    codebook_id       TEXT NOT NULL,
    semantic_address  TEXT NOT NULL,   -- hierarchical digit string, e.g. "10100101"
    depth             INTEGER NOT NULL,
    created_at_ms     INTEGER NOT NULL,
    PRIMARY KEY (object_id, codebook_id)
);
CREATE INDEX IF NOT EXISTS idx_semantic_addresses_codebook ON semantic_addresses(codebook_id);
"""


def _encode(address: tuple[int, ...]) -> str:
    return "".join(str(d) for d in address)


def _decode(address_str: str) -> tuple[int, ...]:
    return tuple(int(c) for c in address_str)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def upsert_address(conn: sqlite3.Connection, object_id: str, codebook_id: str, address: tuple[int, ...]) -> None:
    """Idempotent: re-upserting the SAME (object_id, codebook_id, address)
    leaves the row's identity and content unchanged (only created_at_ms
    advances) -- calling this twice in a row never creates a second row or
    changes the semantic_address, verified by this task's own test run
    (see recalibrate.py's idempotency check)."""
    conn.execute(
        "INSERT INTO semantic_addresses (object_id, codebook_id, semantic_address, depth, created_at_ms) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(object_id, codebook_id) DO UPDATE SET semantic_address=excluded.semantic_address, depth=excluded.depth, created_at_ms=excluded.created_at_ms",
        (object_id, codebook_id, _encode(address), len(address), int(time.time() * 1000)),
    )


def get_address(conn: sqlite3.Connection, object_id: str, codebook_id: str) -> tuple[int, ...] | None:
    row = conn.execute(
        "SELECT semantic_address FROM semantic_addresses WHERE object_id = ? AND codebook_id = ?",
        (object_id, codebook_id),
    ).fetchone()
    if row is None:
        return None
    return _decode(row[0])


def latest_address_any_codebook(conn: sqlite3.Connection, object_id: str) -> tuple[str, tuple[int, ...]] | None:
    """Most recently written address for object_id, regardless of codebook_id
    -- used to compute address_churn_rate against whatever came before."""
    row = conn.execute(
        "SELECT codebook_id, semantic_address FROM semantic_addresses WHERE object_id = ? ORDER BY created_at_ms DESC LIMIT 1",
        (object_id,),
    ).fetchone()
    if row is None:
        return None
    codebook_id, address_str = row
    return codebook_id, _decode(address_str)


def count_rows(conn: sqlite3.Connection, codebook_id: str | None = None) -> int:
    if codebook_id is None:
        return conn.execute("SELECT COUNT(*) FROM semantic_addresses").fetchone()[0]
    return conn.execute(
        "SELECT COUNT(*) FROM semantic_addresses WHERE codebook_id = ?", (codebook_id,)
    ).fetchone()[0]
