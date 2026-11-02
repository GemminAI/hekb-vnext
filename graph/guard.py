"""One guarded write for every node and edge record (D13, D11).

Both entrances -- ``GraphStore`` and the HEXT projection layer -- write through
``put``. Under one lock on the graph directory (``<data_dir>/graph``, shared by every
process and by both entrances) it does, as a single operation:

  1. look the record's identity up (``node_id``; for an edge, ``[type, src, dst, attrs]``),
  2. compare the hash of the record with the stored one,
  3. same hash: report an idempotent success and write nothing,
  4. different hash: raise ``CollisionError`` and write nothing,
  5. nothing stored under that identity: append the record to the log.

The hash is ``sha256(canonical(record))`` with the chain log's own ``canonical`` over the
existing record shape, so no stored record, log line or chain hash changes.

The lock is ``flock`` on the directory itself, not on a lock file: every file under a data
dir counts in ``graph/rebuild_check.py``'s output digests (``rglob``), and a lock file would
change them.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

from graph.chainlog import ChainLog, canonical
from graph.model import GraphError


class CollisionError(GraphError):
    """The same identity was written with a different record. Nothing was written."""

    def __init__(self, kind: str, key: str, existing_hash: str, new_hash: str, where: str) -> None:
        self.kind = kind
        self.node_id = key  # the node_id; for an edge, the canonical [type, src, dst, attrs]
        self.existing_hash = existing_hash
        self.new_hash = new_hash
        self.where = where
        super().__init__(
            f"{kind} {key!r} already exists in {where} with a different content: "
            f"existing sha256 {existing_hash}, new sha256 {new_hash}"
        )


def record_hash(record: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(record).encode("utf-8")).hexdigest()


def node_key(record: dict[str, Any]) -> str:
    return str(record["node_id"])


def edge_key(record: dict[str, Any]) -> str:
    return canonical([record["type"], record["src"], record["dst"], record.get("attrs", {})])


class GraphLock:
    """Re-entrant, process-wide lock on one graph directory (``flock`` on the directory)."""

    _registry: dict[str, GraphLock] = {}
    _registry_guard = threading.Lock()

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._fd = -1

    @classmethod
    def of(cls, directory: Path) -> GraphLock:
        directory.mkdir(parents=True, exist_ok=True)
        key = str(directory.resolve())
        with cls._registry_guard:
            return cls._registry.setdefault(key, cls(directory))

    def __enter__(self) -> GraphLock:
        self._thread_lock.acquire()
        if self._depth == 0:
            self._fd = os.open(self.directory, os.O_RDONLY)
            fcntl.flock(self._fd, fcntl.LOCK_EX)
        self._depth += 1
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._depth -= 1
        if self._depth == 0:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = -1
        self._thread_lock.release()


class Ledger:
    """The identities stored in one chain log, kept current against other processes' appends."""

    def __init__(self, log: ChainLog, key_of: Callable[[dict[str, Any]], str], what: str) -> None:
        self.log = log
        self.key_of = key_of
        self.what = what
        self._hashes: dict[str, str] = {}
        for record in log.records():
            self._hashes.setdefault(key_of(record), record_hash(record))

    def refresh(self) -> None:
        """Index what other processes appended (call under the lock)."""
        records = self.log.resync()
        if self.log.reset_by_last_resync:  # the file was replaced: start over
            self._hashes = {}
        for record in records:
            self._hashes.setdefault(self.key_of(record), record_hash(record))

    def lookup(self, key: str) -> str | None:
        return self._hashes.get(key)

    def note(self, key: str, digest: str) -> None:
        self._hashes[key] = digest


class PeerNodes:
    """A read-only view of another log's node ids (the other entrance's log of one data dir).

    Re-read only when the file changed (inode, size or mtime), and never created here.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stamp: tuple[int, int, int] | None = None
        self._hashes: dict[str, str] = {}

    def lookup(self, key: str) -> str | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._stamp, self._hashes = None, {}
            return None
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if stamp != self._stamp:
            hashes: dict[str, str] = {}
            with self.path.open(encoding="utf-8") as handle:
                for raw in handle:
                    record = json.loads(raw)["record"]
                    hashes.setdefault(node_key(record), record_hash(record))
            self._stamp, self._hashes = stamp, hashes
        return self._hashes.get(key)


def put(
    lock: GraphLock,
    ledger: Ledger,
    record: dict[str, Any],
    *,
    peers: Sequence[PeerNodes] = (),
) -> bool:
    """Write ``record`` unless it is already stored. True if appended, False if idempotent.

    Raises ``CollisionError`` if the identity exists, here or (for a node) in a peer log,
    with a different content hash.
    """
    key = ledger.key_of(record)
    digest = record_hash(record)
    with lock:
        ledger.refresh()
        stored = ledger.lookup(key)
        if stored is not None:
            if stored == digest:
                return False
            raise CollisionError(ledger.what, key, stored, digest, str(ledger.log.path))
        for peer in peers:
            other = peer.lookup(key)
            if other is not None and other != digest:
                raise CollisionError(ledger.what, key, other, digest, str(peer.path))
        ledger.log.append(record)
        ledger.note(key, digest)
        return True
