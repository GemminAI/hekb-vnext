"""An append-only JSON-lines log with a hash chain.

Each line: ``{"seq": n, "prev": <hash of the previous line>, "record": {...}, "hash": <sha256 of
prev + canonical(record)>}``. Lines are only ever appended; ``verify`` recomputes the chain.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash(prev: str, record: dict[str, Any]) -> str:
    return hashlib.sha256((prev + canonical(record)).encode("utf-8")).hexdigest()


class ChainLogError(RuntimeError):
    pass


class ChainLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._head = GENESIS
        self._pos = 0  # bytes of the file the in-memory (seq, head) account for
        self.reset_by_last_resync = False  # the file was replaced/removed: re-read everything
        self._ino: int | None = None
        if path.exists():
            for line in self.verify():
                self._seq = line["seq"] + 1
                self._head = line["hash"]
            stat = path.stat()
            self._pos, self._ino = stat.st_size, stat.st_ino

    def __len__(self) -> int:
        return self._seq

    @property
    def head(self) -> str:
        return self._head

    def resync(self) -> list[dict[str, Any]]:
        """Take over lines another process appended since this object last looked.

        Returns the records of those lines (the chain is verified as it is read). If the file
        was replaced or shortened, the whole file is read again. A partial last line is not
        consumed. The in-memory ``(seq, head)`` then match the file again, so a following
        ``append`` continues the chain instead of forking it.
        """
        self.reset_by_last_resync = False
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self.reset_by_last_resync = self._ino is not None
            self._seq, self._head, self._pos, self._ino = 0, GENESIS, 0, None
            return []
        replaced = self._ino is not None and (stat.st_ino != self._ino or stat.st_size < self._pos)
        if replaced:
            self.reset_by_last_resync = True
            self._seq, self._head, self._pos = 0, GENESIS, 0
        if stat.st_size == self._pos and not replaced:
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("rb") as handle:
            handle.seek(self._pos)
            for raw in handle:
                if not raw.endswith(b"\n"):
                    break
                line = json.loads(raw)
                if line["seq"] != self._seq or line["prev"] != self._head or line["hash"] != _hash(self._head, line["record"]):
                    raise ChainLogError(f"{self.path}: chain broken at seq {self._seq}")
                self._head, self._seq = str(line["hash"]), self._seq + 1
                self._pos += len(raw)
                records.append(line["record"])
        self._ino = stat.st_ino
        return records

    def append(self, record: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        if self.path.exists() and self.path.stat().st_size != self._pos:
            self.resync()  # another process appended: continue its chain, do not fork it
        digest = _hash(self._head, record)
        line = {"seq": self._seq, "prev": self._head, "record": record, "hash": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
        self._seq += 1
        self._head = digest
        stat = self.path.stat()
        self._pos, self._ino = stat.st_size, stat.st_ino
        return line

    def verify(self) -> Iterator[dict[str, Any]]:
        prev, seq = GENESIS, 0
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = json.loads(raw)
                if line["seq"] != seq or line["prev"] != prev or line["hash"] != _hash(prev, line["record"]):
                    raise ChainLogError(f"{self.path}: chain broken at seq {seq}")
                prev, seq = str(line["hash"]), seq + 1
                yield line

    def sync(self) -> None:
        if self.path.exists():
            with self.path.open("rb") as handle:
                os.fsync(handle.fileno())

    def records(self) -> Iterator[dict[str, Any]]:
        for line in self.verify():
            yield line["record"]
