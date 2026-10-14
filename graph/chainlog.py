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
        if path.exists():
            for line in self.verify():
                self._seq = line["seq"] + 1
                self._head = line["hash"]

    def __len__(self) -> int:
        return self._seq

    @property
    def head(self) -> str:
        return self._head

    def append(self, record: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        digest = _hash(self._head, record)
        line = {"seq": self._seq, "prev": self._head, "record": record, "hash": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n")
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
        self._seq += 1
        self._head = digest
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
