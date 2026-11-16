"""An append-only JSON-lines log with a hash chain.

Each line: ``{"seq": n, "prev": <hash of the previous line>, "record": {...}, "hash": <sha256 of
prev + canonical(record)>}``. Lines are only ever appended; ``verify`` recomputes the chain.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
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


# ---------------------------------------------------------------------------------------------- Formal Store only (U6T-2b-a)
class ChainChangedExternally(ChainLogError):
    """The file is not what this handle last wrote (size or last line differ from memory). Nothing was written."""


class TornTailError(ChainLogError):
    """The torn last line could not be set aside safely; the file was not changed."""


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class GuardedChainLog(ChainLog):
    """The chain of a Formal Store. Two additions to ``ChainLog``; the plain ``ChainLog`` is not touched.

    1. Before every append the file is compared with the memory (its size, and the ``hash`` of its last line against ``head``).
       A difference means another writer got in: nothing is written, the handle is dead for writing (``dead``), reads still work.
       This is detection, not a lock: a writer that arrives between the check and the write is not seen (U6T-8).
    2. A last line that does not end in a newline is a line that was being written when the process stopped. On open it is read as
       "not written": ``open_lines`` holds the complete lines only, ``torn`` says what was left. ``set_aside_torn_tail`` then moves
       the bytes to ``<dir of the chain>/quarantine/`` and only after the evidence is on disk cuts the file back. A line that ends
       in a newline but cannot be read, a chain that does not add up and a fork are not touched: they fail as before.
    """

    QUARANTINE_DIR = "quarantine"
    RECORD_FILE = "torn_tails.jsonl"

    def __init__(self, path: Path) -> None:                       # deliberately not ChainLog.__init__ (that one reads strictly)
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._head = GENESIS
        self._size = 0
        self.dead: str | None = None
        self.torn: dict[str, Any] | None = None
        self.open_lines: list[dict[str, Any]] = []
        data = path.read_bytes() if path.exists() else b""
        cut = data.rfind(b"\n") + 1
        complete, tail = data[:cut], data[cut:]
        self.open_lines = list(self._verify_bytes(complete))
        if self.open_lines:
            self._seq = self.open_lines[-1]["seq"] + 1
            self._head = self.open_lines[-1]["hash"]
        self._size = cut
        if tail:
            self.torn = {"offset": cut, "bytes": tail, "original_length": len(data)}

    # ---- reading
    @staticmethod
    def _verify_bytes(data: bytes) -> Iterator[dict[str, Any]]:
        prev, seq = GENESIS, 0
        for raw in data.split(b"\n")[:-1]:
            line = json.loads(raw.decode("utf-8"))
            if line["seq"] != seq or line["prev"] != prev or line["hash"] != _hash(prev, line["record"]):
                raise ChainLogError(f"chain broken at seq {seq}")
            prev, seq = str(line["hash"]), seq + 1
            yield line

    def verify(self) -> Iterator[dict[str, Any]]:
        """Re-reads the file. A torn last line that was not set aside yet is an error here (it is not part of the chain)."""
        data = self.path.read_bytes() if self.path.exists() else b""
        if data and not data.endswith(b"\n"):
            raise ChainLogError(f"{self.path}: the last line does not end in a newline")
        try:
            yield from self._verify_bytes(data)
        except ChainLogError as ex:
            raise ChainLogError(f"{self.path}: {ex}") from ex

    # ---- the check before an append
    def _last_line_hash(self) -> str:
        if not self._size:
            return GENESIS
        with self.path.open("rb") as handle:
            handle.seek(max(0, self._size - 1 - 65536))
            block = handle.read(min(self._size, 1 + 65536))
        body = block[:-1] if block.endswith(b"\n") else block
        last = body[body.rfind(b"\n") + 1:]
        try:
            return str(json.loads(last)["hash"])
        except (ValueError, KeyError, TypeError):
            return ""

    def check_unchanged(self) -> None:
        """Raises ``ChainChangedExternally`` (and kills the handle for writing) unless the file is as this handle left it."""
        if self.dead:
            raise ChainChangedExternally(self.dead)
        try:
            size = self.path.stat().st_size if self.path.exists() else 0
            same = size == self._size and (self._size == 0 or self._last_line_hash() == self._head)
        except OSError as ex:
            self.dead = f"{self.path}: cannot be read before an append: {type(ex).__name__}"
            raise ChainChangedExternally(self.dead) from ex
        if not same:
            self.dead = f"{self.path}: the file is not what this handle wrote (size {size}, expected {self._size})"
            raise ChainChangedExternally(self.dead)

    def append(self, record: dict[str, Any], *, sync: bool = False) -> dict[str, Any]:
        if self.torn is not None:
            raise TornTailError(f"{self.path}: a torn last line is not set aside yet")
        self.check_unchanged()
        digest = _hash(self._head, record)
        line = {"seq": self._seq, "prev": self._head, "record": record, "hash": digest}
        text = json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            if sync:
                handle.flush()
                os.fsync(handle.fileno())
        self._size += len(text.encode("utf-8"))
        self._seq += 1
        self._head = digest
        return line

    # ---- the torn last line: evidence first, cut last
    @property
    def quarantine_dir(self) -> Path:
        return self.path.parent / self.QUARANTINE_DIR

    @property
    def record_path(self) -> Path:
        return self.quarantine_dir / self.RECORD_FILE

    def torn_records(self) -> list[dict[str, Any]]:
        """Every tail that was ever set aside for this store (the quarantine record, verified)."""
        return [line["record"] for line in ChainLog(self.record_path).verify()] if self.record_path.exists() else []

    def _step_write_evidence(self, target: Path, tail: bytes) -> None:           # step 1
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != tail:
                raise TornTailError(f"{target} exists with other bytes")
            return
        part = target.with_name(target.name + ".part")
        with part.open("wb") as handle:
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(part, target)
        _fsync_dir(target.parent)

    def _step_write_record(self, record: dict[str, Any]) -> None:                # step 2
        log = ChainLog(self.record_path)
        if any(r.get("torn_sha256") == record["torn_sha256"] and r.get("original_length") == record["original_length"]
               and r.get("file") == record["file"] for r in (line["record"] for line in log.verify())):
            return                                                               # an earlier try got this far
        log.append(record, sync=True)
        _fsync_dir(self.record_path.parent)

    def _step_cut(self, offset: int) -> None:                                    # step 3
        with self.path.open("r+b") as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())

    def set_aside_torn_tail(self) -> dict[str, Any] | None:
        """1) the bytes to ``quarantine/<file>.<original length>.torn``, fsync; 2) their SHA-256, length, offset and time to the
        quarantine record, fsync; 3) only then cut the file back to the start of the torn line, fsync. A failure in 1) or 2) leaves
        the file exactly as it was and raises ``TornTailError``. Returns the record, or ``None`` when there is no torn tail."""
        if self.torn is None:
            return None
        tail, offset, original = self.torn["bytes"], self.torn["offset"], self.torn["original_length"]
        name = f"{self.path.name}.{original}.torn"
        record = {
            "kind": "torn_tail",
            "file": self.path.name,
            "original_length": original,
            "offset": offset,
            "torn_length": len(tail),
            "torn_sha256": hashlib.sha256(tail).hexdigest(),
            "torn_file": f"{self.QUARANTINE_DIR}/{name}",
            "quarantined_at": _utc_now(),
        }
        try:
            self._step_write_evidence(self.quarantine_dir / name, tail)
            self._step_write_record(record)
        except Exception as ex:
            raise TornTailError(f"{self.path}: the torn last line was not set aside ({type(ex).__name__}: {ex}); the file is unchanged") from ex
        try:
            self._step_cut(offset)
        except Exception as ex:
            raise TornTailError(f"{self.path}: the torn last line is kept but the file could not be cut ({type(ex).__name__}: {ex})") from ex
        self.torn = None
        self._size = offset
        return record
