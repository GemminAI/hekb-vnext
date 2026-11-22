"""IC-04 local entry of HEKB: put / get / recall / lineage for IC-00 envelopes (no signature, no key, customer side).

Responsibilities (D-121, D-124): HEKB stores a Gate-passed envelope AS IT IS, recalls it, and keeps the lineage walkable.
It does not interpret content and does not judge contract conformance (the Gate does). Its own acceptance is minimal:
  1. contract.version is one the store was opened to accept (default: IC-00 1.0.0 only; pass ``accepted_versions`` to add one),
  2. content_id / artifact_id recompute to the stored values (vendor/sensos_ic00, locked copy of the sensos reference ids),
  3. every parent is already stored (policy ``missing_parents``: "reject" default, or "hold" = keep in a pending log, never recallable),
  4. every blob the caller hands over matches its ``blob:sha256:<hex>`` uri.
A rejected envelope is never written. The primary key is the IC-00 ``artifact_id``; the chain ``seq`` / ``hash`` are HEKB's own index and
are never used as an IC-00 identifier. The existing signed ``POST /experience`` is untouched.

Storage: ``<data_dir>/graph/ic00_envelopes.jsonl`` (hash chain, same record shape as the earlier connection adapter) and ``blobs/``.
Chain record: {"artifact_id", "content_id", "envelope", "blob_shas", "ic04": {"gate_ref": ...|None}}.

Formal Store (U6T-2a, design U6T-2 v0.1): a store made by ``IC04Store.create_formal`` and opened with ``require_formal=True``. Its seq 0 is a
declaration (``{"ic04_store": {...}}``, no ``artifact_id``); ``put`` needs a GateVerdict that HEKB checks (C-0 .. C-8, graph/formal.py) and
keeps in the record (``ic04.gate_verdict``, ``ic04.checker_bundle_sha256``); a record that breaks the declaration, found on open, puts the
store in quarantine (reads work, ``put`` is refused). A store not opened with ``require_formal`` behaves as before.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graph import formal
from graph.blobs import read_blob, write_blob
from graph.chainlog import (
    ChainChangedExternally,
    ChainLog,
    GuardedChainLog,
    TornTailError,
)
from vendor.sensos_ic00 import ids as _ids
from vendor.sensos_ic00.errors import ContractError

IC04_VERSION = "0.1.0-provisional"
DEFAULT_ACCEPTED_VERSIONS = ("1.0.0",)
RECALL_EVENT_SCHEMA = ("hekb.recall-event", "0.1.0-provisional")
RECALL_PRODUCER = {"name": "hekb-vnext", "version": "0", "build": "ic04-local"}
_REQUIRED = ("contract", "content_id", "artifact_id", "kind", "payload_schema", "payload", "producer", "session_id", "observation_id",
             "event_span", "space_id", "parents", "policy_refs", "calibration_ref", "rule_refs", "seq", "produced_at", "extensions")


class IC04Error(Exception):
    """``code`` is one of: malformed, unsupported_contract_version, id_uncomputable, id_mismatch, parent_unknown, blob_mismatch,
    conflict, bad_query, admit_rejected; Formal Store: formal_declaration_missing / _invalid / _misplaced, formal_open_args_conflict,
    formal_store_needs_require_formal, formal_store_not_empty, formal_store_required, formal_store_quarantined, gate_ref_not_accepted,
    payload_schema_not_accepted, gate_required, gate_verdict_malformed, gate_catalog_mismatch, gate_subject_mismatch, gate_rule_mismatch,
    gate_lookup_mismatch, calibration_unknown, gate_rejected, gate_unverified_not_allowed, gate_waived_not_allowed,
    checker_bundle_missing, checker_bundle_mismatch, recall_not_supported_in_formal_store, store_changed_externally,
    torn_tail_not_set_aside; a violation (quarantine) code: formal_pending_not_empty."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


# ---- recall rankings: a registry, so a new ranking is an addition (a trajectory-nearness ranking would be computed customer-side, D-104) ----
RANKINGS: dict[str, Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] = {
    "seq_desc": lambda rows: sorted(rows, key=lambda r: (-r["envelope"]["seq"], r["artifact_id"])),
    "seq_asc": lambda rows: sorted(rows, key=lambda r: (r["envelope"]["seq"], r["artifact_id"])),
}
_QUERY_KEYS = {"artifact_ids", "space_id", "kind", "producer", "schema_id", "where", "exclude_where", "exclude_artifact_ids", "ranking"}


def _pointer(doc: Any, ptr: str) -> tuple[bool, Any]:
    cur = doc
    for part in [p.replace("~1", "/").replace("~0", "~") for p in ptr.split("/")[1:]]:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


def _same(a: Any, b: Any) -> bool:
    return type(a) is type(b) and a == b


def _cond_hit(env: Mapping[str, Any], cond: Mapping[str, Any]) -> bool:
    found, val = _pointer(env, cond["pointer"])
    return found and _same(val, cond["equals"])


class IC04Store:
    def __init__(self, data_dir: Path | str, *, accepted_versions: Sequence[str] | None = None, missing_parents: str | None = None,
                 event_contract_version: str | None = None, require_formal: bool = False) -> None:
        """Legacy (``require_formal`` false): ``accepted_versions`` defaults to ("1.0.0",), ``missing_parents`` to "reject", as always.
        Formal: both come from the declaration; an argument that differs from it is ``formal_open_args_conflict`` (nothing is silently overridden)."""
        if missing_parents not in (None, "reject", "hold"):
            raise ValueError("missing_parents must be 'reject' or 'hold'")
        self.data_dir = Path(data_dir)
        self.formal: dict[str, Any] | None = None      # the declaration of a Formal Store
        self.violations: list[dict[str, Any]] = []     # records that break the declaration (non-empty = quarantined)
        self.log: ChainLog = (GuardedChainLog if require_formal else ChainLog)(self.data_dir / "graph" / "ic00_envelopes.jsonl")
        self.pending = ChainLog(self.data_dir / "graph" / "ic00_pending.jsonl")
        self._index: dict[str, dict[str, Any]] = {}
        # a Formal Store reads its chain leniently (a torn last line is set aside after the declaration is checked); a legacy store strictly
        lines = list(self.log.open_lines) if isinstance(self.log, GuardedChainLog) else list(self.log.verify())
        declared = bool(lines) and isinstance(lines[0]["record"], dict) and "ic04_store" in lines[0]["record"]
        if require_formal:
            self._open_formal(lines, accepted_versions, missing_parents)
        else:
            if declared:
                raise IC04Error("formal_store_needs_require_formal", "seq 0 is a Formal Store declaration; open it with require_formal=True")
            self.accepted_versions = tuple(DEFAULT_ACCEPTED_VERSIONS if accepted_versions is None else accepted_versions)
            self.missing_parents = "reject" if missing_parents is None else missing_parents
            self.event_contract_version = event_contract_version or self.accepted_versions[0]   # IC-00 version of the recall-events HEKB builds
            if self.event_contract_version not in self.accepted_versions:
                raise ValueError("event_contract_version must be an accepted version")
            for line in lines:
                self._index[line["record"]["artifact_id"]] = line

    # ------------------------------------------------------------------ Formal Store: create / open / quarantine
    @classmethod
    def create_formal(cls, data_dir: Path | str, declaration: Mapping[str, Any]) -> IC04Store:
        """Make a Formal Store in an empty folder: seq 0 is ``{"ic04_store": declaration}``. The declaration is never changed afterwards
        (there is no API for it); to revise it, make a new store. Returns the store opened with ``require_formal=True``."""
        errs = formal.declaration_errors(dict(declaration))
        if errs:
            raise IC04Error("formal_declaration_invalid", "; ".join(errs[:3]))
        root = Path(data_dir)
        chains = [root / "graph" / "ic00_envelopes.jsonl", root / "graph" / "ic00_pending.jsonl"]
        blobs = root / "blobs"
        if any(c.exists() for c in chains) or (blobs.is_dir() and any(f.is_file() for f in blobs.rglob("*"))):
            raise IC04Error("formal_store_not_empty", "a Formal Store is made only in a folder with no chain file and no blob")
        ChainLog(chains[0]).append({"ic04_store": copy.deepcopy(dict(declaration))}, sync=True)
        return cls(root, require_formal=True)

    def _open_formal(self, lines: list[dict[str, Any]], accepted_versions: Sequence[str] | None, missing_parents: str | None) -> None:
        first = lines[0]["record"] if lines else None
        if not (isinstance(first, dict) and "ic04_store" in first):
            raise IC04Error("formal_declaration_missing", "seq 0 is not a declaration (an empty or an older store)")
        decl = first["ic04_store"]
        errs = formal.declaration_errors(decl) + ([] if set(first) == {"ic04_store"} else ["seq 0 has keys besides ic04_store"])
        if errs:
            raise IC04Error("formal_declaration_invalid", "; ".join(errs[:3]))
        for line in lines[1:]:
            if isinstance(line["record"], dict) and "ic04_store" in line["record"]:
                raise IC04Error("formal_declaration_misplaced", f"a declaration at seq {line['seq']}")
        if accepted_versions is not None and list(accepted_versions) != decl["accepted_versions"]:
            raise IC04Error("formal_open_args_conflict", f"accepted_versions {list(accepted_versions)} is not the declaration's {decl['accepted_versions']}")
        if missing_parents is not None and missing_parents != decl["missing_parents"]:
            raise IC04Error("formal_open_args_conflict", f"missing_parents {missing_parents!r} is not the declaration's {decl['missing_parents']!r}")
        assert isinstance(self.log, GuardedChainLog)
        if self.log.torn is not None:               # U6T-2b-a: evidence first, cut last; any failure = the store does not open, the file is unchanged
            try:
                self.log.set_aside_torn_tail()
            except TornTailError as ex:
                raise IC04Error("torn_tail_not_set_aside", str(ex)) from ex
        self.formal = copy.deepcopy(decl)
        self.accepted_versions = tuple(decl["accepted_versions"])
        self.missing_parents = decl["missing_parents"]
        self.event_contract_version = self.accepted_versions[0]
        if any(True for _ in self.pending.verify()):   # a Formal Store never holds (missing_parents is 'reject'): a pending line is a way around that
            self.violations.append({"hekb_seq": None, "artifact_id": None, "code": "formal_pending_not_empty",
                                    "detail": "ic00_pending.jsonl has a line; a Formal Store keeps nothing pending"})
        for line in lines[1:]:                      # 5: every record keeps the declaration's promise; the ones that do not are listed
            record = line["record"]
            aid = record.get("artifact_id") if isinstance(record, dict) else None
            try:
                self._revalidate(record)
            except (IC04Error, formal.Refusal) as ex:
                self.violations.append({"hekb_seq": line["seq"], "artifact_id": aid, "code": ex.code, "detail": ex.detail})
            except (KeyError, TypeError, AttributeError, ValueError) as ex:
                self.violations.append({"hekb_seq": line["seq"], "artifact_id": aid, "code": "record_malformed", "detail": f"{type(ex).__name__}: {ex}"})
            if isinstance(aid, str) and isinstance(record.get("envelope"), dict):
                self._index[aid] = line             # readable even when it is a violation

    def _revalidate(self, record: Mapping[str, Any]) -> None:
        """What ``put`` would have required of this record (the store's own index holds only the earlier ones). Raises on the first failure."""
        assert self.formal is not None
        env, ic04 = record["envelope"], record["ic04"]
        if record["artifact_id"] != env.get("artifact_id") or record["content_id"] != env.get("content_id"):
            raise IC04Error("id_mismatch", "the record's ids are not the envelope's")
        self._accept(env, {})
        if not formal.payload_accepted(self.formal, env):
            raise IC04Error("payload_schema_not_accepted", str(env.get("payload_schema")))
        if "gate_ref" in ic04:
            raise IC04Error("gate_ref_not_accepted", "a Formal Store record has no gate_ref")
        verdict = ic04.get("gate_verdict")
        if verdict is None:
            raise IC04Error("gate_required", "the record has no gate_verdict")
        formal.check_verdict(env, verdict, self.formal, self._known)
        bundle = verdict["gate"]["checker"]["bundle_sha256"]
        if ic04.get("checker_bundle_sha256") != bundle:
            raise IC04Error("checker_bundle_mismatch", "checker_bundle_sha256 is not the verdict's gate.checker.bundle_sha256")
        try:
            read_blob(self.data_dir, bundle)
        except (OSError, ValueError) as ex:
            raise IC04Error("checker_bundle_missing", f"the bundle blob {bundle} is not readable: {type(ex).__name__}") from ex

    def _known(self, artifact_id: str) -> dict[str, Any] | None:
        """``{content_id, seq, observation_id}`` of a stored envelope (the values a verdict's ``inputs.lookup`` must repeat), or ``None``."""
        line = self._index.get(artifact_id)
        if line is None:
            return None
        env = line["record"]["envelope"]
        return {"content_id": env["content_id"], "seq": env["seq"], "observation_id": env["observation_id"]}

    def torn_tails(self) -> list[dict[str, Any]]:
        """The last lines that were found cut off and set aside (graph/quarantine/torn_tails.jsonl). A Formal Store only."""
        return self.log.torn_records() if isinstance(self.log, GuardedChainLog) else []

    @property
    def is_formal(self) -> bool:
        return self.formal is not None

    @property
    def quarantined(self) -> bool:
        return bool(self.violations)

    @property
    def declaration(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.formal)

    def __len__(self) -> int:
        return len(self._index)

    def has(self, artifact_id: str) -> bool:
        return artifact_id in self._index

    # ------------------------------------------------------------------ acceptance
    def _lookup(self, env: Mapping[str, Any]) -> dict[str, dict[str, str]]:
        out = {}
        for p in env["parents"]:
            for a in p["artifact_ids"]:
                if a in self._index:
                    out[a] = {"content_id": self._index[a]["record"]["content_id"]}
        return out

    def _accept(self, env: Any, blobs: Mapping[str, bytes]) -> tuple[list[str], list[str]]:
        """Returns (blob_shas, missing_parent_ids) or raises IC04Error. Writes nothing."""
        if not isinstance(env, dict) or any(k not in env for k in _REQUIRED):
            raise IC04Error("malformed", "not an object, or a required envelope field is absent")
        c = env["contract"]
        ver = c.get("version") if isinstance(c, dict) else None
        if not (isinstance(c, dict) and c.get("id") == "IC-00" and isinstance(ver, str) and ver in self.accepted_versions):
            raise IC04Error("unsupported_contract_version", f"{ver!r} not in {list(self.accepted_versions)}")
        try:
            parents = env["parents"]
            wanted = [a for p in parents for a in p["artifact_ids"]]
            missing = [a for a in wanted if a not in self._index]
            if missing and self.missing_parents == "reject":
                raise IC04Error("parent_unknown", missing[0])
            if not missing:
                lk = self._lookup(env)
                if _ids.content_id(env, lk) != env["content_id"] or _ids.artifact_id(env, lk) != env["artifact_id"]:
                    raise IC04Error("id_mismatch", "content_id or artifact_id does not recompute")
        except IC04Error:
            raise
        except (ContractError, KeyError, TypeError, AttributeError) as ex:
            raise IC04Error("id_uncomputable", f"{type(ex).__name__}: {ex}") from ex
        shas = []
        for uri, data in blobs.items():
            sha = hashlib.sha256(data).hexdigest()
            if uri != "blob:sha256:" + sha:
                raise IC04Error("blob_mismatch", uri)
            shas.append(sha)
        return sorted(shas), missing

    # ------------------------------------------------------------------ put
    def put(self, envelope: dict[str, Any], blobs: Mapping[str, bytes] | None = None, *, gate_ref: Mapping[str, Any] | None = None,
            gate_verdict: Mapping[str, Any] | None = None, checker_bundle: bytes | None = None) -> dict[str, Any]:
        """Legacy store: as always (``gate_ref`` optional). Formal Store: ``gate_verdict`` is required, ``gate_ref`` is refused (see ``_put_formal``)."""
        if self.formal is not None:
            return self._put_formal(envelope, blobs, gate_ref, gate_verdict, checker_bundle)
        if gate_verdict is not None or checker_bundle is not None:
            raise IC04Error("formal_store_required", "gate_verdict / checker_bundle are for a Formal Store (create_formal, require_formal=True)")
        return self._put_legacy(envelope, blobs, gate_ref)

    def _put_legacy(self, envelope: dict[str, Any], blobs: Mapping[str, bytes] | None, gate_ref: Mapping[str, Any] | None) -> dict[str, Any]:
        """Store a Gate-passed envelope unchanged. Returns {artifact_id, content_id, hekb_seq, hekb_hash, already_stored, held}.
        The same artifact_id with the SAME envelope returns the same receipt (count unchanged); with a different envelope it is a
        ``conflict`` (seq / produced_at / extensions are outside the id preimage, so the id alone does not prove identity).
        ``gate_ref`` (checker, version, result ...) is kept beside the envelope in the chain record, not inside it; first one wins."""
        blobs = dict(blobs or {})
        aid = envelope.get("artifact_id") if isinstance(envelope, dict) else None
        if aid in self._index:
            if self._index[aid]["record"]["envelope"] != envelope:
                raise IC04Error("conflict", f"{aid} is stored with a different envelope body")
            return self._receipt(self._index[aid], already=True)
        shas, missing = self._accept(envelope, blobs)
        if missing:   # policy 'hold': recorded, never visible to get / recall / lineage
            self.pending.append({"artifact_id": aid, "envelope": copy.deepcopy(envelope), "missing_parents": missing}, sync=True)
            return {"artifact_id": aid, "content_id": envelope["content_id"], "hekb_seq": None, "hekb_hash": None, "already_stored": False, "held": missing}
        for data in blobs.values():
            write_blob(self.data_dir, data)
        rec = {"artifact_id": aid, "content_id": envelope["content_id"], "envelope": copy.deepcopy(envelope), "blob_shas": shas,
               "ic04": {"gate_ref": copy.deepcopy(dict(gate_ref)) if gate_ref else None}}
        line = self.log.append(rec, sync=True)
        self._index[aid] = line
        return self._receipt(line, already=False)

    def _put_formal(self, envelope: dict[str, Any], blobs: Mapping[str, bytes] | None, gate_ref: Mapping[str, Any] | None,
                    gate_verdict: Mapping[str, Any] | None, checker_bundle: bytes | None) -> dict[str, Any]:
        """The ``put`` of a Formal Store (design U6T-2 v0.1, 4). Nothing is written unless every check passes.
        C-0 not quarantined | C-1 the old acceptance + the payload is in ``accepted_payload_schemas`` | C-2 a verdict (and no ``gate_ref``) |
        C-3 form, hashes, catalogue | C-4 subject | C-5 rules and lookup | C-6 calibration known | C-7 the kind of pass |
        C-8a the checker bundle is a blob of the store (or is given) | C-8 write. The same artifact_id with the same envelope returns the
        stored receipt (the first verdict wins); with another envelope it is a ``conflict``."""
        assert self.formal is not None
        assert isinstance(self.log, GuardedChainLog)
        if self.log.dead:                                                                           # U6T-2b-a: a handle that saw another writer never writes again
            raise IC04Error("store_changed_externally", self.log.dead)
        if self.violations:                                                                         # C-0
            raise IC04Error("formal_store_quarantined", f"{len(self.violations)} record(s) break the declaration")
        if gate_ref is not None:
            raise IC04Error("gate_ref_not_accepted", "a Formal Store takes a gate_verdict, not a gate_ref")
        blobs = dict(blobs or {})
        aid = envelope.get("artifact_id") if isinstance(envelope, dict) else None
        if aid in self._index:
            if self._index[aid]["record"]["envelope"] != envelope:
                raise IC04Error("conflict", f"{aid} is stored with a different envelope body")
            return self._receipt(self._index[aid], already=True)
        shas, _missing = self._accept(envelope, blobs)                                              # C-1
        assert isinstance(aid, str)   # _accept refuses an envelope without a usable artifact_id
        if not formal.payload_accepted(self.formal, envelope):
            raise IC04Error("payload_schema_not_accepted", str(envelope.get("payload_schema")))
        if gate_verdict is None:                                                                    # C-2
            raise IC04Error("gate_required", "a Formal Store needs the GateVerdict of this envelope")
        verdict = copy.deepcopy(gate_verdict)
        try:
            formal.check_verdict(envelope, verdict, self.formal, self._known)                       # C-3 .. C-7
        except formal.Refusal as ex:
            raise IC04Error(ex.code, ex.detail) from ex
        bundle_sha = verdict["gate"]["checker"]["bundle_sha256"]                                   # C-8a
        stored_bundle = True
        try:
            read_blob(self.data_dir, bundle_sha)
        except (OSError, ValueError):
            stored_bundle = False
        if checker_bundle is not None and formal.sha256_hex(checker_bundle) != bundle_sha:
            raise IC04Error("checker_bundle_mismatch", "the given bundle is not the one the verdict names")
        if not stored_bundle and checker_bundle is None:
            raise IC04Error("checker_bundle_missing", f"the bundle {bundle_sha} is not in the store and was not given")
        try:
            self.log.check_unchanged()                                                              # before the first byte is written (U6T-2b-a)
        except ChainChangedExternally as ex:
            raise IC04Error("store_changed_externally", str(ex)) from ex
        for data in blobs.values():                                                                 # C-8
            write_blob(self.data_dir, data)
        if not stored_bundle:
            assert checker_bundle is not None
            write_blob(self.data_dir, checker_bundle)
        rec = {"artifact_id": aid, "content_id": envelope["content_id"], "envelope": copy.deepcopy(envelope), "blob_shas": shas,
               "ic04": {"gate_verdict": verdict, "checker_bundle_sha256": bundle_sha}}
        try:
            line = self.log.append(rec, sync=True)
        except ChainChangedExternally as ex:
            raise IC04Error("store_changed_externally", str(ex)) from ex
        self._index[aid] = line
        return self._receipt(line, already=False)

    @staticmethod
    def _receipt(line: Mapping[str, Any], *, already: bool) -> dict[str, Any]:
        r = line["record"]
        return {"artifact_id": r["artifact_id"], "content_id": r["content_id"], "hekb_seq": line["seq"], "hekb_hash": line["hash"],
                "already_stored": already, "held": []}

    # ------------------------------------------------------------------ get / lineage
    def get(self, artifact_id: str) -> dict[str, Any] | None:
        line = self._index.get(artifact_id)
        if line is None:
            return None
        r = line["record"]
        blobs = {"blob:sha256:" + s: read_blob(self.data_dir, s) for s in r["blob_shas"]}
        got = {"envelope": copy.deepcopy(r["envelope"]), "blobs": blobs, "hekb_seq": line["seq"], "hekb_hash": line["hash"]}
        if self.formal is not None:
            got["gate_verdict"] = copy.deepcopy(r["ic04"].get("gate_verdict"))
        else:
            got["gate_ref"] = copy.deepcopy(r["ic04"]["gate_ref"])
        return got

    def gate_status(self, artifact_id: str) -> dict[str, Any] | None:
        """``{"self": <state>, "verdict": <what the record holds>}`` or ``None`` if not stored. ``self`` is NONE / LEGACY / INCONSISTENT /
        FAIL_RECORDED / WAIVED / PASS_UNVERIFIED / PASS (design v0.3 5.2), from the record alone; the aggregation over a lineage is not here."""
        line = self._index.get(artifact_id)
        if line is None:
            return None
        r = line["record"]
        held = r["ic04"].get("gate_verdict", r["ic04"].get("gate_ref"))
        return {"self": formal.self_state(r["envelope"], held), "verdict": copy.deepcopy(held)}

    def checker_bundle(self, artifact_id: str) -> bytes | None:
        """The bytes of the checker bundle named by the stored verdict (a Blob of a Formal Store), or ``None``."""
        line = self._index.get(artifact_id)
        sha = line["record"]["ic04"].get("checker_bundle_sha256") if line else None
        return read_blob(self.data_dir, sha) if sha else None

    def lineage(self, artifact_id: str, *, max_depth: int | None = None) -> dict[str, Any]:
        """Envelopes reached by walking ``parents`` (breadth first, each once, root first). ``missing`` lists ids not stored."""
        entries, missing, seen, frontier = [], [], {artifact_id}, [artifact_id]
        depth = 0
        while frontier and (max_depth is None or depth <= max_depth):
            nxt = []
            for a in frontier:
                line = self._index.get(a)
                if line is None:
                    missing.append(a)
                    continue
                env = copy.deepcopy(line["record"]["envelope"])
                entries.append({"artifact_id": a, "depth": depth, "envelope": env})
                for p in env["parents"]:
                    for pa in p["artifact_ids"]:
                        if pa not in seen:
                            seen.add(pa)
                            nxt.append(pa)
            frontier, depth = nxt, depth + 1
        return {"root": artifact_id, "entries": entries, "missing": missing}

    # ------------------------------------------------------------------ recall
    def select(self, query: Mapping[str, Any], limit: int) -> tuple[list[str], bool]:
        """Filter + rank the stored envelopes (no content interpretation: filters are field equality / JSON-pointer equality).
        query keys: artifact_ids, space_id, kind, producer, schema_id, where, exclude_where [{pointer, equals}], exclude_artifact_ids, ranking."""
        bad = set(query) - _QUERY_KEYS
        if bad:
            raise IC04Error("bad_query", f"unknown query key(s): {sorted(bad)}")
        ranking = query.get("ranking", "seq_desc")
        if ranking not in RANKINGS:
            raise IC04Error("bad_query", f"unknown ranking {ranking!r} (known: {sorted(RANKINGS)})")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise IC04Error("bad_query", "limit must be an integer >= 1")
        for key in ("where", "exclude_where"):
            for cond in query.get(key, ()):
                if not (isinstance(cond, Mapping) and set(cond) == {"pointer", "equals"} and isinstance(cond["pointer"], str) and cond["pointer"].startswith("/")):
                    raise IC04Error("bad_query", f"{key} entries are {{pointer, equals}}")
        only = set(query["artifact_ids"]) if "artifact_ids" in query else None
        excl = set(query.get("exclude_artifact_ids", ()))
        rows = []
        for aid, line in self._index.items():
            env = line["record"]["envelope"]
            if only is not None and aid not in only: continue
            if aid in excl: continue
            if "space_id" in query and not _same(env["space_id"], query["space_id"]): continue
            if "kind" in query and env["kind"] != query["kind"]: continue
            if "producer" in query and env["producer"]["name"] != query["producer"]: continue
            if "schema_id" in query and env["payload_schema"]["id"] != query["schema_id"]: continue
            if not all(_cond_hit(env, c) for c in query.get("where", ())): continue
            if any(_cond_hit(env, c) for c in query.get("exclude_where", ())): continue
            rows.append({"artifact_id": aid, "envelope": env})
        rows = RANKINGS[ranking](rows)
        return [r["artifact_id"] for r in rows[:limit]], len(rows) > limit

    def recall(self, query: Mapping[str, Any], *, requester: str, context: Mapping[str, Any], limit: int = 10,
               admit: Callable[[dict[str, Any], dict[str, dict[str, str]]], None] | None = None) -> dict[str, Any]:
        """Select, build the recall-event (observation, producer hekb-vnext), STORE it, then return it with the recalled envelopes.
        ``context`` = {session_id, observation_id, seq, produced_at}: IC-00 fields only the caller's runtime owns (HEKB has no clock or
        seq counter of the IC-00 sequence). ``admit(envelope, lookup)`` is the caller's Gate: if given, it runs on the recall-event
        BEFORE it is stored and may raise to refuse (-> IC04Error admit_rejected, nothing stored). ``requester`` is a caller-declared
        label, not an authenticated identity (no keys here). Results are the stored envelopes as they are (the recall-event itself and
        the caller's own current input can be among them unless excluded by the query)."""
        if self.formal is not None:
            raise IC04Error("recall_not_supported_in_formal_store", "a Formal Store does not store recall-events in this version (U6T-4)")
        ids, truncated = self.select(query, limit)
        for k in ("session_id", "observation_id", "seq", "produced_at"):
            if k not in context:
                raise IC04Error("bad_query", f"context.{k} is required")
        payload = {"requester": requester, "query": copy.deepcopy(dict(query)), "limit": limit, "ranking": query.get("ranking", "seq_desc"),
                   "truncated": truncated, "results": [{"rank": i, "artifact_id": a} for i, a in enumerate(ids)]}
        env: dict[str, Any] = {
            "contract": {"id": "IC-00", "version": self.event_contract_version},
            "content_id": "cnt:" + "0" * 64, "artifact_id": "art:" + "0" * 64, "kind": "observation",
            "payload_schema": {"id": RECALL_EVENT_SCHEMA[0], "version": RECALL_EVENT_SCHEMA[1]}, "payload": payload,
            "producer": dict(RECALL_PRODUCER), "session_id": context["session_id"], "observation_id": context["observation_id"], "event_span": None,
            "space_id": None, "parents": [], "policy_refs": [], "calibration_ref": None, "rule_refs": [], "observed_at": None,
            "seq": context["seq"], "produced_at": context["produced_at"], "extensions": {}}
        env["content_id"] = _ids.content_id(env, {})
        env["artifact_id"] = _ids.artifact_id(env, {})
        if admit is not None:
            try:
                admit(env, {})
            except Exception as ex:
                raise IC04Error("admit_rejected", f"{type(ex).__name__}: {ex}") from ex
        receipt = self.put(env, None, gate_ref={"checker": "caller-admit", "applied": admit is not None})
        results = [{"artifact_id": a, **{k: v for k, v in self.get(a).items() if k != "gate_ref"}} for a in ids]
        return {"event": env, "event_receipt": receipt, "results": results}

    # ------------------------------------------------------------------ integrity
    def verify_all(self) -> dict[str, dict[str, Any]]:
        """Re-read both chains from disk (raises ChainLogError if broken); returns {artifact_id: chain line} of the stored envelopes."""
        list(self.pending.verify())
        return {l["record"]["artifact_id"]: l for l in self.log.verify() if "ic04_store" not in l["record"]}
