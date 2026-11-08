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
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from graph.blobs import read_blob, write_blob
from graph.chainlog import ChainLog
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
    conflict, bad_query, admit_rejected."""

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
    def __init__(self, data_dir: Path | str, *, accepted_versions: Sequence[str] = DEFAULT_ACCEPTED_VERSIONS, missing_parents: str = "reject",
                 event_contract_version: str | None = None) -> None:
        if missing_parents not in ("reject", "hold"):
            raise ValueError("missing_parents must be 'reject' or 'hold'")
        self.data_dir = Path(data_dir)
        self.accepted_versions = tuple(accepted_versions)
        self.missing_parents = missing_parents
        self.event_contract_version = event_contract_version or self.accepted_versions[0]   # IC-00 version of the recall-events HEKB builds
        if self.event_contract_version not in self.accepted_versions:
            raise ValueError("event_contract_version must be an accepted version")
        self.log = ChainLog(self.data_dir / "graph" / "ic00_envelopes.jsonl")
        self.pending = ChainLog(self.data_dir / "graph" / "ic00_pending.jsonl")
        self._index: dict[str, dict[str, Any]] = {}
        for line in self.log.verify():
            self._index[line["record"]["artifact_id"]] = line

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
    def put(self, envelope: dict[str, Any], blobs: Mapping[str, bytes] | None = None, *, gate_ref: Mapping[str, Any] | None = None) -> dict[str, Any]:
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
        return {"envelope": copy.deepcopy(r["envelope"]), "blobs": blobs, "hekb_seq": line["seq"], "hekb_hash": line["hash"],
                "gate_ref": copy.deepcopy(r["ic04"]["gate_ref"])}

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
        return {l["record"]["artifact_id"]: l for l in self.log.verify()}
