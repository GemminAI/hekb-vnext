"""HEXT v2 JSON -> graph projection (a derived, rebuildable layer; addendum §8).

Only the HEXT document (an Evidence record) and what it declares are projected: its claims, relations,
observation_refs and provenance. Trajectories, causal logs, Interpretations, Selections and Outcomes are never
created from a HEXT document. A HEXT ``object.id``, its ``state_hash`` and a HEKB ``object_id`` are separate
attributes; the graph node ids are derived from the ``state_hash`` only (deterministic).

Creating an edge records that an identifier resolved; whether the relation is meaningful is a separate
Verification record. A reference that does not resolve creates no edge and an UNRESOLVED record.
No wall-clock value enters the projection, so the same document gives the same projection.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from graph.chainlog import ChainLog
from graph.model import GraphError

SCHEMA = "hextproj/1"
_REF = re.compile(r"^urn:gmm:nvs:session:(?P<sid>[^#]+)(?:#seq=(?P<a>\d+)-(?P<b>\d+))?$")

#: node-id derivation (table in the addendum)
DOC_PREFIX, CLAIM_PREFIX = "hextdoc:", "hextclaim:"


def doc_node_id(state_hash: str) -> str:
    return DOC_PREFIX + state_hash


def claim_node_id(state_hash: str, time_id: str) -> str:
    return f"{CLAIM_PREFIX}{state_hash}#{time_id}"


def _strict_load(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        d: dict[str, Any] = {}
        for k, v in items:
            if k in d:
                raise GraphError(f"duplicate key {k!r}")
            d[k] = v
        return d

    def no_float(s: str) -> Any:
        raise GraphError(f"float {s} is not a sealable value")

    doc = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_float=no_float)
    if not isinstance(doc, dict) or doc.get("hext") != "2.0.0" or doc.get("kind") not in ("source", "annotation"):
        raise GraphError("not a HEXT 2.0.0 source/annotation document")
    return doc


def project_document(
    raw: bytes,
    *,
    resolve_session: Callable[[str], str | None],
    known_doc_hashes: Iterable[str] = (),
    known_claims: dict[str, set[str]] | None = None,
    hekb_object_id: str | None = None,
    validated_level: int | None = None,
    epistemic_status: str = "derived",
) -> dict[str, Any]:
    """Pure function of the document bytes and the two lookups. Returns {nodes, edges, unresolved}."""
    doc = _strict_load(raw)
    sh = doc["final_seal"]["state_hash"]
    did = doc_node_id(sh)
    known = set(known_doc_hashes)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    meas = doc.get("measurement") or {}
    attrs: dict[str, Any] = {
        "d70_kind": "Evidence", "epistemic_status": epistemic_status, "hext": doc["hext"], "hext_kind": doc["kind"],
        "hext_object_id": doc["object"]["id"], "state_hash": sh,
        "blob_sha256": hashlib.sha256(raw).hexdigest(),
        "hekb_object_id": hekb_object_id, "validated_level": validated_level,
        "object_type": doc["object"]["type"], "lifecycle": doc["object"]["lifecycle"],
        "contributors": doc["provenance"]["contributors"], "agency": doc["provenance"]["agency"],
        "observed_at": doc["time"]["observed_at"], "recorded_at": doc["time"]["recorded_at"],
        "space_id": meas.get("space_id"), "calibration": meas.get("calibration"),
        "causal_log_ref": meas.get("causal_log_ref"),
        "evidence_ids": [e["evidence_id"] for e in doc["evidence"]],
        "sealed_at": doc["final_seal"]["sealed_at"],
    }
    nodes.append({"node_id": did, "node_type": "hext_document", "kind": "observation", "verification": "unverified",
                  "attrs": attrs})
    claim_ids: dict[str, str] = {}
    for c in doc["time"]["claims"]:
        cid = claim_node_id(sh, c["time_id"])
        claim_ids[c["time_id"]] = cid
        nodes.append({"node_id": cid, "node_type": "hext_claim", "kind": "observation", "verification": "unverified",
                      "attrs": {"d70_kind": "Evidence", "epistemic_status": epistemic_status, "time_id": c["time_id"],
                                "frame": c["frame"], "expression": c["expression"], "status": c["status"],
                                "value": c.get("value"), "event_ref": c.get("event_ref"), "source": c.get("source"),
                                "anchor_ref": c.get("anchor_ref"), "document": did}})
        edges.append({"type": "has_claim", "src": did, "dst": cid, "attrs": {"time_id": c["time_id"]}})
    # an annotation's relations may name claims of the source it derives from (only when that source is present)
    inherited: dict[str, str] = {}
    for ref in doc["provenance"]["derived_from"]:
        for tid in sorted((known_claims or {}).get(ref["state_hash"], ())):
            inherited.setdefault(tid, claim_node_id(ref["state_hash"], tid))
    for r in doc["time"]["relations"]:
        a, b = claim_ids.get(r["a"]) or inherited.get(r["a"]), claim_ids.get(r["b"]) or inherited.get(r["b"])
        if a is None or b is None:
            unresolved.append({"kind": "relation_endpoint", "document": did, "ref": f"{r['a']}->{r['b']}",
                               "rel_id": r["rel_id"], "reason": "a relation endpoint is not a claim of this document "
                                         "or of a present source document"})
            continue
        edges.append({"type": "time_relation", "src": a, "dst": b,
                      "attrs": {"rel_id": r["rel_id"], "relation": r["relation"], "offset": r.get("offset"),
                                "status": r["status"], "document": did}})
    for ref in doc["provenance"]["derived_from"]:
        target = doc_node_id(ref["state_hash"])
        if ref["state_hash"] in known:
            edges.append({"type": "derived_from", "src": did, "dst": target, "attrs": {"hext_object_id": ref["id"]}})
        else:
            unresolved.append({"kind": "derived_from", "document": did, "ref": ref["state_hash"],
                               "reason": "the source document is not in the projected set"})
    for ref_s in meas.get("observation_refs", []):
        m = _REF.match(ref_s)
        if not m:
            unresolved.append({"kind": "observation_ref", "document": did, "ref": ref_s, "reason": "unparsable reference"})
            continue
        traj = resolve_session(m.group("sid"))
        if traj is None:
            unresolved.append({"kind": "observation_ref", "document": did, "ref": ref_s, "session_id": m.group("sid"),
                               "reason": "no stored trajectory with this session id in the graph"})
            continue
        edges.append({"type": "observation_ref", "src": did, "dst": traj,
                      "attrs": {"session_id": m.group("sid"), "ref": ref_s,
                                "seq_from": int(m.group("a")) if m.group("a") else None,
                                "seq_to": int(m.group("b")) if m.group("b") else None}})
    return {"document": did, "nodes": nodes, "edges": edges, "unresolved": unresolved}


class HextProjectionLayer:
    """The projection as its own append-only, hash-chained logs in ``<data_dir>/graph/hext_projection/``.
    ``rebuild`` writes a fresh copy and swaps it in; the main graph logs (Verification, ...) are untouched."""

    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir) / "graph" / "hext_projection"

    def rebuild(
        self,
        docs: list[tuple[bytes, dict[str, Any]]],
        *,
        resolve_session: Callable[[str], str | None],
    ) -> dict[str, Any]:
        """docs: (raw bytes, {hekb_object_id, validated_level, epistemic_status})."""
        parsed = sorted(docs, key=lambda d: hashlib.sha256(d[0]).hexdigest())
        loaded = [_strict_load(raw) for raw, _m in parsed]
        hashes = [d["final_seal"]["state_hash"] for d in loaded]
        claims = {d["final_seal"]["state_hash"]: {c["time_id"] for c in d["time"]["claims"]} for d in loaded}
        tmp = self.root.with_name("hext_projection.tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        nodes, edges, unres = ChainLog(tmp / "nodes.jsonl"), ChainLog(tmp / "edges.jsonl"), ChainLog(tmp / "unresolved.jsonl")
        seen_nodes: set[str] = set()
        for raw, meta in parsed:
            p = project_document(raw, resolve_session=resolve_session, known_doc_hashes=hashes, known_claims=claims, **meta)
            for n in p["nodes"]:
                if n["node_id"] not in seen_nodes:
                    seen_nodes.add(n["node_id"])
                    nodes.append(n)
            for e in p["edges"]:
                edges.append(e)
            for u in p["unresolved"]:
                unres.append(u)
        for log in (nodes, edges, unres):
            log.sync()
        manifest = {"schema": SCHEMA, "documents": len(parsed), "nodes": len(nodes), "edges": len(edges),
                    "unresolved": len(unres),
                    "heads": {"nodes": nodes.head, "edges": edges.head, "unresolved": unres.head},
                    "input_blob_sha256": sorted(hashlib.sha256(r).hexdigest() for r, _m in parsed)}
        (tmp / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=1))
        if self.root.exists():
            shutil.rmtree(self.root)
        tmp.rename(self.root)
        return manifest

    def manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = json.loads((self.root / "manifest.json").read_text())
        return out

    def nodes(self) -> list[dict[str, Any]]:
        return list(ChainLog(self.root / "nodes.jsonl").records())

    def edges(self) -> list[dict[str, Any]]:
        return list(ChainLog(self.root / "edges.jsonl").records())

    def unresolved(self) -> list[dict[str, Any]]:
        return list(ChainLog(self.root / "unresolved.jsonl").records())
