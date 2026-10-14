"""Verification: a recorded, machine-checked RELATION between existing records (addendum to GRAPH_SEARCH v0).

Nothing here changes an existing node. A verification records which inputs (by sha256) and which rule
(id, version, sha256) were used, the status, and ``verified_means`` -- the exact condition that was
confirmed (typically an identifier match). It never asserts that two records mean the same thing and never a
causal relation. Records carry no wall-clock time, so the same inputs and rules give the same record.

Statuses
  VERIFIED       the rule's condition held for the named inputs
  FAILED         the rule ran and its condition did not hold
  NOT_EVALUATED  the rule could not run (a required input is absent); ``reason`` says why
  UNRESOLVED     a reference's target could not be confirmed to exist in the searched scope (the search scope
                 is part of the record); this is not FAILED and not "the target does not exist"
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Collection
from typing import Any

from graph.chainlog import canonical
from graph.model import EPISTEMIC_STATUSES, VERIFICATION_STATUSES, GraphError
from graph.store import GraphStore

RULES: dict[str, dict[str, str]] = {
    "R-VER-RAW-01": {
        "version": "1",
        "text": "causal-log blob <- raw log: the raw file's sha256 equals the stored source_file sha256, and for every "
                "record its raw-line sha256 and line number equal the stored provenance.raw_ref",
    },
    "R-VER-DOC-TRAJ-01": {
        "version": "1",
        "text": "HEXT v2 document <-> trajectory: the session id and token range in measurement.observation_refs and "
                "measurement.space_id equal the stored trajectory's, and the document verified at level 2",
    },
    "R-VER-DOC-CL-01": {
        "version": "1",
        "text": "HEXT v2 document <-> causal log by measurement.causal_log_ref (NOT_EVALUATED if the document has none)",
    },
    "R-VER-ID-HEXT10-01": {
        "version": "1",
        "text": "hext/1.0 reference identity: cycle_raw.hext.object_id == ports.P01.raw.parent_event_id == "
                "provenance.cycle_provenance[0].hext_object_id of the same record",
    },
    "R-VER-OBSREF-RESOLVE-01": {
        "version": "1",
        "text": "a HEXT measurement.observation_refs entry resolves to a stored trajectory (session id found in the "
                "graph); an unresolved entry creates no edge and is recorded as UNRESOLVED",
    },
    "R-VER-DOC-TOKEN-01": {
        "version": "1",
        "text": "token containment: a HEXT document's observation_ref names the token interval [seq_from, seq_to] "
                "(0-based, both ends inclusive, token_index of the stored causal log) of a trajectory that has n_steps "
                "tokens; VERIFIED if 0 <= seq_from <= seq_to <= n_steps-1, FAILED if outside, NOT_EVALUATED if the "
                "reference has no #seq part, UNRESOLVED if the session does not resolve",
    },
    "R-VER-REF-RESOLVE-01": {
        "version": "1",
        "text": "a hext/1.0 object_id resolves to a stored document: an id equal to a stored HEXT document's object.id, "
                "state_hash or blob sha256, or to a stored object_id, within the stated search scope",
    },
}


def rule_ref(rule_id: str) -> dict[str, str]:
    r = RULES[rule_id]
    body = {"id": rule_id, "version": r["version"], "text": r["text"]}
    return {"id": rule_id, "version": r["version"], "sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}


def make_verification(
    *,
    rule_id: str,
    subject_a: str,
    subject_b: str | None,
    status: str,
    verified_means: str,
    inputs: list[dict[str, str]],
    detail: dict[str, Any] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if status not in VERIFICATION_STATUSES:
        raise GraphError(f"verification status must be one of {VERIFICATION_STATUSES}, got {status!r}")
    if not inputs:
        raise GraphError("a verification names its inputs (sha256)")
    for i in inputs:
        h = i.get("sha256", "")
        if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
            raise GraphError(f"input {i.get('role')!r}: sha256 must be 64 hex characters")
    if status in ("NOT_EVALUATED", "UNRESOLVED", "FAILED") and not (reason and reason.strip()):
        raise GraphError(f"{status} needs a reason")
    if status == "VERIFIED" and not verified_means.strip():
        raise GraphError("VERIFIED needs verified_means (what was confirmed)")
    rule = rule_ref(rule_id)
    body = {"rule": rule, "a": subject_a, "b": subject_b, "inputs": sorted(inputs, key=lambda x: (x["role"], x["sha256"])),
            "status": status, "verified_means": verified_means, "detail": detail or {}, "reason": reason}
    vid = "ver:" + hashlib.sha256(canonical(body).encode()).hexdigest()
    return {"verification_id": vid, "epistemic_status": "verified_relation", **body}


def write_verification(store: GraphStore, ver: dict[str, Any], known_ids: Collection[str] = ()) -> bool:
    """Append the verification as a node and link it to its subjects. Existing nodes are never touched."""
    if ver.get("epistemic_status") != "verified_relation" or "verified_relation" not in EPISTEMIC_STATUSES:
        raise GraphError("a verification record is epistemic_status=verified_relation")
    for sid in (ver["a"], ver["b"]):
        if sid is not None and not store.has_node(sid) and sid not in known_ids:
            raise GraphError(f"subject node {sid!r} does not exist (add it first; verification never creates observations)")
    attrs = {k: v for k, v in ver.items() if k != "verification_id"}
    created = store.add_node(
        ver["verification_id"], "verification", kind="evaluation",
        verification="unverified" if ver["status"] == "NOT_EVALUATED" else "machine_verified",
        attrs=attrs,
    )
    store.add_edge("verification_input", ver["verification_id"], ver["a"], {"role": "subject_a"})
    if ver["b"] is not None:
        store.add_edge("verification_input", ver["verification_id"], ver["b"], {"role": "subject_b"})
    return created


def unresolved_reverse_index(store: GraphStore) -> dict[str, list[dict[str, Any]]]:
    """target id -> the records that reference it and could not be resolved (rebuildable from the graph)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for n in store.nodes():
        if n["node_type"] != "verification":
            continue
        a = n["attrs"]
        if a["status"] == "UNRESOLVED" and a["rule"]["id"] == "R-VER-REF-RESOLVE-01":
            d = a["detail"]
            out[d["target_id"]].append({
                "verification_id": n["node_id"], "session": d.get("session"), "token_index": d.get("token_index"),
                "referrer": a["a"], "search_scope": d.get("search_scope"),
            })
    return {k: sorted(v, key=lambda x: (str(x["session"]), x["token_index"] or 0)) for k, v in sorted(out.items())}


def epistemic_status(node: dict[str, Any]) -> str:
    """Epistemic position of a stored node: explicit attrs.epistemic_status, else the default of its type."""
    explicit = node.get("attrs", {}).get("epistemic_status")
    if explicit:
        return str(explicit)
    return {"verification": "verified_relation", "hext_document": "derived"}.get(node["node_type"], "observed")


def dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
