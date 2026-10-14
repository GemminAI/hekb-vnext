"""Token alignment records, schema ``causal-align-token/2``.

Built from a v1 record (CAUSAL-ALIGN002 Phase 2) and the graph; the v1 record is never modified. Changes
(and only these) relative to v1:

 * ``hext_ref``  -> ``hext10_reference``: renamed, because it names the kernel's ``hext/1.0`` wire object, not a
   HEXT document. ``document: "ABSENT"`` becomes ``document: "NOT_STORED"`` and ``resolution`` is stated;
 * ``parties_present.hext_reference`` -> ``hext10_reference``; ``parties_present.hext_document`` ->
   ``hext_v2_document`` (True iff a HEXT v2 document contains this token);
 * added ``hext_v2_documents`` (containment, see graph/containment.py), ``schema``, ``alignment_id_v1`` (the old
   id, kept) and ``alignment_id`` (new, over the v2 body);
 * every other field (identity/temporal/semantic statuses, rules, basis, findings, ...) is copied unchanged.
"""

from __future__ import annotations

import hashlib
from typing import Any

from graph.chainlog import canonical

SCHEMA = "causal-align-token/2"
RENAMED = {"hext_ref": "hext10_reference"}


def to_v2(v1: dict[str, Any], *, containing: list[dict[str, Any]], resolution: str,
          resolution_verification: str | None, doc_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {k: v for k, v in v1.items() if k not in ("hext_ref", "parties_present", "alignment_id")}
    h = v1.get("hext_ref")
    out["hext10_reference"] = None if h is None else {
        "object_id": h["hext_object_id"], "schema_version": h["schema_version"], "source": h["source"],
        "kind": "kernel_wire_object", "document": "NOT_STORED", "resolution": resolution,
        "resolution_verification": resolution_verification,
    }
    pp = dict(v1["parties_present"])
    out["parties_present"] = {
        "trajectory": pp["trajectory"], "causal_log": pp["causal_log"],
        "hext10_reference": pp["hext_reference"], "hext_v2_document": bool(containing),
    }
    out["hext_v2_documents"] = [
        {"hextdoc": c["document"], "state_hash": doc_meta[c["document"]]["state_hash"],
         "hext_object_id": doc_meta[c["document"]]["hext_object_id"], "intervals": c["intervals"]}
        for c in containing
    ]
    out["schema"] = SCHEMA
    out["alignment_id_v1"] = v1["alignment_id"]
    out["alignment_id"] = hashlib.sha256(canonical(out).encode()).hexdigest()
    return out


def diff_v1_v2(v1: dict[str, Any], v2: dict[str, Any]) -> list[str]:
    """Differences other than the declared ones; an empty list means v2 only renames/adds as documented."""
    bad: list[str] = []
    for k, v in v1.items():
        if k in ("hext_ref", "parties_present", "alignment_id"):
            continue
        if v2.get(k) != v:
            bad.append(f"changed:{k}")
    h, n = v1.get("hext_ref"), v2.get("hext10_reference")
    if (h is None) != (n is None):
        bad.append("hext10_reference differs from hext_ref")
    elif h is not None and n is not None and (
        h["hext_object_id"] != n["object_id"] or h["schema_version"] != n["schema_version"]
    ):
        bad.append("hext10_reference differs from hext_ref")
    pp, np_ = v1["parties_present"], v2["parties_present"]
    if (pp["trajectory"], pp["causal_log"], pp["hext_reference"]) != (
        np_["trajectory"], np_["causal_log"], np_["hext10_reference"]
    ):
        bad.append("parties_present changed beyond the rename")
    if v2["alignment_id_v1"] != v1["alignment_id"]:
        bad.append("alignment_id_v1")
    return bad
