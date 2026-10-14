"""Token <-> HEXT v2 document containment (addendum §9).

A token is identified by (trajectory object_id, token_index) -- the ``token_index`` of the stored causal log,
0-based, which is also the row index of the board blob. A document CONTAINS a token when one of its resolved
``observation_ref`` edges names that trajectory and an interval [seq_from, seq_to] (both ends inclusive) that
holds the index. Containment is a relation derived from edges that already exist; whether the interval is
valid for the trajectory (inside 0..n_steps-1) is a separate Verification (R-VER-DOC-TOKEN-01).

Rules fixed here:
 * one fact per (document, trajectory, token): several refs of one document to the same trajectory are a
   union (set semantics), never two facts;
 * a token may be contained by several documents (different documents or versions); they are all listed,
   sorted by node id;
 * a reference without ``#seq`` names no interval: nothing is contained (NOT_EVALUATED), never "the whole session";
 * an unresolved reference (no edge) contains nothing and stays an UNRESOLVED record;
 * an interval outside the trajectory contains only the tokens that exist, and is FAILED by the verification.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.verification import make_verification


def intervals(edges: Iterable[dict[str, Any]]) -> dict[str, list[tuple[str, int, int, str]]]:
    """trajectory node id -> [(document node id, seq_from, seq_to, ref)] from resolved observation_ref edges that
    carry an interval (edges without #seq name none)."""
    out: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    for e in edges:
        if e["type"] != "observation_ref":
            continue
        a = e["attrs"]
        if a.get("seq_from") is None or a.get("seq_to") is None:
            continue
        out[e["dst"]].append((e["src"], int(a["seq_from"]), int(a["seq_to"]), a["ref"]))
    return out


def containing_documents(
    ivs: dict[str, list[tuple[str, int, int, str]]], trajectory: str, token_index: int
) -> list[dict[str, Any]]:
    """The documents that contain the token, one entry per document (union of its intervals), sorted."""
    per_doc: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for doc, lo, hi, ref in ivs.get(trajectory, []):
        if lo <= token_index <= hi:
            per_doc[doc].append((lo, hi, ref))
    return [{"document": d, "intervals": sorted(set(v))} for d, v in sorted(per_doc.items())]


def range_verification(
    *, doc_node: str, trajectory: str, n_steps: int, seq_from: int | None, seq_to: int | None, ref: str,
    doc_sha256: str, causal_log_sha256: str,
) -> dict[str, Any]:
    inputs = [{"role": "hext_document", "sha256": doc_sha256}, {"role": "causal_log", "sha256": causal_log_sha256}]
    detail = {"ref": ref, "n_steps": n_steps, "seq_from": seq_from, "seq_to": seq_to}
    if seq_from is None or seq_to is None:
        return make_verification(rule_id="R-VER-DOC-TOKEN-01", subject_a=doc_node, subject_b=trajectory,
                                 status="NOT_EVALUATED", verified_means="", inputs=inputs, detail=detail,
                                 reason="the observation_ref has no #seq part; no token interval is named")
    ok = 0 <= seq_from <= seq_to <= n_steps - 1
    return make_verification(
        rule_id="R-VER-DOC-TOKEN-01", subject_a=doc_node, subject_b=trajectory,
        status="VERIFIED" if ok else "FAILED",
        verified_means=f"0 <= {seq_from} <= {seq_to} <= {n_steps - 1}: the named token interval lies inside the "
                       "trajectory's token_index range" if ok else "",
        inputs=inputs, detail=detail,
        reason=None if ok else f"interval [{seq_from},{seq_to}] is not inside 0..{n_steps - 1}")
