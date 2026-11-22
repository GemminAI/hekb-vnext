"""Node and edge vocabulary of the graph layer.

Every node carries a ``kind`` (what sort of statement it is) and a ``verification`` (how far it has
been checked) from the first version, even though v0.1 only writes observations that are
``machine_verified`` or ``unverified`` (D-69).
"""

from __future__ import annotations

from typing import Any, Final

#: What a node says. ``observation``: something measured or recorded. ``claim``: an assertion made
#: from observations (not written in v0.1). ``evaluation``: a judgement against content, evidence,
#: intent or outcome (not written in v0.1).
KINDS: Final = ("observation", "claim", "evaluation")

#: How far a node has been checked.
#:   unverified        written, no check was made (or the check could not be made)
#:   machine_verified  a machine recomputed or compared it against an independent record
#:   human_verified    a person checked it (HANKO), recorded as a decision node
#:   contested         checked and found in conflict with another node
#:   retracted         withdrawn; stays in the log (records are not rewritten, D-18)
VERIFICATIONS: Final = ("unverified", "machine_verified", "human_verified", "contested", "retracted")

#: What v0.1 may write.
V01_ALLOWED: Final = frozenset(
    {("observation", "machine_verified"), ("observation", "unverified")}
)

#: Verifications that may be handed to a RAG context as evidence (Q6). ``unverified`` never is.
EVIDENCE_VERIFICATIONS: Final = frozenset({"machine_verified", "human_verified"})

NODE_TYPES: Final = ("trajectory", "causal_log", "source_text", "hext_kernel", "policy_set")
EDGE_TYPES: Final = (
    "has_causal_log",  # trajectory -> causal_log blob (token range)
    "has_source_text",  # trajectory -> source_text blob (token range, character range)
    "token_hext",  # causal_log node (token index) -> kernel hext/1.0 object
    "measured_under",  # trajectory -> policy_set
    "same_content_as",  # trajectory <-> trajectory (same content_id; stored once, a < b)
)


class GraphError(ValueError):
    pass


def validate_node(node: dict[str, Any], *, v01: bool = True) -> None:
    for key in ("node_id", "node_type", "kind", "verification"):
        if not isinstance(node.get(key), str) or not node[key]:
            raise GraphError(f"node field {key!r} is required")
    if node["node_type"] not in NODE_TYPES:
        raise GraphError(f"unknown node_type {node['node_type']!r}")
    if node["kind"] not in KINDS:
        raise GraphError(f"kind must be one of {KINDS}, got {node['kind']!r}")
    if node["verification"] not in VERIFICATIONS:
        raise GraphError(f"verification must be one of {VERIFICATIONS}, got {node['verification']!r}")
    if v01 and (node["kind"], node["verification"]) not in V01_ALLOWED:
        raise GraphError(
            f"v0.1 writes only observation with machine_verified/unverified, got "
            f"({node['kind']}, {node['verification']})"
        )


def validate_edge(edge: dict[str, Any]) -> None:
    for key in ("type", "src", "dst"):
        if not isinstance(edge.get(key), str) or not edge[key]:
            raise GraphError(f"edge field {key!r} is required")
    if edge["type"] not in EDGE_TYPES:
        raise GraphError(f"unknown edge type {edge['type']!r}")
    if not isinstance(edge.get("attrs", {}), dict):
        raise GraphError("edge attrs must be an object")
