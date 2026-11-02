"""node_type -> the ``kind`` that is stored, and the D-70 class (CTP data model v0.3.3 section 1).

One table, two columns:

* ``stored kind``: the value a *new* write must carry for that ``node_type`` (what the code has
  always written). ``graph.guard.put`` refuses a node whose ``node_type`` is not in the table or whose
  ``kind`` differs, on both entrances (``GraphStore`` and the HEXT projection layer).
* ``D-70 class``: observation, evidence, interpretation, selection or outcome. It is read from the
  ``node_type`` only, never from the stored ``kind``, so records written before D-70 (``claim``,
  ``evaluation``, ``observation`` kinds) classify correctly and are not rewritten.

``KINDS``, ``V01_ALLOWED`` and ``validate_node`` in ``graph/model.py`` are unchanged.
Rows marked "to confirm" are not in the v0.3.3 section 1 table; their class is read from what the
code does with them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, NamedTuple

from graph.model import GraphError

D70_CLASSES: Final = ("observation", "evidence", "interpretation", "selection", "outcome")


class KindRule(NamedTuple):
    stored_kind: str
    d70_class: str
    source: str  # where the row comes from (documentation only)


KIND_MAP: Final[Mapping[str, KindRule]] = {
    "trajectory": KindRule("observation", "observation", "v0.3.3 s1"),
    "hext_document": KindRule("observation", "evidence", "v0.3.3 s1"),
    "hext_claim": KindRule("observation", "evidence", "v0.3.3 s1"),
    "hext_kernel": KindRule("observation", "evidence", "v0.3.3 s1"),
    "verification": KindRule("evaluation", "interpretation", "v0.3.3 s1"),
    # to confirm: not in the v0.3.3 table; GRAPH_SEARCH_v0_1.md s2.4 reads these as Observation
    "causal_log": KindRule("observation", "observation", "to confirm (GRAPH_SEARCH s2.4)"),
    "source_text": KindRule("observation", "observation", "to confirm (GRAPH_SEARCH s2.4)"),
    "policy_set": KindRule("observation", "observation", "to confirm (GRAPH_SEARCH s2.4)"),
    # to confirm: in no document; only tools/graph_align001.py writes it (a pointer to a raw log)
    "raw_log": KindRule("observation", "observation", "to confirm (no document)"),
}


class KindMapError(GraphError):
    """A node whose node_type is unknown to the table, or whose kind is not the stored kind."""

    def __init__(self, node_id: str, node_type: str, kind: str | None, reason: str) -> None:
        self.node_id, self.node_type, self.kind = node_id, node_type, kind
        super().__init__(f"node {node_id!r} (node_type {node_type!r}, kind {kind!r}): {reason}")


def _rule(node: Mapping[str, Any]) -> KindRule:
    node_type = node.get("node_type")
    rule = KIND_MAP.get(node_type) if isinstance(node_type, str) else None
    if rule is None:
        raise KindMapError(str(node.get("node_id")), str(node_type), node.get("kind"), "node_type is not in the kind map")
    return rule


def check_stored_kind(node: Mapping[str, Any]) -> None:
    """Refuse a new node whose node_type is unknown or whose kind is not the stored kind of its type."""
    rule = _rule(node)
    if node.get("kind") != rule.stored_kind:
        raise KindMapError(
            str(node.get("node_id")), str(node["node_type"]), node.get("kind"),
            f"kind must be {rule.stored_kind!r} for this node_type",
        )


def d70_class(node: Mapping[str, Any]) -> str:
    """The D-70 class of a stored node, from its node_type (not from its stored kind)."""
    return _rule(node).d70_class
