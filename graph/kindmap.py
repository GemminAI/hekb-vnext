"""node_type -> the ``kind`` that is stored, and the D-70 class (CTP data model v0.3.3 section 1).

One table, two columns:

* ``stored kind``: the value a *new* write must carry for that ``node_type`` (what the code has
  always written). ``graph.guard.put`` refuses a node whose ``node_type`` is not in the table or whose
  ``kind`` differs, on both entrances (``GraphStore`` and the HEXT projection layer).
* ``D-70 class``: observation, evidence, interpretation, selection or outcome. It is read from the
  ``node_type`` only, never from the stored ``kind``, so records written before D-70 (``claim``,
  ``evaluation``, ``observation`` kinds) classify correctly and are not rewritten.

``KINDS``, ``V01_ALLOWED`` and ``validate_node`` in ``graph/model.py`` are unchanged.
The four rows that v0.3.3 section 1 does not list (causal_log, raw_log, source_text, policy_set) were
decided by Tomo on 2026-11-03 (option A): D-70 Evidence is HEXT only, so they are observations; a
declaration becomes evidence when it is sealed as a HEXT v2 source.
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


DECISION_A: Final = (
    "Tomo decision 2026-11-03 (option A: D-70 Evidence is HEXT only; "
    "a declaration is evidence when sealed as a HEXT v2 source)"
)

KIND_MAP: Final[Mapping[str, KindRule]] = {
    "trajectory": KindRule("observation", "observation", "v0.3.3 s1"),
    "hext_document": KindRule("observation", "evidence", "v0.3.3 s1"),
    "hext_claim": KindRule("observation", "evidence", "v0.3.3 s1"),
    "hext_kernel": KindRule("observation", "evidence", "v0.3.3 s1"),
    "verification": KindRule("evaluation", "interpretation", "v0.3.3 s1"),
    "axis_verification": KindRule("interpretation", "interpretation", "v0.3.2 s2"),
    # CTP episode records, first half (graph/ctp.py): the stored kind is the D-70 word of the class
    "episode": KindRule("observation", "observation", "CTP v0.3 s3"),
    "context_observation": KindRule("observation", "observation", "CTP v0.3 s3"),
    "step_output": KindRule("observation", "observation", "CTP v0.3 s3"),
    "applicability": KindRule("interpretation", "interpretation", "CTP v0.3 s3"),
    "carry_evidence": KindRule("interpretation", "interpretation", "CTP v0.3 s3"),
    "cle_result": KindRule("interpretation", "interpretation", "CTP v0.3 s3"),
    "compliance": KindRule("interpretation", "interpretation", "CTP v0.3 s3"),
    "plan": KindRule("selection", "selection", "CTP v0.3 s3"),
    "governance": KindRule("selection", "selection", "CTP v0.3 s3"),
    "outcome": KindRule("outcome", "outcome", "CTP v0.3 s3"),
    "assignment": KindRule("selection", "selection", "CTP v0.3 s3"),
    "experience": KindRule("selection", "selection", "CTP v0.3 s3"),
    # not in the v0.3.3 table; decided by Tomo (see DECISION_A)
    "causal_log": KindRule("observation", "observation", DECISION_A),
    "source_text": KindRule("observation", "observation", DECISION_A),
    "policy_set": KindRule("observation", "observation", DECISION_A),
    "raw_log": KindRule("observation", "observation", DECISION_A),
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
