"""``axis_verification``: a record of one axis' check of one target, and how the current state is read.

CTP data model v0.3 section 5, v0.3.1 section 3, v0.3.2 sections 2-4.

A record is a node (``node_type="axis_verification"``, ``kind="interpretation"``) tied to its target by a
``verifies`` edge. It is never edited: a better judgement is a new record that ``supersedes`` (a newer
method or evaluator) or ``corrects`` (a mistake) older ones of the same target and axis. The *current state*
of ``(target, axis)`` is computed from the records nobody has replaced:

  none               -> ``pending``
  all the same value -> that value
  different values   -> ``conflict`` (every valid record is returned; none is chosen, and no clock is read)

Node record (the existing five fields; the CTP fields live in ``attrs``)::

  node_id      "axis_verification:" + sha256(JCS({axis, target, input_hash, result_hash, producer_fingerprint}))
  node_type    "axis_verification"
  kind         "interpretation"
  verification "unverified"  (a constant: the column is not where this state lives, see below)
  attrs        {axis, value, method, detail_ref, target, input_hash, producer}

``input_hash`` = sha256(JCS([{node_id: target, record_hash: <D13 hash of the target's record>}]));
``result_hash`` = sha256(JCS({value, method, detail_ref})); ``producer_fingerprint`` = sha256(JCS(producer)).
``attrs`` pass through ``graph.jcs``: a float or an integer outside +-(2**53 - 1) is refused (only here).

Why ``verification`` is the constant ``"unverified"``: the column means "was this record itself checked"
(the existing writers: ``machine_verified`` when a machine recomputed it, ``unverified`` otherwise), the v0.3
data model keeps verification state out of node fields and in these records, and a value that depended on
the axis or on ``value`` would make the same fact two different records.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from graph.jcs import JcsError, canonical_bytes, canonicalize
from graph.kindmap import KindMapError, d70_class
from graph.model import GraphError

NODE_TYPE: Final = "axis_verification"
STORED_KIND: Final = "interpretation"
STORED_VERIFICATION: Final = "unverified"

#: v0.3 section 5: the four axes and their values.
AXIS_VALUES: Final[Mapping[str, tuple[str, ...]]] = {
    "structural": ("passed", "failed", "pending", "not_applicable"),
    "integrity": ("sealed", "invalid", "unverified", "not_applicable"),
    "semantic": ("verified", "contradicted", "undetermined", "pending", "not_applicable"),
    "independent": ("audited", "rejected", "pending", "not_applicable"),
}
AXES: Final = tuple(AXIS_VALUES)
PENDING: Final = "pending"
CONFLICT: Final = "conflict"
REPLACING_EDGES: Final = ("supersedes", "corrects")
VERIFIABLE_CLASSES: Final = ("evidence", "interpretation")
PRODUCER_FIELDS: Final = ("component", "version", "config_sha256")


class AxisVerificationError(GraphError):
    """An axis_verification record that breaks the rules of its type."""


class EdgeRuleError(GraphError):
    """A verifies / supersedes / corrects edge that breaks the rules of its type."""


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def input_hash(target: str, target_record_hash: str) -> str:
    return _sha([{"node_id": target, "record_hash": target_record_hash}])


def result_hash(value: str, method: Any, detail_ref: str | None) -> str:
    return _sha({"value": value, "method": method, "detail_ref": detail_ref})


def producer_fingerprint(producer: Mapping[str, Any]) -> str:
    return _sha(dict(producer))


def axis_node_id(axis: str, target: str, in_hash: str, res_hash: str, fingerprint: str) -> str:
    identity = {
        "axis": axis,
        "target": target,
        "input_hash": in_hash,
        "result_hash": res_hash,
        "producer_fingerprint": fingerprint,
    }
    return f"{NODE_TYPE}:{_sha(identity)}"


def build_node(
    *,
    axis: str,
    value: str,
    method: Any,
    detail_ref: str | None,
    target: str,
    target_record_hash: str,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    """The node record for one check. Nothing here reads a clock or a random source."""
    in_h = input_hash(target, target_record_hash)
    attrs = {
        "axis": axis,
        "value": value,
        "method": method,
        "detail_ref": detail_ref,
        "target": target,
        "input_hash": in_h,
        "producer": dict(producer),
    }
    node_id = axis_node_id(axis, target, in_h, result_hash(value, method, detail_ref), producer_fingerprint(producer))
    return {
        "node_id": node_id,
        "node_type": NODE_TYPE,
        "kind": STORED_KIND,
        "verification": STORED_VERIFICATION,
        "attrs": attrs,
    }


def check_node(record: Mapping[str, Any], target_record_hash: Callable[[str], str | None]) -> None:
    """Refuse a new axis_verification node that breaks its rules (called under the graph lock)."""
    node_id = str(record.get("node_id"))
    attrs = record.get("attrs")

    def refuse(reason: str) -> AxisVerificationError:
        return AxisVerificationError(f"axis_verification {node_id!r}: {reason}")

    if record.get("verification") != STORED_VERIFICATION:
        raise refuse(f"the verification column is {STORED_VERIFICATION!r}")
    if not isinstance(attrs, dict):
        raise refuse("attrs must be an object")
    missing = [k for k in ("axis", "value", "method", "detail_ref", "target", "input_hash", "producer") if k not in attrs]
    if missing:
        raise refuse(f"attrs lack {missing}")
    if set(attrs) != {"axis", "value", "method", "detail_ref", "target", "input_hash", "producer"}:
        raise refuse("attrs hold fields outside the axis_verification shape")
    axis, value = attrs["axis"], attrs["value"]
    if axis not in AXIS_VALUES:
        raise refuse(f"axis {axis!r} is not one of {AXES}")
    if value not in AXIS_VALUES[axis]:
        raise refuse(f"value {value!r} is not a value of the {axis} axis {AXIS_VALUES[axis]}")
    if not isinstance(attrs["target"], str) or not attrs["target"]:
        raise refuse("target must be a node_id")
    if attrs["detail_ref"] is not None and not isinstance(attrs["detail_ref"], str):
        raise refuse("detail_ref must be a string or null")
    method = attrs["method"]
    if not (isinstance(method, str) and method) and not (isinstance(method, dict) and method):
        raise refuse("method must be a non-empty string or object")
    producer = attrs["producer"]
    if not isinstance(producer, dict) or set(producer) != set(PRODUCER_FIELDS) or not all(isinstance(producer[k], str) and producer[k] for k in PRODUCER_FIELDS):
        raise refuse(f"producer must be an object with the string fields {PRODUCER_FIELDS}")
    try:
        canonicalize(attrs)  # a float or an out-of-range integer is refused here, for this type only
    except JcsError as exc:
        raise refuse(str(exc)) from exc
    stored = target_record_hash(attrs["target"])
    if stored is None:
        raise refuse(f"target {attrs['target']!r} is not a stored node")
    expected = axis_node_id(
        axis, attrs["target"], input_hash(attrs["target"], stored),
        result_hash(value, method, attrs["detail_ref"]), producer_fingerprint(producer),
    )
    if node_id != expected or attrs["input_hash"] != input_hash(attrs["target"], stored):
        raise refuse("node_id (or input_hash) is not the one derived from the record's inputs")


class AxisIndex:
    """What the store keeps in memory about axis records: derived from its logs, never written."""

    def __init__(self) -> None:
        self.records: dict[str, dict[str, Any]] = {}  # node_id -> record of every axis_verification
        self.by_target_axis: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.replaced_by: dict[str, set[str]] = defaultdict(set)  # old id -> the ids that supersede/correct it

    def observe_node(self, record: dict[str, Any]) -> None:
        if record.get("node_type") == NODE_TYPE:
            a = record["attrs"]
            self.records[record["node_id"]] = record
            self.by_target_axis[(a["target"], a["axis"])].append(record["node_id"])

    def observe_edge(self, record: dict[str, Any]) -> None:
        if record.get("type") in REPLACING_EDGES:
            self.replaced_by[record["dst"]].add(record["src"])


def check_edge(
    record: Mapping[str, Any],
    index: AxisIndex,
    node_type_of: Callable[[str], str | None],
) -> None:
    """Refuse a new verifies / supersedes / corrects edge that breaks its rules."""
    kind, src, dst = record.get("type"), str(record.get("src")), str(record.get("dst"))
    if kind not in ("verifies", *REPLACING_EDGES):
        return

    def refuse(reason: str) -> EdgeRuleError:
        return EdgeRuleError(f"{kind} edge {src!r} -> {dst!r}: {reason}")

    start = index.records.get(src)
    if start is None:
        raise refuse("the start must be a stored axis_verification node")
    if kind == "verifies":
        dst_type = node_type_of(dst)
        if dst_type is None:
            raise refuse("the end is not a stored node")
        try:
            cls = d70_class({"node_id": dst, "node_type": dst_type, "kind": None})
        except KindMapError as exc:
            raise refuse(str(exc)) from exc
        if cls not in VERIFIABLE_CLASSES:
            raise refuse(f"the end is classified {cls!r}; only {VERIFIABLE_CLASSES} can be verified")
        if start["attrs"]["target"] != dst:
            raise refuse(f"the start's target is {start['attrs']['target']!r}, not the end")
        return
    end = index.records.get(dst)
    if end is None:
        raise refuse("the end must be a stored axis_verification node")
    if src == dst:
        raise refuse("a record cannot replace itself")
    for field in ("target", "axis"):
        if start["attrs"][field] != end["attrs"][field]:
            raise refuse(f"the two records have a different {field} ({start['attrs'][field]!r} vs {end['attrs'][field]!r})")


@dataclass(frozen=True)
class AxisState:
    target: str
    axis: str
    state: str  # a value of the axis, "pending" or "conflict"
    valid: tuple[dict[str, Any], ...]  # the valid records, sorted by node_id (all of them, also when state is conflict)


def current_state(index: AxisIndex, target: str, axis: str) -> AxisState:
    """The state of (target, axis) from the records nobody has superseded or corrected."""
    if axis not in AXIS_VALUES:
        raise AxisVerificationError(f"axis {axis!r} is not one of {AXES}")
    ids = index.by_target_axis.get((target, axis), [])
    valid = tuple(index.records[i] for i in sorted(set(ids)) if not index.replaced_by.get(i))
    values = {r["attrs"]["value"] for r in valid}
    if not valid:
        state = PENDING
    elif len(values) == 1:
        state = next(iter(values))
    else:
        state = CONFLICT
    return AxisState(target, axis, state, valid)


__all__ = [
    "AXES", "AXIS_VALUES", "CONFLICT", "PENDING", "AxisIndex", "AxisState", "AxisVerificationError",
    "EdgeRuleError", "axis_node_id", "build_node", "check_edge", "check_node", "current_state",
    "input_hash", "producer_fingerprint", "result_hash",
]
