"""CTP episode records, first half (CTP001 Stage D): ten node types and their edges.

CTP common data model v0.3 (sections 2-4, 7, 8) with v0.3.1-v0.3.4; v0.3.4 decides where things live:

* Every record keeps the existing five fields; the CTP common fields live in ``attrs`` next to the type's own
  fields: ``schema`` (``"ctp.datamodel/0.3"``), ``refs``, ``producer`` ``{component, version, config_sha256}`` and
  ``observed_at`` (a label that is never part of an id and never orders anything).
* ``kind`` is the D-70 word of the type and the ``verification`` column is the constant ``"unverified"`` (as for
  ``axis_verification``). Write with ``GraphStore(v01=False)``.
* ``attrs`` pass through ``graph.jcs``: a floating-point number or an integer beyond +-(2**53 - 1) is refused.

====================  ===============  ===========================================================
node_type             D-70 / kind      what it records
====================  ===============  ===========================================================
episode               observation      one run (spec hash, item, condition, replicate, seed)
context_observation   observation      the input context before any experience (never experience)
step_output           observation      the output of a stage f1/f2/f3 and the input it was given
applicability         interpretation   whether a constraint applies to this plan
carry_evidence        interpretation   whether a stage kept a property (declared or by evidence)
cle_result            interpretation   one CLE rule's result for an episode
compliance            interpretation   a compliance judgement at level L1/L2/L3/f3
plan                  selection        the plan the Planner chose (no probability: unobserved)
governance            selection        admit or hold before execution (no probability at all)
outcome               outcome          the result: two-axis class and its parts (no reward)
====================  ===============  ===========================================================

``node_id`` (v0.3 section 4):

* observations: ``"<type>:" + sha256(JCS(identity fields))`` (an episode's identity includes the spec hash);
* judgements and selections: ``"<type>:" + sha256(JCS({identity fields, input_hash, result_hash,
  producer_fingerprint}))``. ``input_hash`` is the hash of the sorted list of the inputs: every input node as
  ``{node_id, record_hash}`` and every input that is not a node (case facts, a quota state) as
  ``{fact, value}``; a governance record also lists ``referenced_inputs`` (the names of what it really read).

Nothing here reads a clock or a random source. ``nondeterministic_groups`` lists judgements that share an identity,
inputs and evaluator but differ in result; saving them is never refused.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from graph.axis import PRODUCER_FIELDS, producer_fingerprint
from graph.blobs import blob_path
from graph.jcs import JcsError, canonical_bytes, canonicalize
from graph.kindmap import KindMapError, d70_class
from graph.model import GraphError

SCHEMA: Final = "ctp.datamodel/0.3"
STORED_VERIFICATION: Final = "unverified"

# ---- vocabularies (v0.3 section 8) ---------------------------------------------------------------------
SPLITS: Final = ("fit", "dev", "locked")
STEPS: Final = ("f1", "f2", "f3")
STATUSES: Final = ("preserved", "lost")
BASES: Final = ("declared", "evidence")
APPLICABILITIES: Final = ("applicable", "not_applicable", "undetermined")
VALUES: Final = ("PASS", "FAIL", "UNVERIFIABLE")
RULES: Final = tuple(f"R{n}" for n in range(1, 10))
MODES: Final = ("M1", "M2")
LEVELS: Final = ("L1", "L2", "L3", "f3")
JUDGES: Final = ("verifier", "independent")
DECISIONS: Final = ("admit", "hold")
TWO_AXIS: Final = ("CARRIED_COMPLIED", "CARRIED_VIOLATED", "LOST_COMPLIED", "LOST_VIOLATED", "UNDETERMINED")
UNDETERMINED_AXES: Final = ("retention", "compliance", "both")
OUTCOME_FIELDS: Final = (
    "carry", "constraint_compliance", "two_axis", "undetermined_axis", "violation_type", "severity",
    "copied_from", "task_quality", "execution_cost",
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_BLOB_REF = re.compile(r"^blob:([0-9a-f]{64})$")


class CtpError(GraphError):
    """A CTP record or edge that breaks the rules of its type."""


@dataclass(frozen=True)
class Spec:
    d70: str  # the D-70 class and the stored kind
    required: tuple[str, ...]  # the type's own fields that must be present
    optional: tuple[str, ...]
    ident: tuple[str, ...]  # fields of the identity
    result: tuple[str, ...] = ()  # fields of result_hash (judgements and selections only)

    @property
    def judgement(self) -> bool:
        return bool(self.result)


SPECS: Final[Mapping[str, Spec]] = {
    "episode": Spec(
        "observation",
        ("experiment_spec_hash", "item_id", "bundle_id", "split", "condition", "replicate", "rng_seed"),
        ("space_id",),
        ("experiment_spec_hash", "item_id", "condition", "replicate", "rng_seed"),
    ),
    "context_observation": Spec(
        "observation",
        ("episode_id", "context_ref", "knowledge_ids", "constraint_annotation_ids", "task_ref", "contains_experience"),
        (),
        ("episode_id", "context_ref"),
    ),
    "step_output": Spec(
        "observation",
        ("episode_id", "step", "text_ref", "input_context_ref", "input_context_sha256"),
        (),
        ("episode_id", "step", "text_ref", "input_context_sha256"),
    ),
    "applicability": Spec(
        "interpretation",
        ("episode_id", "constraint_id", "applicability", "applicability_basis"),
        (),
        ("episode_id", "constraint_id"),
        ("applicability", "applicability_basis"),
    ),
    "carry_evidence": Spec(
        "interpretation",
        ("episode_id", "step", "property", "status", "basis", "evidence_ref"),
        ("source",),
        ("episode_id", "step", "property", "basis"),
        ("status", "evidence_ref"),
    ),
    "cle_result": Spec(
        "interpretation",
        ("episode_id", "rule", "mode", "value", "reason_codes", "locations", "history_hash"),
        (),
        ("episode_id", "rule", "mode"),
        ("value", "reason_codes", "locations"),
    ),
    "compliance": Spec(
        "interpretation",
        ("episode_id", "level", "judge", "value", "reason_codes"),
        (),
        ("episode_id", "level", "judge"),
        ("value", "reason_codes"),
    ),
    "plan": Spec(
        "selection",
        ("episode_id", "plan_id", "steps", "probability_source"),
        (),
        ("episode_id",),
        ("plan_id", "steps"),
    ),
    "governance": Spec(
        "selection",
        ("episode_id", "plan_id", "decision", "reason_codes", "referenced_inputs"),
        (),
        ("episode_id", "plan_id"),
        ("decision", "reason_codes"),
    ),
    "outcome": Spec(
        "outcome",
        ("episode_id", "carry", "constraint_compliance", "two_axis"),
        tuple(f for f in OUTCOME_FIELDS if f not in ("carry", "constraint_compliance", "two_axis")),
        ("episode_id",),
        OUTCOME_FIELDS,
    ),
}
CTP_NODE_TYPES: Final = tuple(SPECS)
COMMON: Final = ("schema", "refs", "producer", "observed_at")
JUDGEMENT_INPUTS: Final = ("input_nodes", "input_facts", "input_hash")
CTP_EDGE_TYPES: Final = (
    "under_manifest", "has_context", "has_step", "retrieved", "interprets", "selects", "considered", "outcome_of", "audits",
)
INTERPRETATIONS: Final = ("applicability", "carry_evidence", "cle_result", "compliance")
SELECTIONS: Final = ("plan", "governance")


class Context(Protocol):
    """What a check may look up about the stored graph (main log and the HEXT projection layer)."""

    def node_type(self, node_id: str) -> str | None: ...

    def attrs(self, node_id: str) -> Mapping[str, Any] | None: ...

    def record_hash(self, node_id: str) -> str | None: ...

    def data_dir(self) -> Path: ...


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# ---- ids and hashes -----------------------------------------------------------------------------------
def input_entries(input_nodes: Mapping[str, str], input_facts: Mapping[str, Any], referenced: list[str] | None = None) -> list[Any]:
    """The input list, sorted: ``{node_id, record_hash}`` per node, ``{fact, value}`` per non-node input."""
    entries: list[Any] = [{"node_id": n, "record_hash": h} for n, h in input_nodes.items()]
    entries += [{"fact": name, "value": value} for name, value in input_facts.items()]
    if referenced is not None:
        entries.append({"referenced_inputs": sorted(referenced)})
    return sorted(entries, key=canonicalize)


def _input_names(node_ids: list[str], facts: Mapping[str, Any]) -> list[str]:
    return sorted([*node_ids, *facts])


def identity_of(spec: Spec, attrs: Mapping[str, Any]) -> dict[str, Any]:
    return {f: attrs[f] for f in spec.ident}


def result_hash(spec: Spec, attrs: Mapping[str, Any]) -> str:
    return _sha({f: attrs.get(f) for f in spec.result})


def node_id_of(node_type: str, attrs: Mapping[str, Any]) -> str:
    spec = SPECS[node_type]
    identity = identity_of(spec, attrs)
    if spec.judgement:
        identity = {
            **identity,
            "input_hash": attrs["input_hash"],
            "result_hash": result_hash(spec, attrs),
            "producer_fingerprint": producer_fingerprint(attrs["producer"]),
        }
    return f"{node_type}:{_sha(identity)}"


def build_node(
    node_type: str,
    fields: Mapping[str, Any],
    *,
    producer: Mapping[str, str],
    hash_of: Callable[[str], str | None],
    input_nodes: tuple[str, ...] = (),
    input_facts: Mapping[str, Any] | None = None,
    refs: Mapping[str, Any] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """The node record for one CTP record. ``hash_of`` gives the stored record hash of each input node."""
    spec = SPECS[node_type]
    attrs: dict[str, Any] = {"schema": SCHEMA, **fields, "refs": dict(refs or {}), "producer": dict(producer), "observed_at": observed_at}
    if spec.judgement:
        facts = dict(input_facts or {})
        ids = sorted(set(input_nodes))
        hashes = {}
        for n in ids:
            h = hash_of(n)
            if h is None:
                raise CtpError(f"{node_type}: input node {n!r} is not stored")
            hashes[n] = h
        if node_type == "governance" and "referenced_inputs" not in attrs:
            attrs["referenced_inputs"] = _input_names(ids, facts)
        attrs["input_nodes"], attrs["input_facts"] = ids, facts
        referenced = attrs["referenced_inputs"] if node_type == "governance" else None
        attrs["input_hash"] = _sha(input_entries(hashes, facts, referenced))
    return {
        "node_id": node_id_of(node_type, attrs),
        "node_type": node_type,
        "kind": spec.d70,
        "verification": STORED_VERIFICATION,
        "attrs": attrs,
    }


# ---- node checks ------------------------------------------------------------------------------------------
def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [k for k in value if isinstance(k, str)] + [x for v in value.values() for x in _walk_keys(v)]
    if isinstance(value, list):
        return [x for v in value for x in _walk_keys(v)]
    return []


def _is_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _str_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_str(x) for x in v)


def _sorted_unique(v: Any) -> bool:
    return _str_list(v) and v == sorted(set(v))


def check_node(record: Mapping[str, Any], ctx: Context) -> None:
    """Refuse a new CTP node that breaks its type's rules (called under the graph lock)."""
    node_type = str(record.get("node_type"))
    spec = SPECS[node_type]
    node_id = str(record.get("node_id"))

    def refuse(reason: str) -> CtpError:
        return CtpError(f"{node_type} {node_id!r}: {reason}")

    if record.get("verification") != STORED_VERIFICATION:
        raise refuse(f"the verification column is {STORED_VERIFICATION!r}")
    attrs = record.get("attrs")
    if not isinstance(attrs, dict):
        raise refuse("attrs must be an object")
    allowed = {*spec.required, *spec.optional, *COMMON, *(JUDGEMENT_INPUTS if spec.judgement else ())}
    missing = [k for k in (*spec.required, *COMMON, *(JUDGEMENT_INPUTS if spec.judgement else ())) if k not in attrs]
    if missing:
        raise refuse(f"attrs lack {missing}")
    _check_forbidden(node_type, attrs, refuse)  # named first: "no probability" says more than "outside the type"
    extra = sorted(set(attrs) - allowed)
    if extra:
        raise refuse(f"attrs hold fields outside the type: {extra}")
    try:
        canonicalize(attrs)  # a float or an out-of-range integer is refused here, for the CTP types only
    except JcsError as exc:
        raise refuse(str(exc)) from exc
    if attrs["schema"] != SCHEMA:
        raise refuse(f"schema must be {SCHEMA!r}")
    if not isinstance(attrs["refs"], dict):
        raise refuse("refs must be an object")
    producer = attrs["producer"]
    if not isinstance(producer, dict) or set(producer) != set(PRODUCER_FIELDS) or not all(_is_str(producer[k]) for k in PRODUCER_FIELDS):
        raise refuse(f"producer must be an object with the string fields {PRODUCER_FIELDS}")
    if attrs["observed_at"] is not None and not _is_str(attrs["observed_at"]):
        raise refuse("observed_at must be a string or null")
    _check_fields(node_type, attrs, ctx, refuse)
    if spec.judgement:
        _check_inputs(node_type, attrs, ctx, refuse)
    if node_id != node_id_of(node_type, attrs):
        raise refuse("node_id is not the one derived from the record's identity, inputs, result and producer")


def _check_forbidden(node_type: str, a: Mapping[str, Any], refuse: Callable[[str], CtpError]) -> None:
    if node_type == "plan":
        bad = [k for k in _walk_keys({k: v for k, v in a.items() if k != "probability_source"}) if "probab" in k.lower()]
        if bad:
            raise refuse(f"a plan has no probability field ({bad[0]!r})")
    elif node_type == "governance":
        bad = [k for k in _walk_keys(dict(a)) if "probab" in k.lower()]
        if bad:
            raise refuse(f"a governance record has no probability and no probability_source ({bad[0]!r})")
    elif node_type == "outcome":
        bad = [k for k in _walk_keys(dict(a)) if "reward" in k.lower()]
        if bad:
            raise refuse(f"an outcome holds no reward ({bad[0]!r}); a reward is computed from outcomes")


def _exists(ctx: Context, node_id: Any, node_type: str) -> bool:
    return isinstance(node_id, str) and ctx.node_type(node_id) == node_type


def _check_fields(node_type: str, a: Mapping[str, Any], ctx: Context, refuse: Callable[[str], CtpError]) -> None:
    def vocab(field: str, allowed: tuple[str, ...]) -> None:
        if a[field] not in allowed:
            raise refuse(f"{field} {a[field]!r} is not one of {allowed}")

    def string(*fields: str) -> None:
        for f in fields:
            if not _is_str(a[f]):
                raise refuse(f"{f} must be a non-empty string")

    if "episode_id" in a and node_type != "episode":
        if not _exists(ctx, a["episode_id"], "episode"):
            raise refuse(f"episode_id {a['episode_id']!r} is not a stored episode")
    if node_type == "episode":
        string("experiment_spec_hash", "item_id", "bundle_id", "condition")
        vocab("split", SPLITS)
        if not _is_int(a["replicate"]) or a["replicate"] < 0 or not _is_int(a["rng_seed"]):
            raise refuse("replicate must be a non-negative integer and rng_seed an integer")
        if a.get("space_id") is not None and not _is_str(a["space_id"]):
            raise refuse("space_id must be a string or absent")
        doc = f"hextdoc:{a['experiment_spec_hash']}"
        docattrs = ctx.attrs(doc)
        if ctx.node_type(doc) != "hext_document" or not docattrs or docattrs.get("hext_kind") != "source":
            raise refuse(f"experiment_spec_hash must be the state_hash of a stored HEXT source document ({doc!r} is not one)")
    elif node_type == "context_observation":
        string("context_ref", "task_ref")
        if a["contains_experience"] is not False:
            raise refuse("contains_experience must be false (a context observation never holds experience)")
        for f in ("knowledge_ids", "constraint_annotation_ids"):
            if not _str_list(a[f]):
                raise refuse(f"{f} must be a list of strings")
    elif node_type == "step_output":
        vocab("step", STEPS)
        string("text_ref", "input_context_ref", "input_context_sha256")
        _check_blob_ref(a, ctx, refuse)
    elif node_type == "applicability":
        string("constraint_id", "applicability_basis")
        vocab("applicability", APPLICABILITIES)
    elif node_type == "carry_evidence":
        vocab("step", STEPS)
        vocab("status", STATUSES)
        vocab("basis", BASES)
        string("property")
        if a["evidence_ref"] is not None and not _is_str(a["evidence_ref"]):
            raise refuse("evidence_ref must be a string or null")
        if a["basis"] == "evidence" and not _is_str(a["evidence_ref"]):
            raise refuse("basis=evidence needs an evidence_ref")
    elif node_type == "cle_result":
        vocab("rule", RULES)
        vocab("mode", MODES)
        vocab("value", VALUES)
        string("history_hash")
        for f in ("reason_codes", "locations"):
            if not _sorted_unique(a[f]):
                raise refuse(f"{f} must be a sorted list of distinct strings")
    elif node_type == "compliance":
        vocab("level", LEVELS)
        vocab("judge", JUDGES)
        vocab("value", VALUES)
        if not _sorted_unique(a["reason_codes"]):
            raise refuse("reason_codes must be a sorted list of distinct strings")
    elif node_type == "plan":
        string("plan_id")
        if not isinstance(a["steps"], list):
            raise refuse("steps must be a list")
        if a["probability_source"] != "unobserved":
            raise refuse("probability_source must be 'unobserved' (the candidate set of a plan is not seen)")
    elif node_type == "governance":
        string("plan_id")
        vocab("decision", DECISIONS)
        if not _sorted_unique(a["reason_codes"]):
            raise refuse("reason_codes must be a sorted list of distinct strings")
        if not _sorted_unique(a["referenced_inputs"]):
            raise refuse("referenced_inputs must be a sorted list of distinct names")
    elif node_type == "outcome":
        vocab("two_axis", TWO_AXIS)
        undetermined = a.get("undetermined_axis")
        if a["two_axis"] == "UNDETERMINED":
            if undetermined not in UNDETERMINED_AXES:
                raise refuse(f"two_axis=UNDETERMINED needs undetermined_axis, one of {UNDETERMINED_AXES}")
        elif undetermined is not None:
            raise refuse("undetermined_axis is only for two_axis=UNDETERMINED")


def _check_blob_ref(a: Mapping[str, Any], ctx: Context, refuse: Callable[[str], CtpError]) -> None:
    """input_context_ref is ``blob:<sha256>``; its hash must be input_context_sha256, and the bytes must hash to it.

    The bytes are read from the data dir's ``blobs/aa/bb/<sha256>`` when the blob is there. HEKB's blob store
    (storage/blobs.py) may keep it in another data dir, so an absent blob is not an error: only the address is checked.
    """
    m = _BLOB_REF.match(a["input_context_ref"])
    if m is None:
        raise refuse("input_context_ref must be 'blob:<sha256 hex>'")
    if not _HEX64.match(a["input_context_sha256"]) or m.group(1) != a["input_context_sha256"]:
        raise refuse("input_context_sha256 is not the sha256 in input_context_ref")
    path = blob_path(ctx.data_dir(), m.group(1))
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != m.group(1):
        raise refuse("the blob's content does not hash to input_context_sha256")


def _check_inputs(node_type: str, a: Mapping[str, Any], ctx: Context, refuse: Callable[[str], CtpError]) -> None:
    if not _sorted_unique(a["input_nodes"]):
        raise refuse("input_nodes must be a sorted list of distinct node_ids")
    if not isinstance(a["input_facts"], dict):
        raise refuse("input_facts must be an object")
    hashes: dict[str, str] = {}
    for n in a["input_nodes"]:
        h = ctx.record_hash(n)
        if h is None:
            raise refuse(f"input node {n!r} is not a stored node")
        hashes[n] = h
    referenced = a["referenced_inputs"] if node_type == "governance" else None
    if node_type == "governance" and a["referenced_inputs"] != _input_names(a["input_nodes"], a["input_facts"]):
        raise refuse("referenced_inputs must name exactly the input nodes and facts the record lists")
    if a["input_hash"] != _sha(input_entries(hashes, a["input_facts"], referenced)):
        raise refuse("input_hash is not the hash of the listed inputs")


# ---- edges ---------------------------------------------------------------------------------------------------
def check_edge(record: Mapping[str, Any], ctx: Context) -> None:
    """Refuse a new CTP edge whose end points are not of the kinds the edge connects."""
    kind = record.get("type")
    if kind not in CTP_EDGE_TYPES:
        return
    src, dst = str(record.get("src")), str(record.get("dst"))

    def refuse(reason: str) -> CtpError:
        return CtpError(f"{kind} edge {src!r} -> {dst!r}: {reason}")

    st, dt = ctx.node_type(src), ctx.node_type(dst)
    sa, da = ctx.attrs(src), ctx.attrs(dst)
    if st is None or dt is None:
        raise refuse("both ends must be stored nodes")
    if kind == "under_manifest":
        if st != "episode" or dt != "hext_document":
            raise refuse("an episode points to the HEXT document of its experiment spec")
        if not da or da.get("hext_kind") != "source" or da.get("state_hash") != (sa or {}).get("experiment_spec_hash"):
            raise refuse("the document must be the HEXT source whose state_hash is the episode's experiment_spec_hash")
    elif kind in ("has_context", "has_step"):
        target = "context_observation" if kind == "has_context" else "step_output"
        if st != "episode" or dt != target:
            raise refuse(f"an episode points to its {target}")
        if (da or {}).get("episode_id") != src:
            raise refuse(f"the {target} belongs to another episode")
    elif kind == "retrieved":
        if st != "step_output" or (sa or {}).get("step") != "f1":
            raise refuse("the f1 step_output points to the knowledge it retrieved")
        if dt != "hext_document" or not da or da.get("hext_kind") != "source":
            raise refuse("the knowledge is a HEXT source document")
    elif kind == "interprets":
        if st not in INTERPRETATIONS:
            raise refuse(f"an interpretation ({INTERPRETATIONS}) is the start")
        if _class(dt) not in ("observation", "evidence"):
            raise refuse("an interpretation interprets an observation or evidence node")
    elif kind == "selects":
        if st != "governance" or dt != "plan" or (sa or {}).get("plan_id") != (da or {}).get("plan_id"):
            raise refuse("a governance record selects the plan it decided on (same plan_id)")
    elif kind == "considered":
        if st not in SELECTIONS:
            raise refuse(f"a selection ({SELECTIONS}) is the start")
        if _class(dt) not in ("observation", "evidence", "interpretation"):
            raise refuse("a selection considers an observation, evidence or interpretation node")
    elif kind == "outcome_of":
        if st != "outcome":
            raise refuse("an outcome is the start")
        if dt == "episode":
            if (sa or {}).get("episode_id") != dst:
                raise refuse("the outcome belongs to another episode")
        elif dt in SELECTIONS:
            if (da or {}).get("episode_id") != (sa or {}).get("episode_id"):
                raise refuse("the selection belongs to another episode")
        else:
            raise refuse("an outcome is of an episode or of a selection")
    elif kind == "audits":
        if st != "compliance" or dt != "compliance" or (sa or {}).get("judge") != "independent" or (da or {}).get("judge") != "verifier":
            raise refuse("an independent compliance judgement audits a verifier's")
        if any((sa or {}).get(f) != (da or {}).get(f) for f in ("episode_id", "level")):
            raise refuse("the two judgements differ in episode or level")


def _class(node_type: str) -> str | None:
    try:
        return d70_class({"node_id": "?", "node_type": node_type, "kind": None})
    except KindMapError:
        return None


# ---- evaluators that disagree -------------------------------------------------------------------------------
@dataclass(frozen=True)
class NondeterminismGroup:
    node_type: str
    identity: tuple[tuple[str, str], ...]  # the identity fields, canonical
    input_hash: str
    producer_fingerprint: str
    node_ids: tuple[str, ...]  # at least two, with different result hashes


def nondeterministic_groups(records: Mapping[str, Mapping[str, Any]]) -> list[NondeterminismGroup]:
    """Judgement/selection nodes with the same identity, input_hash and producer_fingerprint but another result.

    ``records`` maps node_id to record. Reporting only: nothing here refuses a write.
    """
    groups: dict[tuple[Any, ...], dict[str, str]] = {}
    for node_id, rec in records.items():
        node_type = rec.get("node_type")
        spec = SPECS.get(str(node_type))
        if spec is None or not spec.judgement:
            continue
        a = rec["attrs"]
        ident = tuple((f, canonicalize(a[f])) for f in spec.ident)
        key = (node_type, ident, a["input_hash"], producer_fingerprint(a["producer"]))
        groups.setdefault(key, {})[node_id] = result_hash(spec, a)
    out = []
    for (node_type, ident, in_hash, fp), members in sorted(groups.items(), key=lambda kv: canonicalize(list(kv[0]))):
        if len(set(members.values())) > 1:
            out.append(NondeterminismGroup(str(node_type), ident, in_hash, fp, tuple(sorted(members))))
    return out
