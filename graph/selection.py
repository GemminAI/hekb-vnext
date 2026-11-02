"""CTP selections that draw: ``assignment`` and ``experience`` (CTP data model v0.3 sections 6.1-6.7, v0.3.1, v0.3.5;
the open points were decided by Tomo and Claude on 2026-11-03).

Pure functions, no graph access: ``graph.ctp`` calls ``check_assignment`` / ``check_experience`` when a record is
written and builds ``input_hash`` from ``definition``.

Recomputation at write time (v0.3 section 6.5). The candidates are read from their blob (a JSON array of strings in
byte order), the choice is drawn again with ``graph.rng`` (``sha256_rejection_v1``) and every value of the record
must equal the recomputed one:

* ``assignment``: one condition out of ``m`` candidates, ``assignment_probability`` = 1/m
  (``assignment_algorithm`` = ``{uniform_within_stratum, 1}``; the stratum is a label and never enters the draw).
* ``experience``: ``k`` of ``n`` candidates drawn one after another without replacement, the order drawn is the order
  presented (``selection_algorithm`` = ``{uniform_ordered_k_subset_without_replacement, 1}``, ``order_considered``
  is always true); ``conditional_selection_probabilities[i]`` = 1/(n-i), ``slate_probability.set`` = 1/C(n,k) and
  ``.ordered`` = (n-k)!/n!. ``arm_id`` = ``"arm:" + sha256(JCS({selected_ids, presentation_mode,
  presentation_format}))``: the episode, the policy, the candidate set and the condition are not in it.
* a probability is an irreducible fraction ``{num, den}`` whose two members are decimal strings (no sign, no leading
  zero, ``den`` is not "0"): an integer member is refused, so a denominator such as n!/(n-k)! never meets the JCS
  limit of 2**53 - 1. Values are compared and recomputed as Python integers.
* the policy fingerprint is ``sha256(JCS({...}))`` over the algorithm, the settings the draw reads and the policy's
  free ``..._policy_params`` object, recomputed at write time; ``probability_source`` is ``selector``, or
  ``single_candidate`` when the blob really holds one candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

from graph import rng
from graph.blobs import blob_path
from graph.jcs import canonical_bytes
from graph.model import GraphError

ASSIGNMENT_ALGORITHM: Final = {"name": "uniform_within_stratum", "version": "1"}
SELECTION_ALGORITHM: Final = {"name": "uniform_ordered_k_subset_without_replacement", "version": "1"}
ORDERING_RULE: Final = "node_id_bytewise_ascending"
PRESENTATION_MODES: Final = ("contrast", "compliant_only", "violation_only", "unrelated")
RETRIEVAL_SPACES: Final = ("R-text", "R-struct", "R-traj")
PROBABILITY_SOURCES: Final = ("selector", "single_candidate")
PREFILL_TRAJECTORY_NOT_AVAILABLE: Final = "PREFILL_TRAJECTORY_NOT_AVAILABLE"

ASSIGNMENT_DEFINITION: Final = (
    "assignment_candidate_set_hash", "assignment_algorithm", "assignment_stratum", "assignment_rng",
    "assignment_policy_id", "assignment_policy_version", "assignment_policy_params", "assignment_policy_fingerprint",
)
EXPERIENCE_DEFINITION: Final = (
    "candidate_set_hash", "selection_algorithm", "k", "replacement", "order_considered", "ordering_rule", "rng",
    "selection_policy_id", "selection_policy_version", "selection_policy_params", "selection_policy_fingerprint",
    "retrieval_space",
)

_BLOB_REF = re.compile(r"^blob:([0-9a-f]{64})$")
_DECIMAL = re.compile(r"^(0|[1-9][0-9]*)$")


class SelectionError(GraphError):
    """A selection record that does not match what its definition draws."""


def definition(node_type: str, attrs: Mapping[str, Any]) -> dict[str, Any]:
    """The part of the record the recomputation reads: it goes into ``input_hash`` (v0.3.1 section 2)."""
    fields = ASSIGNMENT_DEFINITION if node_type == "assignment" else EXPERIENCE_DEFINITION
    return {f: attrs.get(f) for f in fields}


# ---- candidates --------------------------------------------------------------------------------------------
def byte_sorted(items: Sequence[str]) -> list[str]:
    return sorted(items, key=lambda s: s.encode("utf-8"))


def candidates_blob(items: Sequence[str]) -> bytes:
    """The blob of a candidate list: a JSON array of the strings in byte order (``candidates_ref``)."""
    return canonical_bytes(byte_sorted(items))


def read_candidates(root: Path | None, ref: Any, set_hash: Any, what: str) -> list[str]:
    m = _BLOB_REF.match(ref) if isinstance(ref, str) else None
    if m is None:
        raise SelectionError(f"{what}: the candidates reference must be 'blob:<sha256 hex>'")
    if set_hash != m.group(1):
        raise SelectionError(f"{what}: the candidate set hash is not the sha256 in the candidates reference")
    if root is None:
        raise SelectionError(f"{what}: no blob location was given to the store (GraphStore(blob_dir=...)), so the candidates cannot be read")
    path = blob_path(root, m.group(1))
    if not path.exists():
        raise SelectionError(f"{what}: the candidates blob {m.group(1)} is not in the blob location")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != m.group(1):
        raise SelectionError(f"{what}: the candidates blob does not hash to its name")
    try:
        items = json.loads(data)
    except ValueError as exc:
        raise SelectionError(f"{what}: the candidates blob is not JSON") from exc
    if not isinstance(items, list) or not items or not all(isinstance(x, str) and x for x in items):
        raise SelectionError(f"{what}: the candidates blob must be a non-empty JSON array of strings")
    if items != byte_sorted(set(items)):
        raise SelectionError(f"{what}: the candidates must be distinct and in byte order")
    return list(items)


# ---- fractions --------------------------------------------------------------------------------------------
def fraction(value: Fraction) -> dict[str, str]:
    return {"num": str(value.numerator), "den": str(value.denominator)}


def check_fraction(value: Any, what: str) -> None:
    if not isinstance(value, dict) or set(value) != {"num", "den"}:
        raise SelectionError(f"{what} must be a fraction {{num, den}}")
    if not all(isinstance(v, str) and _DECIMAL.match(v) for v in value.values()):
        raise SelectionError(f"{what}: num and den must be decimal strings (digits only, no sign, no leading zero)")
    num, den = int(value["num"]), int(value["den"])
    if den == 0:
        raise SelectionError(f"{what}: den must not be 0")
    if math.gcd(num, den) != 1:
        raise SelectionError(f"{what} is not an irreducible fraction")


def policy_fingerprint(material: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(material)).hexdigest()


def selection_fingerprint(a: Mapping[str, Any]) -> str:
    return policy_fingerprint({f: a.get(f) for f in ("selection_algorithm", "k", "replacement", "order_considered", "ordering_rule", "selection_policy_params")})


def assignment_fingerprint(a: Mapping[str, Any]) -> str:
    return policy_fingerprint({f: a.get(f) for f in ("assignment_algorithm", "assignment_stratum", "assignment_policy_params")})


def _check_params(a: Mapping[str, Any], field: str) -> None:
    if not isinstance(a[field], dict):
        raise SelectionError(f"{field} must be an object")


def _check_rng(value: Any, what: str) -> str:
    if not isinstance(value, dict) or set(value) != {"algorithm", "seed"}:
        raise SelectionError(f"{what} must be {{algorithm, seed}}")
    if value["algorithm"] != rng.ALGORITHM:
        raise SelectionError(f"{what}.algorithm must be {rng.ALGORITHM!r}")
    try:
        return rng.check_seed(value["seed"])
    except rng.RngError as exc:
        raise SelectionError(f"{what}.seed: {exc}") from exc


def _check_source(a: Mapping[str, Any], n: int) -> None:
    if a["probability_source"] not in PROBABILITY_SOURCES:
        raise SelectionError(f"probability_source must be one of {PROBABILITY_SOURCES}")
    if a["probability_source"] == "single_candidate" and n != 1:
        raise SelectionError(f"probability_source single_candidate is only for a candidate set of one (the blob holds {n})")


def _strings(a: Mapping[str, Any], *fields: str) -> None:
    for f in fields:
        if not isinstance(a[f], str) or not a[f]:
            raise SelectionError(f"{f} must be a non-empty string")


# ---- assignment -----------------------------------------------------------------------------------------------
def assignment_fields(
    experiment_spec_hash: str, item_id: str, replicate: int, candidates: Sequence[str], candidates_sha: str, *,
    seed: str, stratum: str, policy_id: str, policy_version: str, policy_params: Mapping[str, Any] | None = None,
    probability_source: str = "selector",
) -> dict[str, Any]:
    """The fields of an assignment, drawn from ``candidates`` (whose blob ``candidates_blob`` is stored under ``candidates_sha``)."""
    ordered = byte_sorted(candidates)
    fields = {
        "experiment_spec_hash": experiment_spec_hash, "item_id": item_id, "replicate": replicate,
        "assignment_candidates_ref": f"blob:{candidates_sha}", "assignment_candidate_set_hash": candidates_sha,
        "assignment_algorithm": dict(ASSIGNMENT_ALGORITHM), "assignment_stratum": stratum,
        "assignment_rng": {"algorithm": rng.ALGORITHM, "seed": seed},
        "assignment_policy_id": policy_id, "assignment_policy_version": policy_version,
        "assignment_policy_params": dict(policy_params or {}),
        "condition": rng.choose_one(ordered, seed), "assignment_probability": fraction(Fraction(1, len(ordered))),
        "probability_source": probability_source,
    }
    return {**fields, "assignment_policy_fingerprint": assignment_fingerprint(fields)}


def check_assignment(a: Mapping[str, Any], root: Path | None) -> None:
    _strings(a, "experiment_spec_hash", "item_id", "assignment_stratum", "assignment_policy_id", "assignment_policy_version", "condition")
    if not isinstance(a["replicate"], int) or isinstance(a["replicate"], bool) or a["replicate"] < 0:
        raise SelectionError("replicate must be a non-negative integer")
    if a["assignment_algorithm"] != ASSIGNMENT_ALGORITHM:
        raise SelectionError(f"assignment_algorithm must be {ASSIGNMENT_ALGORITHM}")
    seed = _check_rng(a["assignment_rng"], "assignment_rng")
    check_fraction(a["assignment_probability"], "assignment_probability")
    _check_params(a, "assignment_policy_params")
    if a["assignment_policy_fingerprint"] != assignment_fingerprint(a):
        raise SelectionError("recomputation: assignment_policy_fingerprint is not sha256(JCS({assignment_algorithm, assignment_stratum, assignment_policy_params}))")
    candidates = read_candidates(root, a["assignment_candidates_ref"], a["assignment_candidate_set_hash"], "assignment")
    _check_source(a, len(candidates))
    expected = rng.choose_one(candidates, seed)
    if a["condition"] != expected:
        raise SelectionError(f"recomputation: the condition drawn from the candidates is {expected!r}, the record says {a['condition']!r}")
    if a["assignment_probability"] != fraction(Fraction(1, len(candidates))):
        raise SelectionError(f"recomputation: assignment_probability must be 1/{len(candidates)}")


# ---- experience -------------------------------------------------------------------------------------------------
def arm_id(selected_ids: Sequence[str], presentation_mode: str, presentation_format: Mapping[str, str]) -> str:
    material = {"selected_ids": list(selected_ids), "presentation_mode": presentation_mode, "presentation_format": dict(presentation_format)}
    return "arm:" + hashlib.sha256(canonical_bytes(material)).hexdigest()


def expected_probabilities(n: int, k: int) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    steps = [fraction(Fraction(1, n - i)) for i in range(k)]
    ordered = Fraction(math.factorial(n - k), math.factorial(n))
    return steps, {"set": fraction(Fraction(1, math.comb(n, k))), "ordered": fraction(ordered)}


def experience_fields(
    episode_id: str, candidates: Sequence[str], candidates_sha: str, *, k: int, seed: str,
    policy_id: str, policy_version: str, policy_params: Mapping[str, Any] | None = None,
    presentation_mode: str, presentation_format: Mapping[str, str],
    retrieval_space: str = "R-text", probability_source: str = "selector",
) -> dict[str, Any]:
    """The fields of an experience, drawn from ``candidates`` (blob stored under ``candidates_sha``)."""
    ordered = byte_sorted(candidates)
    selected = rng.choose_ordered(ordered, k, seed)
    steps, slate = expected_probabilities(len(ordered), k)
    fields = {
        "episode_id": episode_id, "candidates_ref": f"blob:{candidates_sha}", "candidate_set_hash": candidates_sha,
        "selection_algorithm": dict(SELECTION_ALGORITHM), "k": k, "replacement": False, "order_considered": True,
        "ordering_rule": ORDERING_RULE, "rng": {"algorithm": rng.ALGORITHM, "seed": seed},
        "selection_policy_id": policy_id, "selection_policy_version": policy_version,
        "selection_policy_params": dict(policy_params or {}),
        "selected_ids": selected, "conditional_selection_probabilities": steps, "slate_probability": slate,
        "arm_id": arm_id(selected, presentation_mode, presentation_format),
        "presentation_mode": presentation_mode, "presentation_format": dict(presentation_format),
        "retrieval_space": retrieval_space, "probability_source": probability_source,
    }
    return {**fields, "selection_policy_fingerprint": selection_fingerprint(fields)}


def check_experience(a: Mapping[str, Any], root: Path | None) -> None:
    if a["retrieval_space"] not in RETRIEVAL_SPACES:
        raise SelectionError(f"retrieval_space must be one of {RETRIEVAL_SPACES}")
    if a["retrieval_space"] == "R-traj":
        raise SelectionError(f"{PREFILL_TRAJECTORY_NOT_AVAILABLE}: R-traj needs the pre-decision read trajectory (Stage F), which does not exist yet")
    _strings(a, "selection_policy_id", "selection_policy_version")
    _check_params(a, "selection_policy_params")
    if a["selection_algorithm"] != SELECTION_ALGORITHM:
        raise SelectionError(f"selection_algorithm must be {SELECTION_ALGORITHM}")
    if a["replacement"] is not False:
        raise SelectionError("replacement must be false")
    if a["order_considered"] is not True:
        raise SelectionError("order_considered must be true (CTP002 draws an ordered slate)")
    if a["ordering_rule"] != ORDERING_RULE:
        raise SelectionError(f"ordering_rule must be {ORDERING_RULE!r}")
    seed = _check_rng(a["rng"], "rng")
    if a["selection_policy_fingerprint"] != selection_fingerprint(a):
        raise SelectionError("recomputation: selection_policy_fingerprint is not sha256(JCS({selection_algorithm, k, replacement, order_considered, ordering_rule, selection_policy_params}))")
    if a["presentation_mode"] not in PRESENTATION_MODES:
        raise SelectionError(f"presentation_mode must be one of {PRESENTATION_MODES}")
    fmt = a["presentation_format"]
    if not isinstance(fmt, dict) or set(fmt) != {"template_id", "version"} or not all(isinstance(v, str) and v for v in fmt.values()):
        raise SelectionError("presentation_format must be {template_id, version}, both non-empty strings")
    candidates = read_candidates(root, a["candidates_ref"], a["candidate_set_hash"], "experience")
    n, k = len(candidates), a["k"]
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise SelectionError(f"k must be an integer from 1 to the number of candidates ({n})")
    _check_source(a, n)
    if not isinstance(a["conditional_selection_probabilities"], list):
        raise SelectionError("conditional_selection_probabilities must be a list of fractions")
    for p in a["conditional_selection_probabilities"]:
        check_fraction(p, "conditional_selection_probabilities entry")
    slate = a["slate_probability"]
    if not isinstance(slate, dict) or set(slate) != {"set", "ordered"}:
        raise SelectionError("slate_probability must be {set, ordered}")
    check_fraction(slate["set"], "slate_probability.set")
    check_fraction(slate["ordered"], "slate_probability.ordered")
    expected = rng.choose_ordered(candidates, k, seed)
    if a["selected_ids"] != expected:
        raise SelectionError(f"recomputation: the {k} drawn from the candidates are {expected}, the record says {a['selected_ids']!r}")
    steps, expected_slate = expected_probabilities(n, k)
    if a["conditional_selection_probabilities"] != steps:
        raise SelectionError(f"recomputation: conditional_selection_probabilities must be 1/(n-i) for n={n}, k={k}")
    if slate != expected_slate:
        raise SelectionError(f"recomputation: slate_probability must be set=1/C({n},{k}) and ordered=({n}-{k})!/{n}!")
    if a["arm_id"] != arm_id(a["selected_ids"], a["presentation_mode"], fmt):
        raise SelectionError("arm_id is not the one derived from selected_ids, presentation_mode and presentation_format")
