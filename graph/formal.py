"""Formal Store (U6T-2a): the declaration, and the checks HEKB makes on a GateVerdict (design U6T-2 v0.1, sections 3-5).

HEKB does not judge what a verdict means. It checks that the verdict has the agreed form (the vendored sensos schema),
is undamaged (``verdict_sha256``), is about THIS envelope (HEKB's own JCS hash), was made with the rules and the lookup
the store itself knows, and is of a kind the declaration allows. Everything here is pure: it reads, never writes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from vendor.sensos_gate_verdict.validate import verdict_errors
from vendor.sensos_ic00.errors import ContractError
from vendor.sensos_ic00.jcs import jcs

DECLARATION_VERSION: Final = "formal-store/1"
KNOWN_VERDICT_VERSIONS: Final = ("gate-verdict/1",)
DECLARATION_KEYS: Final = (
    "declaration",
    "accepted_versions",
    "accepted_payload_schemas",
    "gate_verdict_versions",
    "allow_unverified",
    "allow_waived",
    "unverifiable_catalog",
    "missing_parents",
    "created_by",
)
#: What U6 stores in a Formal Store (D-177 (3)): nvs.hidden-trajectory and the msr.* payloads of IC-02 0.10.0 (sensos gate.PAYLOAD_SCHEMAS).
DEFAULT_PAYLOAD_SCHEMAS: Final = (
    {"id": "nvs.hidden-trajectory", "version": "0.1.0"},
    {"id": "msr.geometry-measurement", "version": "0.10.0"},
    {"id": "msr.trajectory-window", "version": "0.10.0"},
    {"id": "msr.calibration", "version": "0.8.0"},
)
#: The two "not checked" rows that belong to a window payload itself (sensos UNVERIFIABLE catalogue). Anything else is a choice.
DEFAULT_ALLOW_UNVERIFIED: Final = (
    {"check": "window_rows", "reason": "raw_not_in_artifact"},
    {"check": "window_space_derivation", "reason": "rule_not_derivable"},
)
_HEX64 = frozenset("0123456789abcdef")


class Refusal(Exception):
    """``code`` is a refusal code of the design (``gate_verdict_malformed`` ...); the store turns it into an ``IC04Error``."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code, self.detail = code, detail


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


# ---------------------------------------------------------------- declaration
def catalog_sha256(rows: Sequence[Sequence[str]]) -> str:
    """The hash sensos gives its UNVERIFIABLE catalogue: SHA-256 of the JCS of the rows, sorted, as {payload_schema, check, reason}."""
    objects = [
        {"payload_schema": s, "check": c, "reason": r}
        for s, c, r in sorted(map(list, rows))
    ]
    return sha256_hex(jcs(objects))


def formal_declaration(
    *,
    catalog_rows: Sequence[Sequence[str]],
    catalog_version: str,
    created_by: str,
    accepted_payload_schemas: Sequence[Mapping[str, str]] | None = None,
    allow_unverified: Sequence[Mapping[str, str]] | None = None,
    accepted_versions: Sequence[str] = ("1.1.0",),
) -> dict[str, Any]:
    """A declaration with the defaults of the design; ``catalog_rows`` are the rows of sensos's UNVERIFIABLE catalogue
    ([payload_schema, check, reason]), kept in the declaration so that HEKB can check the allow-list against them."""
    rows = sorted(map(list, catalog_rows))
    return {
        "declaration": DECLARATION_VERSION,
        "accepted_versions": list(accepted_versions),
        "accepted_payload_schemas": [
            dict(p) for p in (accepted_payload_schemas or DEFAULT_PAYLOAD_SCHEMAS)
        ],
        "gate_verdict_versions": list(KNOWN_VERDICT_VERSIONS),
        "allow_unverified": [
            dict(a)
            for a in (
                DEFAULT_ALLOW_UNVERIFIED
                if allow_unverified is None
                else allow_unverified
            )
        ],
        "allow_waived": False,
        "unverifiable_catalog": {
            "version": catalog_version,
            "sha256": catalog_sha256(rows),
            "rows": rows,
        },
        "missing_parents": "reject",
        "created_by": created_by,
    }


def _nonempty_str(v: Any) -> bool:
    return isinstance(v, str) and v != ""


def _row_match(
    allowed_check: str, allowed_reason: str, check: str, reason: str
) -> bool:
    """An allow-list row (or a catalogue row) matches a check; a check ending in ``.<field>`` stands for any field name after its prefix."""
    if reason != allowed_reason:
        return False
    if check == allowed_check:
        return True
    if allowed_check.endswith(".<field>"):
        prefix = allowed_check[: -len("<field>")]
        return check.startswith(prefix) and len(check) > len(prefix)
    return False


def declaration_errors(d: Any) -> list[str]:
    """The reasons a declaration is not of the form ``formal-store/1`` (empty = it is). Used on create and on open (③)."""
    if not isinstance(d, dict):
        return ["the declaration is not an object"]
    errs: list[str] = []
    if set(d) != set(DECLARATION_KEYS):
        errs.append(f"keys must be exactly {list(DECLARATION_KEYS)}; got {sorted(d)}")
        return errs
    if d["declaration"] != DECLARATION_VERSION:
        errs.append(f"declaration must be {DECLARATION_VERSION!r}")
    av = d["accepted_versions"]
    if not (
        isinstance(av, list)
        and av
        and all(_nonempty_str(v) for v in av)
        and len(set(av)) == len(av)
    ):
        errs.append("accepted_versions must be a non-empty list of distinct strings")
    ps = d["accepted_payload_schemas"]
    if not (
        isinstance(ps, list)
        and ps
        and all(
            isinstance(p, dict)
            and set(p) == {"id", "version"}
            and _nonempty_str(p["id"])
            and _nonempty_str(p["version"])
            for p in ps
        )
    ):
        errs.append(
            "accepted_payload_schemas must be a non-empty list of {id, version}"
        )
    elif len({(p["id"], p["version"]) for p in ps}) != len(ps):
        errs.append("accepted_payload_schemas has a duplicate")
    gv = d["gate_verdict_versions"]
    if not (
        isinstance(gv, list)
        and gv
        and all(v in KNOWN_VERDICT_VERSIONS for v in gv)
        and len(set(gv)) == len(gv)
    ):
        errs.append(
            f"gate_verdict_versions must be a non-empty subset of {list(KNOWN_VERDICT_VERSIONS)}"
        )
    if d["allow_waived"] is not False:
        errs.append("allow_waived must be false (waive is not available before U6T-6)")
    if d["missing_parents"] != "reject":
        errs.append("missing_parents must be 'reject'")
    if not _nonempty_str(d["created_by"]):
        errs.append("created_by must be a non-empty string")
    cat = d["unverifiable_catalog"]
    rows: list[list[str]] = []
    if not (
        isinstance(cat, dict)
        and set(cat) == {"version", "sha256", "rows"}
        and _nonempty_str(cat["version"])
        and is_hex64(cat["sha256"])
        and isinstance(cat["rows"], list)
        and all(
            isinstance(r, list) and len(r) == 3 and all(_nonempty_str(x) for x in r)
            for r in cat["rows"]
        )
    ):
        errs.append(
            "unverifiable_catalog must be {version, sha256 (64 hex), rows [[payload_schema, check, reason], ...]}"
        )
    else:
        rows = cat["rows"]
        if cat["sha256"] != catalog_sha256(rows):
            errs.append("unverifiable_catalog.sha256 is not the hash of its rows")
    au = d["allow_unverified"]
    if not (
        isinstance(au, list)
        and all(
            isinstance(a, dict)
            and set(a) == {"check", "reason"}
            and _nonempty_str(a["check"])
            and _nonempty_str(a["reason"])
            for a in au
        )
    ):
        errs.append("allow_unverified must be a list of {check, reason}")
    elif rows:
        for a in au:
            if not any(a["check"] == r[1] and a["reason"] == r[2] for r in rows):
                errs.append(
                    f"allow_unverified {a['check']}/{a['reason']} is not a row of the catalogue"
                )
        if len({(a["check"], a["reason"]) for a in au}) != len(au):
            errs.append("allow_unverified has a duplicate")
    return errs


def unverifiable_allowed(decl: Mapping[str, Any], check: str, reason: str) -> bool:
    return any(
        _row_match(a["check"], a["reason"], check, reason)
        for a in decl["allow_unverified"]
    )


def payload_accepted(decl: Mapping[str, Any], envelope: Mapping[str, Any]) -> bool:
    ps = envelope.get("payload_schema")
    return isinstance(ps, dict) and any(
        ps.get("id") == p["id"] and ps.get("version") == p["version"]
        for p in decl["accepted_payload_schemas"]
    )


# ---------------------------------------------------------------- the GateVerdict
def _body_sha(verdict: Mapping[str, Any]) -> str:
    return sha256_hex(jcs({k: v for k, v in verdict.items() if k != "verdict_sha256"}))


def envelope_sha256(envelope: Mapping[str, Any]) -> str:
    """HEKB's own hash of the envelope: SHA-256 of its JCS (the vendored RFC 8785)."""
    try:
        return sha256_hex(jcs(envelope))
    except ContractError as ex:
        raise Refusal(
            "gate_subject_mismatch", f"the envelope cannot be canonicalised: {ex.code}"
        ) from ex


def form_problem(
    verdict: Any, versions: Sequence[str] = KNOWN_VERDICT_VERSIONS
) -> str | None:
    """C-3 without the catalogue: the version, the vendored schema, ``verdict_sha256``, ``lookup_sha256``. ``None`` = fine."""
    if not isinstance(verdict, dict):
        return "the verdict is not an object"
    if verdict.get("verdict_version") not in versions:
        return f"verdict_version {verdict.get('verdict_version')!r} is not one of {list(versions)}"
    errs = verdict_errors(verdict)
    if errs:
        return "; ".join(errs[:3])
    try:
        if _body_sha(verdict) != verdict["verdict_sha256"]:
            return "verdict_sha256 is not the hash of the rest"
        if (
            sha256_hex(jcs(verdict["inputs"]["lookup"]))
            != verdict["inputs"]["lookup_sha256"]
        ):
            return "inputs.lookup_sha256 is not the hash of inputs.lookup"
    except ContractError as ex:
        return f"the verdict cannot be canonicalised: {ex.code}"
    return None


def binding_problem(
    envelope: Mapping[str, Any], verdict: Mapping[str, Any]
) -> str | None:
    """The part of C-4 / C-5 that needs no index: ``subject`` and ``rules`` against the envelope. ``None`` = fine."""
    subject = verdict["subject"]
    if subject["artifact_id"] != envelope.get("artifact_id") or subject[
        "content_id"
    ] != envelope.get("content_id"):
        return "subject.artifact_id / content_id is not the envelope's"
    try:
        if subject["envelope_sha256"] != envelope_sha256(envelope):
            return "subject.envelope_sha256 is not the hash of this envelope"
    except Refusal as r:
        return r.detail
    return None


def rules_problem(
    envelope: Mapping[str, Any], verdict: Mapping[str, Any]
) -> str | None:
    rules = verdict["rules"]
    contract = envelope.get("contract") or {}
    ps = envelope.get("payload_schema") or {}
    if rules["ic00_contract_version"] != contract.get("version"):
        return "rules.ic00_contract_version is not the envelope's contract.version"
    if rules["payload_schema"] != {"id": ps.get("id"), "version": ps.get("version")}:
        return "rules.payload_schema is not the envelope's payload_schema"
    return None


def required_ids(envelope: Mapping[str, Any]) -> tuple[list[str], str | None]:
    """The parents' artifact_ids (in order) and the calibration artifact_id, which the Gate's lookup must cover."""
    parents = [
        a for p in envelope.get("parents") or [] for a in p.get("artifact_ids") or []
    ]
    cal = envelope.get("calibration_ref")
    return parents, (cal.get("artifact_id") if isinstance(cal, dict) else None)


def check_verdict(
    envelope: Mapping[str, Any],
    verdict: Any,
    decl: Mapping[str, Any],
    known: Callable[[str], Mapping[str, Any] | None],
) -> None:
    """C-3 .. C-7. ``known(artifact_id)`` returns ``{content_id, seq, observation_id}`` of a stored envelope, or ``None``.
    Raises ``Refusal`` with the code of the first check that fails."""
    # waive is not available: refuse it by its own name before the form check would call it malformed (the schema has no WAIVED)
    waived = isinstance(verdict, dict) and (
        verdict.get("result") == "WAIVED" or verdict.get("waived")
    )
    if waived and not decl["allow_waived"]:
        raise Refusal(
            "gate_waived_not_allowed", "WAIVED / waived is not allowed by this store"
        )
    problem = form_problem(verdict, decl["gate_verdict_versions"])
    if problem:
        raise Refusal("gate_verdict_malformed", problem)
    if (
        verdict["gate"]["unverifiable_catalog"]["sha256"]
        != decl["unverifiable_catalog"]["sha256"]
    ):
        raise Refusal(
            "gate_catalog_mismatch",
            "the verdict was made against another UNVERIFIABLE catalogue",
        )
    problem = binding_problem(envelope, verdict)
    if problem:
        raise Refusal("gate_subject_mismatch", problem)
    problem = rules_problem(envelope, verdict)
    if problem:
        raise Refusal("gate_rule_mismatch", problem)
    parents, calibration = required_ids(envelope)
    lookup = verdict["inputs"]["lookup"]
    for aid in parents + ([calibration] if calibration else []):
        if aid not in lookup and not (aid == calibration and known(aid) is None):
            raise Refusal("gate_lookup_mismatch", f"{aid} is not in inputs.lookup")
    for aid, entry in lookup.items():
        held = known(aid)
        if held is None:
            if aid == calibration:
                continue  # C-6 names it
            raise Refusal(
                "gate_lookup_mismatch", f"{aid} in inputs.lookup is not in the store"
            )
        for key in ("content_id", "seq", "observation_id"):
            if entry.get(key) != held[key] or type(entry.get(key)) is not type(
                held[key]
            ):
                raise Refusal(
                    "gate_lookup_mismatch", f"{aid}: {key} is not the store's value"
                )
    if calibration and known(calibration) is None:
        raise Refusal("calibration_unknown", calibration)
    result = verdict["result"]
    if result == "FAIL":
        raise Refusal("gate_rejected", f"result FAIL: {verdict['codes']}")
    if result == "PASS_UNVERIFIED":
        for u in verdict["unverifiable"]:
            if not unverifiable_allowed(decl, u["check"], u["reason"]):
                raise Refusal(
                    "gate_unverified_not_allowed", f"{u['check']}/{u['reason']}"
                )


# ---------------------------------------------------------------- the state of one record (design v0.3 5.2)
def self_state(envelope: Mapping[str, Any], gate: Any) -> str:
    """``NONE`` / ``LEGACY`` / ``INCONSISTENT`` / ``FAIL_RECORDED`` / ``WAIVED`` / ``PASS_UNVERIFIED`` / ``PASS``, from the record alone."""
    if gate is None:
        return "NONE"
    if isinstance(gate, dict) and "verdict_version" not in gate:
        return "LEGACY"
    if (
        not isinstance(gate, dict)
        or form_problem(gate) is not None
        or binding_problem(envelope, gate)
        or rules_problem(envelope, gate)
    ):
        return "INCONSISTENT"
    return {
        "FAIL": "FAIL_RECORDED",
        "WAIVED": "WAIVED",
        "PASS_UNVERIFIED": "PASS_UNVERIFIED",
        "PASS": "PASS",
    }[gate["result"]]
