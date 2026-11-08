"""Section 2.1 / 2.2: content_id, artifact_id, session_id, observation_id.

``content_id`` preimage (JCS object; key names fixed here because the design
lists the fields but not their JSON names)::

    contract, kind, payload_schema, payload,
    producer {name, version},            # producer.build is NOT here
    policy_refs, rule_refs,              # sorted as sets
    calibration_content_id,              # calibration_ref.content_id or null
    space_id, event_span,
    parent_contents [{role, ordered, content_ids}]

``artifact_id`` preimage::

    content_id, session_id, observation_id, producer_build,
    parents [{role, ordered, artifact_ids}]

Parents: roles by UTF-8 byte order of the role name; ``ordered: true`` keeps
the given order; ``ordered: false`` sorts by id bytes (content ids for
content_id, artifact ids for artifact_id), duplicate artifact ids refused.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

from .errors import (
    E_ID_INPUT,
    E_LOOKUP,
    E_PARENTS,
    ContractError,
)
from .jcs import MAX_SAFE, jcs

Lookup = Mapping[str, Mapping[str, Any]]
_SES = re.compile(r"^ses:[0-9a-f]{64}$")


def _b(text: str) -> bytes:
    return text.encode("utf-8")


def _sha(obj: Any) -> str:
    return hashlib.sha256(jcs(obj)).hexdigest()


def sorted_set(items: list[dict[str, Any]], fields: tuple[str, ...], what: str) -> list[dict[str, Any]]:
    """Sort a set-like array by the UTF-8 bytes of ``fields`` in order; refuse duplicates."""
    ordered = sorted(items, key=lambda item: tuple(_b(item[f]) for f in fields))
    keys = [tuple(item[f] for f in fields) for item in ordered]
    if len(set(keys)) != len(keys):
        raise ContractError(E_PARENTS, f"duplicate entry in {what}")
    return ordered


def _roles(parents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    names = [p["role"] for p in parents]
    if len(set(names)) != len(names):
        raise ContractError(E_PARENTS, "duplicate role in parents")
    out = []
    for parent in sorted(parents, key=lambda p: _b(p["role"])):
        ids = list(parent["artifact_ids"])
        if not parent["ordered"]:
            ids.sort(key=_b)
            if len(set(ids)) != len(ids):
                raise ContractError(E_PARENTS, f"duplicate artifact id in role {parent['role']!r}")
        out.append({"role": parent["role"], "ordered": parent["ordered"], "artifact_ids": ids})
    return out


def _content_of(artifact_id: str, lookup: Lookup | None) -> str:
    if lookup is None or artifact_id not in lookup or "content_id" not in lookup[artifact_id]:
        raise ContractError(E_LOOKUP, f"parent {artifact_id} not in lookup")
    return str(lookup[artifact_id]["content_id"])


def content_id(envelope: Mapping[str, Any], lookup: Lookup | None = None) -> str:
    parent_contents = []
    for role in _roles(list(envelope["parents"])):
        cids = [_content_of(a, lookup) for a in role["artifact_ids"]]
        if not role["ordered"]:
            cids.sort(key=_b)
        parent_contents.append({"role": role["role"], "ordered": role["ordered"], "content_ids": cids})
    calibration = envelope["calibration_ref"]
    preimage = {
        "contract": envelope["contract"],
        "kind": envelope["kind"],
        "payload_schema": envelope["payload_schema"],
        "payload": envelope["payload"],
        "producer": {
            "name": envelope["producer"]["name"],
            "version": envelope["producer"]["version"],
        },
        "policy_refs": sorted_set(list(envelope["policy_refs"]), ("name", "sha256"), "policy_refs"),
        "rule_refs": sorted_set(
            list(envelope["rule_refs"]), ("name", "version", "sha256"), "rule_refs"
        ),
        "calibration_content_id": None if calibration is None else calibration["content_id"],
        "space_id": envelope["space_id"],
        "event_span": envelope["event_span"],
        "parent_contents": parent_contents,
    }
    return "cnt:" + _sha(preimage)


def artifact_id(envelope: Mapping[str, Any], lookup: Lookup | None = None) -> str:
    preimage = {
        "content_id": content_id(envelope, lookup),
        "session_id": envelope["session_id"],
        "observation_id": envelope["observation_id"],
        "producer_build": envelope["producer"]["build"],
        "parents": _roles(list(envelope["parents"])),
    }
    return "art:" + _sha(preimage)


def _require_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= MAX_SAFE:
        raise ContractError(E_ID_INPUT, f"{name} must be an int in [{minimum}, 2^53-1]")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(E_ID_INPUT, f"{name} must be a non-empty string")
    return value


def session_id_experiment(
    experiment_spec_hash: str, item_id: str, condition: str, replicate: int, rng_seed: int
) -> str:
    """Experiment form: re-running the same experiment yields the same id."""
    return "ses:" + _sha(
        {
            "experiment_spec_hash": _require_str(experiment_spec_hash, "experiment_spec_hash"),
            "item_id": _require_str(item_id, "item_id"),
            "condition": _require_str(condition, "condition"),
            "replicate": _require_int(replicate, "replicate", 0),
            "rng_seed": _require_int(rng_seed, "rng_seed", -MAX_SAFE),
        }
    )


def session_id_operational(random_bytes: bytes | None = None) -> str:
    """Operational form: 256 random bits, recorded by nvs-runtime (no replay equality)."""
    raw = os.urandom(32) if random_bytes is None else random_bytes
    if len(raw) != 32:
        raise ContractError(E_ID_INPUT, "random_bytes must be 32 bytes")
    return "ses:" + raw.hex()


def observation_id(session_id: str, ordinal: int) -> str:
    """``obs:`` + sha256(JCS({session_id, ordinal})); ordinal counts from 0."""
    if not isinstance(session_id, str) or not _SES.match(session_id):
        raise ContractError(E_ID_INPUT, "session_id must match ^ses:[0-9a-f]{64}$")
    return "obs:" + _sha({"session_id": session_id, "ordinal": _require_int(ordinal, "ordinal", 0)})
