"""Builders for the Formal Store tests: IC-00 1.1.0 envelopes, and GateVerdicts made from them (the way U6T-1's GateClient makes
them: the hashes are computed here with the vendored JCS, nothing is copied from a stored verdict)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from graph import formal
from vendor.sensos_ic00 import ids
from vendor.sensos_ic00.jcs import jcs

SES = "ses:" + "1" * 64
OBS = "obs:" + "2" * 64
HT_SCHEMA = {"id": "nvs.hidden-trajectory", "version": "0.1.0"}
CATALOG_ROWS = [
    ["msr.trajectory-window", "window_rows", "raw_not_in_artifact"],
    ["msr.trajectory-window", "window_space_derivation", "rule_not_derivable"],
    ["nvs.hidden-trajectory", "contract_binding", "contract_absent"],
    ["nvs.hidden-trajectory", "contract_binding", "contract_not_supplied"],
    ["nvs.hidden-trajectory", "contract_binding.<field>", "contract_field_unknown"],
    ["nvs.hidden-trajectory", "points_bytes", "blob_not_supplied"],
    ["nvs.hidden-trajectory", "token_index_bytes", "blob_not_supplied"],
]
CATALOG_SHA = formal.catalog_sha256(CATALOG_ROWS)
BUNDLE = json.dumps(
    {"version": "gate-checker-bundle/1", "files": []}, sort_keys=True
).encode()
BUNDLE_SHA = hashlib.sha256(BUNDLE).hexdigest()
SENSOS_COMMIT = "4e5a75da8b54c1a9193c47b9ce8c75ec16bb2bd9"


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def seal(
    env: dict[str, Any], lookup: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    plain = {k: {"content_id": v["content_id"]} for k, v in (lookup or {}).items()}
    env["content_id"] = ids.content_id(env, plain)
    env["artifact_id"] = ids.artifact_id(env, plain)
    return env


def make_env(
    *,
    kind: str = "observation",
    payload: Any = None,
    parents: Any = (),
    seq: int = 1,
    obs: str = OBS,
    version: str = "1.1.0",
    schema: Mapping[str, str] = HT_SCHEMA,
    lookup: Mapping[str, Any] | None = None,
    space_id: str | None = "hid:" + "3" * 64,
    **fields: Any,
) -> dict[str, Any]:
    env: dict[str, Any] = {
        "contract": {"id": "IC-00", "version": version},
        "content_id": "",
        "artifact_id": "",
        "kind": kind,
        "payload_schema": dict(schema),
        "payload": payload if payload is not None else {"a": 1},
        "producer": {"name": "nvs-runtime", "version": "1.0.0", "build": "t"},
        "session_id": SES,
        "observation_id": obs,
        "event_span": None,
        "space_id": space_id,
        "parents": [
            {"role": r, "ordered": False, "artifact_ids": list(i)} for r, i in parents
        ],
        "policy_refs": [],
        "calibration_ref": None,
        "rule_refs": [],
        "observed_at": None,
        "seq": seq,
        "produced_at": "2026-11-17T00:00:00Z",
        "extensions": {},
    }
    env.update(fields)
    return seal(env, lookup)


def lookup_of(*envs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        e["artifact_id"]: {
            "content_id": e["content_id"],
            "seq": e["seq"],
            "observation_id": e["observation_id"],
        }
        for e in envs
    }


def seal_verdict(v: dict[str, Any]) -> dict[str, Any]:
    v["inputs"]["lookup_sha256"] = sha(jcs(v["inputs"]["lookup"]))
    v["verdict_sha256"] = sha(
        jcs({k: x for k, x in v.items() if k != "verdict_sha256"})
    )
    return v


def make_verdict(
    env: Mapping[str, Any],
    *,
    lookup: Mapping[str, Any] | None = None,
    result: str = "PASS",
    unverifiable: tuple[tuple[str, str], ...] = (),
    edit: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """A GateVerdict about ``env``. ``result`` FAIL gives an L1 failure (S8), PASS_UNVERIFIED lists ``unverifiable`` (check, reason).
    ``edit`` may change the document before it is sealed (to break one thing); call ``seal_verdict`` yourself after later edits."""
    ps = env["payload_schema"]
    window = ps["id"] == "msr.trajectory-window"
    v: dict[str, Any] = {
        "verdict_version": "gate-verdict/1",
        "subject": {
            "artifact_id": env["artifact_id"],
            "content_id": env["content_id"],
            "envelope_sha256": formal.envelope_sha256(env),
        },
        "result": "PASS",
        "layers": {
            "L1": {"status": "PASS", "codes": []},
            "L2": {
                "status": "PASS",
                "kind": "ic02" if window else "hidden_trajectory",
                "codes": [],
            },
            "L3": {"status": "PASS" if window else "not_applicable", "codes": []},
        },
        "codes": [],
        "violations": [],
        "unverifiable": [],
        "waived": [],
        "rules": {
            "ic00_contract_version": env["contract"]["version"],
            "payload_schema": {"id": ps["id"], "version": ps["version"]},
        },
        "inputs": {
            "lookup": dict(lookup or {}),
            "lookup_sha256": "0" * 64,
            "stored": [],
            "blobs_checked": [sha("points"), sha("index")],
            "contract_facts_sha256": sha("facts"),
        },
        "gate": {
            "checker": {
                "bundle_sha256": BUNDLE_SHA,
                "sensos_commit": SENSOS_COMMIT,
                "nvs_runtime_commit": SENSOS_COMMIT,
                "files": [
                    {
                        "repo": "sensos",
                        "path": "tools/contracts/gate_nvs.py",
                        "sha256": sha("gate_nvs"),
                    }
                ],
            },
            "runtime": {
                "python": "3.14.7",
                "implementation": "CPython",
                "packages": {"jsonschema": "4.26.0", "referencing": "0.37.0"},
                "platform": {"system": "Darwin", "machine": "arm64"},
            },
            "unverifiable_catalog": {"version": SENSOS_COMMIT, "sha256": CATALOG_SHA},
        },
        "produced_by": {
            "name": "nvs-runtime",
            "version": "1.0.0",
            "build": SENSOS_COMMIT,
        },
        "verdict_sha256": "0" * 64,
    }
    if result == "FAIL":
        v["result"] = "FAIL"
        v["layers"] = {
            "L1": {"status": "FAIL", "codes": ["S8"]},
            "L2": {"status": "not_run", "kind": None, "codes": []},
            "L3": {"status": "not_run", "codes": []},
        }
        v["codes"] = ["S8"]
        v["violations"] = [
            {"layer": "L1", "code": "S8", "message": "observation_id must be inherited"}
        ]
    elif result == "PASS_UNVERIFIED":
        v["result"] = "PASS_UNVERIFIED"
        v["layers"]["L2"]["status"] = "PASS_UNVERIFIED"
        v["unverifiable"] = [
            {"layer": "L2", "check": c, "reason": r} for c, r in unverifiable
        ]
    if edit is not None:
        edit(v)
    return seal_verdict(v)


def decl(**changes: Any) -> dict[str, Any]:
    d = formal.formal_declaration(
        catalog_rows=CATALOG_ROWS, catalog_version=SENSOS_COMMIT, created_by="tests"
    )
    d.update(changes)
    return d


def clone(x: Any) -> Any:
    return copy.deepcopy(x)
