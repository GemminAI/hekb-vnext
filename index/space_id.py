"""space_id: BLAKE2b-256 of the normalized tuple identifying a measurement space
(HEKBv3 改修設計書 v0.1 §3.1, H5). Two Trajectories only compose with each other if their
space_id matches -- this is the sole guard; it does not interpret the fields themselves.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

#: Required keys, per the design doc's §3.1 worked example. All must be present and of a
#: JSON-primitive type (str/int/float); extra keys are rejected so a caller can't silently
#: add an unnormalized field that two otherwise-identical spaces disagree on.
#: ``wiring_policy_sha256`` and ``time_policy_sha256`` (NVS-Kernel v7: how each port's arguments
#: are built, how ``dt`` is taken) are part of the space: trajectories measured under a different
#: wiring or time rule must not be composed. Adding them changes every ``space_id``; there is no
#: stored experience to keep compatible with (Tomo, 2026-10-12).
REQUIRED_KEYS = (
    "model_ref", "layer", "projection_sha256", "w_sha256",
    "anchor_policy_sha256", "wiring_policy_sha256", "time_policy_sha256",
    "board_format", "quant_step_log2",
)


#: Optional key, part of the fingerprint ONLY when present: the ingest dropped one or more
#: valueless columns (board-42-v2's P15, ``--drop-valueless-columns``). Written ``"P15=valueless"``
#: (comma-separated, sorted, for several). A space without it hashes exactly as before, so every
#: existing ``space_id`` (including the 38-column board-42-v1 ones) is unchanged.
OPTIONAL_KEYS = ("excluded_columns",)


class SpaceIdError(ValueError):
    pass


def _canonical_bytes(space: dict[str, Any]) -> bytes:
    missing = [k for k in REQUIRED_KEYS if k not in space]
    if missing:
        raise SpaceIdError(f"space is missing required keys: {missing}")
    extra = [k for k in space if k not in REQUIRED_KEYS and k not in OPTIONAL_KEYS]
    if extra:
        raise SpaceIdError(f"space has unexpected keys: {extra}")
    for k in REQUIRED_KEYS:
        if not isinstance(space[k], (str, int, float)):
            raise SpaceIdError(f"space[{k!r}] must be str/int/float, got {type(space[k]).__name__}")
    ordered = {k: space[k] for k in REQUIRED_KEYS}
    if "excluded_columns" in space:
        excluded = space["excluded_columns"]
        if not isinstance(excluded, str) or not excluded:
            raise SpaceIdError("space['excluded_columns'] must be a non-empty string (e.g. 'P15=valueless') or absent")
        ordered["excluded_columns"] = excluded
    return json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def compute_space_id(space: dict[str, Any]) -> str:
    """space: dict with exactly REQUIRED_KEYS -> hex BLAKE2b-256 digest."""
    return hashlib.blake2b(_canonical_bytes(space), digest_size=32).hexdigest()
