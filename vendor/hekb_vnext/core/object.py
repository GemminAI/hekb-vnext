"""MemoryObject — an Object A in Ob(C), per RFC-HEKB10 v1.1 Section A
(Category) and Section B (Object).

Category theory ascribes no internal structure to an object beyond its
identity and its network of morphisms (RFC-HEKB10 v1.1 Section B, citing
`RFC-STS00` Axiom 1 / the Yoneda Lemma corollary). This module implements
only that: an immutable, content-addressed container. It does not implement
Functor, Adjunction, or any construct beyond Object itself — those remain
out of Stage 1 scope per the task's own Guardrail 5 (No Premature Advanced
Math).

Pure standard library only (Guardrail 4): `hashlib`, `json`, `dataclasses`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    """Deterministic byte serialization used for content-addressing
    throughout this package. Restricted to JSON-representable values
    (str, int, float, bool, None, list/tuple, dict) with sorted object
    keys and no extraneous whitespace, so that identical logical content
    always serializes to identical bytes regardless of construction
    order. This is the minimal determinism guarantee TC-CAT-04 requires;
    no external canonicalization library is used (Guardrail 4)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


@dataclass(frozen=True)
class MemoryObject:
    """An object of the category. Immutable (frozen dataclass — any
    attempt to reassign a field after construction raises
    `dataclasses.FrozenInstanceError`, per TC-CAT-04).

    `object_id` is a SHA-256 hash of `payload`'s own canonical bytes — the
    object's identity is a deterministic function of its content, not an
    externally assigned label. Two `MemoryObject` instances constructed
    from equal payloads are the same object (`__eq__`/`__hash__` compare
    `object_id` only, matching content-addressing semantics)."""

    payload: Any
    object_id: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        digest = hashlib.sha256(canonical_bytes(self.payload)).hexdigest()
        object.__setattr__(self, "object_id", digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryObject):
            return NotImplemented
        return self.object_id == other.object_id

    def __hash__(self) -> int:
        return hash(self.object_id)

    def __repr__(self) -> str:
        return f"MemoryObject(object_id={self.object_id[:12]}...)"
