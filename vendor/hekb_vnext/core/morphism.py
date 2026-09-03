"""MemoryMorphism / IdentityMorphism — a Morphism f: A -> B in Hom(A, B),
per RFC-HEKB10 v1.1 Section C (Morphism) and Section A (Category, for the
Identity requirement).

Only the minimal structure Section C actually defines is implemented:
domain, codomain, and participation in composition/identity. Composition
itself lives in `composition.py`, not here — this module does not decide
when two morphisms may compose, only what one morphism *is*.

`dom`/`cod` are `MemoryObject`s, so a morphism's own identity (`morphism_id`)
is derived from `dom.object_id`, `cod.object_id`, and an opaque `label` — a
deterministic function of exactly the data RFC-HEKB10 Section C requires a
morphism to carry, nothing more. Composed-morphism canonicalization beyond
this triple is explicitly `NOT_ESTABLISHED` (task Section 3.1's own note)
and is not attempted here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .object import MemoryObject, canonical_bytes


@dataclass(frozen=True)
class MemoryMorphism:
    """A morphism f: dom -> cod. Immutable (TC-CAT-05).

    Multiple `MemoryMorphism` instances may share the same `dom` without
    contradiction — RFC-HEKB10 v1.1 Section A's category definition places
    no cardinality constraint on `Hom(A, B)` beyond it being a set, and the
    task's own Section 3.1 explicitly designates multiple same-domain
    morphisms as Structural Branching, not an error condition. Uniqueness
    of a *specific* morphism (by `morphism_id`) is still guaranteed by
    content-addressing; uniqueness of *how many* morphisms leave a given
    object is not, and must not be, enforced by this class."""

    dom: MemoryObject
    cod: MemoryObject
    label: Any = None
    morphism_id: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        payload = {"dom": self.dom.object_id, "cod": self.cod.object_id, "label": self.label}
        digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        object.__setattr__(self, "morphism_id", digest)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryMorphism):
            return NotImplemented
        return self.morphism_id == other.morphism_id

    def __hash__(self) -> int:
        return hash(self.morphism_id)

    def __repr__(self) -> str:
        return f"MemoryMorphism({self.dom!r} -> {self.cod!r}, id={self.morphism_id[:12]}...)"


class IdentityMorphism(MemoryMorphism):
    """The distinguished identity morphism id_A: A -> A, per RFC-HEKB10
    v1.1 Section A: "for every object A, an identity morphism
    id_A in Hom(A, A)". Exactly one is materialized per object by
    `Category.add_object` (see `category.py`) — this class itself does not
    enforce that uniqueness; it only marks an instance as *being* an
    identity, distinguishable from an ordinary self-loop a caller might
    otherwise construct with the same `(dom, cod)`.

    Deliberately not decorated with `@dataclass` itself: it reuses
    `MemoryMorphism`'s frozen field slots directly via `object.__setattr__`
    in its own `__init__`, avoiding the ambiguity of re-deriving a
    dataclass `__init__` over an already-dataclass parent."""

    _LABEL = "__identity__"

    def __init__(self, obj: MemoryObject) -> None:
        payload = {"dom": obj.object_id, "cod": obj.object_id, "label": self._LABEL}
        digest = hashlib.sha256(canonical_bytes(payload)).hexdigest()
        object.__setattr__(self, "dom", obj)
        object.__setattr__(self, "cod", obj)
        object.__setattr__(self, "label", self._LABEL)
        object.__setattr__(self, "morphism_id", digest)

    def __repr__(self) -> str:
        return f"IdentityMorphism(id_{self.dom!r})"
