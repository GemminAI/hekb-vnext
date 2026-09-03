"""Provenance Preservation — deterministic backward traversal over
Branching/Reconsolidation edges, from a descendant `MemoryObject` back to
its root.

This is deliberately distinct from Stage 1's `Category.trace_provenance`,
which unpacks ONE composite morphism (built via `core.composition.compose`)
into its ordered base-morphism sequence. `branch()`/`reconsolidate()` never
call `compose()` — they each register one ordinary, non-composite
`MemoryMorphism` per call — so what Branching/Reconsolidation actually
produce is a CHAIN of separate morphisms across the object graph
(`root -> ... -> A -> B`), not a single composite. This module walks that
chain instead. Where a caller's chain does happen to include a composite
morphism (e.g. because `core.category.Category.compose` was also used
directly), `Category.trace_provenance` remains the correct tool for that
morphism specifically — it is not reimplemented here, and this module does
not call it, since Branching/Reconsolidation never produce composites.

`core/*.py` is read through its existing public methods only
(`objects()`, `morphisms_from()`) — `category.py` is not modified or
subclassed to add a reverse index; this module builds its own, fresh, on
every call, so it cannot desynchronize from the source graph."""

from __future__ import annotations

from core.category import Category
from core.morphism import MemoryMorphism
from core.object import MemoryObject


class AmbiguousProvenanceError(RuntimeError):
    """Raised when an object has more than one incoming (non-identity)
    morphism recorded in the category. This module defines no
    merge/precedence rule for that case — it is NOT_ESTABLISHED — and
    refuses to guess one rather than silently picking an edge."""


class ProvenanceCycleError(RuntimeError):
    """Raised if backward traversal would revisit an object. Not expected
    under append-only Branching/Reconsolidation; guarded defensively."""


def _incoming_index(category: Category) -> dict[str, list[MemoryMorphism]]:
    """A read-only reverse index (object_id -> morphisms with that `cod`),
    rebuilt from `category`'s own public registry on every call."""
    index: dict[str, list[MemoryMorphism]] = {}
    seen_morphism_ids: set[str] = set()
    for obj in category.objects():
        for m in category.morphisms_from(obj):
            if m.morphism_id in seen_morphism_ids:
                continue
            seen_morphism_ids.add(m.morphism_id)
            if m.dom.object_id == m.cod.object_id:
                continue  # identity self-loop — not a provenance edge
            index.setdefault(m.cod.object_id, []).append(m)
    return index


def provenance_path(category: Category, obj: MemoryObject) -> list[MemoryMorphism]:
    """Returns the ordered chain `[root_edge, ..., edge_into_obj]` of
    morphisms leading to `obj`. Returns `[]` if `obj` has no recorded
    incoming morphism (it is itself a root).

    Deterministic for any fixed graph state: this function performs no
    scoring, ranking, or tie-breaking — see `AmbiguousProvenanceError` for
    what happens when the input graph does not have the single-parent
    shape this walk requires."""
    incoming = _incoming_index(category)
    chain: list[MemoryMorphism] = []
    current = obj
    seen: set[str] = set()
    while True:
        candidates = incoming.get(current.object_id, [])
        if not candidates:
            break
        if len(candidates) > 1:
            raise AmbiguousProvenanceError(
                f"object {current.object_id!r} has {len(candidates)} incoming morphisms; "
                "no merge/precedence rule is defined for this case (NOT_ESTABLISHED)"
            )
        if current.object_id in seen:
            raise ProvenanceCycleError(f"cycle detected at object {current.object_id!r}")
        seen.add(current.object_id)
        edge = candidates[0]
        chain.append(edge)
        current = edge.dom
    return list(reversed(chain))
