"""ContextualRecall — read-only extraction of an Active Subgraph from an
existing `Category`, given an explicit, caller-supplied selection
predicate. Never mutates the source graph.

NOT_ESTABLISHED, and deliberately not implemented anywhere in this module:
any numerical activation score — cosine similarity, embedding distance,
top-k ranking, salience, threshold, or time decay. No document this
investigation has read (canonical HEKB, RFC-HEKB00-14, NVS-Kernel,
`hekb_v3`, or this task's own Stage 0 design draft) defines a concrete,
adopted formula for any of these. The Stage 0 draft *proposed* reusing an
existing HEX00x cosine-similarity pattern as one future option — that is
this project's own unaccepted proposal, not a specification, and per this
task's explicit instruction an unspecified scoring function is not
invented here. "Active" in this module means "selected by an explicit,
caller-supplied predicate," full stop — a mechanical filter, not a ranked
activation.

Non-mutation guarantee: `recall()` calls no `Category` method other than
`objects()` and `morphisms_from()` (both read-only), and returns
references to the Category's own already-registered `MemoryObject`/
`MemoryMorphism` instances — it constructs no new object or morphism, so
it cannot itself introduce a hash it did not already read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.category import Category
from core.morphism import MemoryMorphism
from core.object import MemoryObject


@dataclass(frozen=True)
class ActiveSubgraph:
    """A snapshot of which already-registered objects/morphisms one
    `recall()` call selected. Holds the same `MemoryObject`/
    `MemoryMorphism` instances the source `Category` already owns —
    no content is copied, no new hash is computed."""

    objects: tuple[MemoryObject, ...]
    morphisms: tuple[MemoryMorphism, ...]

    def object_ids(self) -> frozenset[str]:
        return frozenset(o.object_id for o in self.objects)

    def morphism_ids(self) -> frozenset[str]:
        return frozenset(m.morphism_id for m in self.morphisms)


def recall(category: Category, predicate: Callable[[MemoryObject], bool]) -> ActiveSubgraph:
    """Returns the `ActiveSubgraph` of objects for which `predicate(obj)`
    is true, together with every morphism (already registered in
    `category`) whose `dom` or `cod` is among those objects.

    Read-only: this function does not call `add_object`, `add_morphism`,
    or `compose` on `category` — see module docstring for the guarantee
    this gives."""
    selected_objects = tuple(o for o in category.objects() if predicate(o))
    selected_ids = {o.object_id for o in selected_objects}
    morphisms: list[MemoryMorphism] = []
    seen_morphism_ids: set[str] = set()
    for obj in category.objects():
        for m in category.morphisms_from(obj):
            if m.morphism_id in seen_morphism_ids:
                continue
            if m.dom.object_id in selected_ids or m.cod.object_id in selected_ids:
                morphisms.append(m)
                seen_morphism_ids.add(m.morphism_id)
    return ActiveSubgraph(objects=selected_objects, morphisms=tuple(morphisms))


def recall_by_ids(category: Category, object_ids: set[str]) -> ActiveSubgraph:
    """Convenience selector: recall by explicit `object_id` membership.
    Still no scoring — membership in a caller-supplied set is itself the
    predicate."""
    return recall(category, lambda o: o.object_id in object_ids)
