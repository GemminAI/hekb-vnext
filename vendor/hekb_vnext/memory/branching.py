"""Branching — creates a new, independent `MemoryObject` B and registers a
`MemoryMorphism` A -> B in an existing `Category`, without altering A in
any way.

This module introduces no new category-theoretic construct. It is
orchestration over `Category.add_object` / `Category.add_morphism`, the
same primitives Stage 1's own TC-CAT-06 (Branch Preservation) already
exercises — Structural Branching (multiple morphisms from one domain) is
Stage 1's own, already-verified behavior; this module only gives it a
named, Memory-Layer-facing entry point. No Functor, Pushout, Colimit, or
Adjunction is used or implied.

`core/*.py` (Stage 1) is imported, never modified.
"""

from __future__ import annotations

from typing import Any

from core.category import Category
from core.morphism import MemoryMorphism
from core.object import MemoryObject

BRANCH_OP = "branch"


def branch(
    category: Category,
    source: MemoryObject,
    new_payload: Any,
    *,
    op: str = BRANCH_OP,
    context: Any = None,
) -> MemoryMorphism:
    """Creates `new_object = MemoryObject(new_payload)` and registers the
    morphism `source -> new_object` in `category`, tagged
    `{"op": op, "context": context}`.

    `source` is read, never written: this function calls no method that
    could mutate it, and `MemoryObject` is itself immutable (Stage 1,
    TC-CAT-04) — `source.object_id` is therefore guaranteed identical
    before and after this call.

    Calling this repeatedly with the same `source` and different
    `new_payload` values registers one additional morphism per call, all
    coexisting (Structural Branching) — no prior branch is disturbed,
    overwritten, or removed."""
    category.add_object(source)  # idempotent registration; never mutates source
    new_object = MemoryObject(payload=new_payload)
    edge = MemoryMorphism(dom=source, cod=new_object, label={"op": op, "context": context})
    return category.add_morphism(edge)


def is_branch_edge(m: MemoryMorphism) -> bool:
    """True iff `m`'s label was produced by `branch()` with `op == BRANCH_OP`
    (i.e. an ordinary branch, not a `reconsolidate()`-tagged derivation —
    see `reconsolidation.py`)."""
    return isinstance(m.label, dict) and m.label.get("op") == BRANCH_OP
