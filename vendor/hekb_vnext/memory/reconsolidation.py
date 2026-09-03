"""Reconsolidation-as-derivation — implements the Stage 0 design hypothesis
("reconsolidation is representable as branch formation, not in-place
mutation") as its own named Memory Layer operation.

Structurally this is exactly `branching.branch()`: a new, independent
`MemoryObject` is created, the source object is never touched, and a
`MemoryMorphism` records the derivation. It is exposed under its own name
and its own edge tag (`RECONSOLIDATION_OP`) only so a provenance reader can
distinguish "this edge records a reconsolidation-style derivation" from an
ordinary branch without inspecting either endpoint's own content — no
additional category-theoretic mechanism is introduced beyond what
`branching.branch()` already uses.

Terminology (mandatory, per task guardrail): this module's own vocabulary
is strictly structural — "derivation", "branch", "provenance". It makes no
claim, in code, comments, or test names, that anything was "remembered",
"recalled", or "reconsidered" in a cognitive sense.
"""

from __future__ import annotations

from typing import Any

from core.category import Category
from core.morphism import MemoryMorphism
from core.object import MemoryObject

from .branching import branch

RECONSOLIDATION_OP = "reconsolidation"


def reconsolidate(category: Category, source: MemoryObject, new_context: Any) -> MemoryMorphism:
    """Creates `derived = MemoryObject(new_context)` and registers
    `source -> derived`, tagged as a reconsolidation-style derivation.
    `source` is never mutated (same guarantee as `branch()`, since this
    function delegates to it directly) — it remains independently
    retrievable, at its original `object_id`, after this call returns."""
    return branch(category, source, new_payload=new_context, op=RECONSOLIDATION_OP)


def is_reconsolidation_edge(m: MemoryMorphism) -> bool:
    """True iff `m`'s label was produced by `reconsolidate()`."""
    return isinstance(m.label, dict) and m.label.get("op") == RECONSOLIDATION_OP
