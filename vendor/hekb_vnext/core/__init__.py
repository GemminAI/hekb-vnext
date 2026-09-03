"""Core Categorical Substrate — HEKB vNext Stage 1.

Isolated CS336-local experimental package (Guardrail 1/4). Implements only
Object, Morphism, Identity, and Composition (Guardrail 6) — no Pushout,
Colimit, Adjunction, or Kan Extension (Guardrail 5). Pure standard library.

Does not import, reference, or depend on `sensos`, `nvs-kernel`, or
`GemminAI/hekb` (Guardrail 2), and does not reuse any code from the
decommissioned `hekb_v3` (Guardrail 3).
"""

from .category import Category, CompositionError, UnknownMorphismError
from .composition import compose, is_composite
from .morphism import IdentityMorphism, MemoryMorphism
from .object import MemoryObject, canonical_bytes

__all__ = [
    "Category",
    "CompositionError",
    "UnknownMorphismError",
    "compose",
    "is_composite",
    "IdentityMorphism",
    "MemoryMorphism",
    "MemoryObject",
    "canonical_bytes",
]
