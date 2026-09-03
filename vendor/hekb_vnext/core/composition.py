"""compose(g, f) — the composition operator (g o f), per RFC-HEKB10 v1.1
Section A: "a composition operation ... subject to associativity and the
identity law."

Composability condition only: `g o f` is defined iff `cod(f) == dom(g)`.
Nothing else is checked or inferred.

Per the task's Section 3.1 note ("合成射の内部データ正規化仕様は現時点で
NOT_ESTABLISHED"), this module does not collapse, hash-compress, or
otherwise reduce a composite morphism's own provenance. The composite's
`label` preserves the ordered pair of the two morphism ids that produced
it — a primitive, traceable coupling relationship — rather than any
invented canonical form. `category.trace_provenance` (in `category.py`)
walks this structure back to its base morphisms; this module only builds
one composition step at a time.
"""

from __future__ import annotations

from .morphism import IdentityMorphism, MemoryMorphism

COMPOSE_TAG = "__compose__"


class CompositionError(ValueError):
    """Raised when g o f is requested but cod(f) != dom(g) — the
    composability condition RFC-HEKB10 v1.1 Section A requires."""


def compose(g: MemoryMorphism, f: MemoryMorphism) -> MemoryMorphism:
    """Returns the composite g o f : dom(f) -> cod(g).

    Composable iff `f.cod == g.dom` (object equality, i.e. same
    `object_id` — see `object.py`). Raises `CompositionError` otherwise;
    this function never silently substitutes a different pairing.

    The Identity Law (RFC-HEKB10 v1.1 Section A: "id_B o f = f = f o id_A")
    is not a canonicalization choice — it is composition's own definition
    when one operand is an `IdentityMorphism`, so it is implemented here
    directly rather than left to a general composite-normalization scheme
    (which remains NOT_ESTABLISHED and is not otherwise attempted by this
    module): composing with an identity returns the *other* operand
    unchanged (same `morphism_id`), not a new composite morphism."""
    if f.cod != g.dom:
        raise CompositionError(
            f"g o f is undefined: cod(f)={f.cod.object_id!r} does not equal dom(g)={g.dom.object_id!r}"
        )
    if isinstance(f, IdentityMorphism):
        return g  # g o id_A = g
    if isinstance(g, IdentityMorphism):
        return f  # id_B o f = f
    label = (COMPOSE_TAG, g.morphism_id, f.morphism_id)
    return MemoryMorphism(dom=f.dom, cod=g.cod, label=label)


def is_composite(m: MemoryMorphism) -> bool:
    """True iff `m` was produced by `compose()` (its label carries the
    ordered (g_id, f_id) provenance pair), rather than constructed
    directly as a primitive morphism or an identity."""
    return isinstance(m.label, tuple) and len(m.label) == 3 and m.label[0] == COMPOSE_TAG
