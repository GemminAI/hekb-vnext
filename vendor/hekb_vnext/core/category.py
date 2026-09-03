"""Category — a container managing one locally-small category's universe of
`MemoryObject`s and `MemoryMorphism`s, per RFC-HEKB10 v1.1 Section A.

This class does not itself prove the Identity or Associativity laws hold
for every possible triple of morphisms — that is what
`tests/test_category_axioms.py` (TC-CAT-01, TC-CAT-03) checks, empirically,
against instances this container produces. What `Category` provides is:

- exactly one `IdentityMorphism` materialized per registered object
  (Guardrail 6: bottom-up from Object/Morphism/Identity/Composition), and
- a registry letting a caller retrieve every morphism leaving a given
  object (Structural Branching, TC-CAT-06) and walk a composite morphism's
  provenance back to its base morphisms (TC-CAT-07), without inventing any
  construct beyond Object/Morphism/Identity/Composition (Guardrail 5: no
  Pushout, Colimit, Adjunction, or Kan Extension appears anywhere below).
"""

from __future__ import annotations

from .composition import CompositionError, compose, is_composite
from .morphism import IdentityMorphism, MemoryMorphism
from .object import MemoryObject


class UnknownMorphismError(KeyError):
    """Raised by `trace_provenance` when a composite's recorded provenance
    references a morphism_id this Category was never given."""


class Category:
    """A registry of `MemoryObject`s and `MemoryMorphism`s. Registration is
    additive only: no method on this class removes or mutates a
    previously-added object or morphism (append-only, matching the
    Immutability guarantee TC-CAT-04/05 already give each individual
    instance)."""

    def __init__(self) -> None:
        self._objects: dict[str, MemoryObject] = {}
        self._identities: dict[str, IdentityMorphism] = {}
        self._morphisms_by_id: dict[str, MemoryMorphism] = {}
        self._morphisms_by_dom: dict[str, list[MemoryMorphism]] = {}

    # -- Objects --------------------------------------------------------

    def add_object(self, obj: MemoryObject) -> MemoryObject:
        """Registers `obj` if not already present, and materializes its
        (unique) `IdentityMorphism`. Returns the canonical registered
        instance for `obj.object_id` (idempotent under re-registration of
        an object with equal content)."""
        if obj.object_id not in self._objects:
            self._objects[obj.object_id] = obj
            identity = IdentityMorphism(obj)
            self._identities[obj.object_id] = identity
            self._morphisms_by_dom.setdefault(obj.object_id, [])
            self._morphisms_by_id[identity.morphism_id] = identity
        return self._objects[obj.object_id]

    def identity_for(self, obj: MemoryObject) -> IdentityMorphism:
        registered = self.add_object(obj)
        return self._identities[registered.object_id]

    def objects(self) -> list[MemoryObject]:
        return list(self._objects.values())

    # -- Morphisms --------------------------------------------------------

    def add_morphism(self, f: MemoryMorphism) -> MemoryMorphism:
        """Registers `f` (and its `dom`/`cod` objects, if not already
        registered). Does not require `f` to have been produced by
        `compose()` — any `MemoryMorphism` may be registered directly,
        matching Structural Branching's requirement that a caller may add
        as many morphisms out of one object as it needs."""
        self.add_object(f.dom)
        self.add_object(f.cod)
        if f.morphism_id not in self._morphisms_by_id:
            self._morphisms_by_id[f.morphism_id] = f
            self._morphisms_by_dom[f.dom.object_id].append(f)
        return f

    def compose(self, g: MemoryMorphism, f: MemoryMorphism) -> MemoryMorphism:
        """Composes and registers `g o f` in one step. Raises
        `CompositionError` (propagated from `composition.compose`) if
        `cod(f) != dom(g)`."""
        composite = compose(g, f)
        return self.add_morphism(composite)

    def morphisms_from(self, obj: MemoryObject) -> list[MemoryMorphism]:
        """Every morphism registered with domain `obj`, in registration
        order. Length > 1 is Structural Branching, not an error
        (TC-CAT-06) — this method does not deduplicate or select among
        them."""
        registered = self.add_object(obj)
        return list(self._morphisms_by_dom.get(registered.object_id, []))

    def get_morphism(self, morphism_id: str) -> MemoryMorphism:
        try:
            return self._morphisms_by_id[morphism_id]
        except KeyError as exc:
            raise UnknownMorphismError(morphism_id) from exc

    # -- Provenance --------------------------------------------------------

    def trace_provenance(self, m: MemoryMorphism) -> list[MemoryMorphism]:
        """Deterministically unpacks `m` into the ordered sequence of base
        (non-composite) morphisms that produced it, left to right in
        application order (i.e. the order `compose()` would need to apply
        them to reproduce `m`'s own `dom -> cod` span).

        A base (non-composite, non-identity) morphism traces to itself,
        `[m]`. `compose(g, f)`'s composite traces to
        `trace_provenance(f) + trace_provenance(g)`, recursively — this is
        a pure read over the `(compose_tag, g_id, f_id)` label structure
        `composition.compose` already writes; no new provenance encoding
        is introduced here (Guardrail 5)."""
        if not is_composite(m):
            return [m]
        _, g_id, f_id = m.label
        g = self.get_morphism(g_id)
        f = self.get_morphism(f_id)
        return self.trace_provenance(f) + self.trace_provenance(g)


__all__ = ["Category", "CompositionError", "UnknownMorphismError"]
