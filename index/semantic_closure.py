"""hekb-vnext Phase B-②: Semantic Closure Query.

Ported from `GemminAI/hekb`'s `origin/main:experiments/_semantic_closure.py`
(EXP-HEKB002/003, "the Semantic Closure Engine — categorical retrieval").
Not vector search, not embedding search: given a query object already
registered in a `core.category.Category`, this computes the **minimal
self-contained subcategory** grounding it — every object/morphism reachable
by following real `MemoryMorphism` edges outward (premises the query
depends on: pullback direction) and inward (objects that depend on the
query: pushout/impact direction) — using hekb-vnext's own `compose()` for
every derived composition. Every output is sorted before being frozen into
the result, so `SemanticClosure` construction is order-independent by
construction, not by convention (same discipline as the source).

Adaptations from the origin implementation, made deliberately, not by
oversight:

- The origin's `KnowledgeRelation`/objects carry an external, caller-supplied
  `kind`/`category` typing (`relation_kind`/`object_category` dicts passed
  into `compute_closure`). hekb-vnext's `MemoryMorphism.label` and
  `ExperiencePayload` have no equivalent typed-kind system (this store is
  homogeneous: every object is an Experience) -- `ClosureMorphism.kind`
  here is instead read directly from `label.get("op")` when `label` is a
  dict (the shape `branch()` and the explicit-morphism endpoint both use),
  falling back to `"unknown"`; `ClosureObject` carries no `category` field
  at all, since there is nothing real to put in it.
- `core.composition.compose(g, f)` requires `f.cod == g.dom` and returns
  `dom(f) -> cod(g)` (see `core/composition.py`'s own docstring) -- the
  opposite parameter order from a same-named call in the origin module.
  `_derived_compositions` below calls `compose(relation, composed)` (not
  `compose(composed, relation)`, which is what the origin file's identical-
  looking loop does under *its* own compose's different convention) so
  that composability actually holds at each fold step under hekb-vnext's
  real, documented rule. This was verified by reading `core/composition.py`
  directly, not assumed from the origin source's parameter order.
- `_outgoing`/`_incoming` are built here entirely through `Category`'s
  public API (`objects()`, `morphisms_from()`) -- never through its private
  `_morphisms_by_dom`/`_morphisms_by_id` attributes -- so this module has
  no dependency on `Category`'s internal storage shape, only its documented
  contract. `core/*.py` itself is imported, never modified.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from core.category import Category
from core.morphism import MemoryMorphism


@dataclass(frozen=True, slots=True)
class ClosureObject:
    id: str


@dataclass(frozen=True, slots=True)
class ClosureMorphism:
    id: str
    source: str
    target: str
    kind: str


@dataclass(frozen=True, slots=True)
class DerivedComposition:
    morphism: str
    formula: str


@dataclass(frozen=True, slots=True)
class SemanticClosure:
    query_object_id: str
    objects: tuple[ClosureObject, ...]
    morphisms: tuple[ClosureMorphism, ...]
    derived_compositions: tuple[DerivedComposition, ...]
    pullback_roots: tuple[str, ...]
    pushout_wavefront: tuple[str, ...]
    is_minimal_self_contained: bool


def _morphism_kind(m: MemoryMorphism) -> str:
    if isinstance(m.label, dict):
        op = m.label.get("op")
        if isinstance(op, str):
            return op
    return "unknown"


def _outgoing(category: Category) -> dict[str, list[MemoryMorphism]]:
    """dom.object_id -> every morphism leaving it. Built entirely via
    `Category`'s public API (`objects()` + `morphisms_from()`).

    Deliberately omits a key for any object with zero outgoing morphisms
    (`Category.add_object` pre-seeds `_morphisms_by_dom` with an empty
    list for every registered object, but that is `Category`'s own
    internal bookkeeping, not part of its public contract -- and
    `pullback_roots`/`pushout_wavefront` below test membership in this
    dict specifically to mean "has at least one real edge", matching the
    origin implementation's own `_outgoing`/`_incoming`, which only ever
    creates a key by appending a relation to it."""
    index: dict[str, list[MemoryMorphism]] = {}
    for obj in category.objects():
        morphisms = list(category.morphisms_from(obj))
        if morphisms:
            index[obj.object_id] = morphisms
    return index


def _incoming(category: Category) -> dict[str, list[MemoryMorphism]]:
    """cod.object_id -> every morphism arriving at it. Derived by inverting
    `_outgoing`'s own result -- still only ever reads the public API."""
    index: dict[str, list[MemoryMorphism]] = {}
    for morphisms in _outgoing(category).values():
        for m in morphisms:
            index.setdefault(m.cod.object_id, []).append(m)
    return index


def _reachable(
    start: str,
    adjacency: dict[str, list[MemoryMorphism]],
    *,
    step_of: Callable[[MemoryMorphism], str],
) -> tuple[set[str], set[str]]:
    """BFS from `start` over `adjacency`; `step_of(morphism)` picks the next
    object id from a morphism touching the current frontier object."""
    visited_objects = {start}
    visited_morphisms: set[str] = set()
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for morphism in adjacency.get(current, ()):
            next_id = step_of(morphism)
            visited_morphisms.add(morphism.morphism_id)
            if next_id not in visited_objects:
                visited_objects.add(next_id)
                frontier.append(next_id)
    return visited_objects, visited_morphisms


def _derived_compositions(
    category: Category,
    query_id: str,
    outgoing: dict[str, list[MemoryMorphism]],
) -> tuple[DerivedComposition, ...]:
    """Fold every query-rooted path in the pullback (outgoing) graph via
    hekb-vnext's real `Category.compose`, one real `MemoryMorphism` per
    path. Not required to be acyclic: a path stops the moment it revisits
    a node already on it, rather than recursing forever (same guard as the
    origin implementation)."""
    results: list[DerivedComposition] = []

    def walk(current_id: str, path: tuple[MemoryMorphism, ...], on_path: frozenset[str]) -> None:
        if len(path) >= 2:
            composed = path[0]
            for morphism in path[1:]:
                # compose(g, f) requires f.cod == g.dom (core/composition.py);
                # `composed` is always f here (dom(composed) == query_id
                # throughout the fold), `morphism` is g.
                composed = category.compose(morphism, composed)
            formula = " o ".join(m.morphism_id for m in reversed(path))
            results.append(
                DerivedComposition(
                    morphism=f"m_derived: {query_id} -> {composed.cod.object_id}",
                    formula=formula,
                )
            )
        for morphism in outgoing.get(current_id, ()):
            next_id = morphism.cod.object_id
            if next_id in on_path:
                continue  # cycle: stop here rather than recurse forever
            walk(next_id, (*path, morphism), on_path | {next_id})

    walk(query_id, (), frozenset({query_id}))
    return tuple(sorted(results, key=lambda d: d.morphism))


def _verify_minimal_self_contained(
    objects: tuple[ClosureObject, ...],
    morphisms: tuple[ClosureMorphism, ...],
    query_id: str,
) -> bool:
    """A real post-hoc check, not a hardcoded flag: every morphism's
    endpoints are among `objects` (no dangling edges), every non-query
    object touches at least one morphism (no orphans a buggy closure
    algorithm could have smuggled in), and no object id repeats."""
    object_ids = [obj.id for obj in objects]
    if len(object_ids) != len(set(object_ids)):
        return False
    object_id_set = set(object_ids)

    touched: set[str] = set()
    for morphism in morphisms:
        if morphism.source not in object_id_set or morphism.target not in object_id_set:
            return False
        touched.add(morphism.source)
        touched.add(morphism.target)

    non_query_ids = object_id_set - {query_id}
    return non_query_ids <= touched


def compute_closure(category: Category, query_id: str) -> SemanticClosure:
    outgoing = _outgoing(category)
    incoming = _incoming(category)

    pullback_objects, pullback_morphisms = _reachable(
        query_id, outgoing, step_of=lambda m: m.cod.object_id
    )
    pushout_objects, pushout_morphisms = _reachable(
        query_id, incoming, step_of=lambda m: m.dom.object_id
    )

    all_object_ids = pullback_objects | pushout_objects
    all_morphism_ids = pullback_morphisms | pushout_morphisms
    # NOT {**outgoing, **incoming}: both dicts share object_id keys (every
    # object with any edge appears in both, once as a source and once as a
    # target), so merging by key would silently drop one side's morphism
    # list at each shared key instead of combining them. Iterate every list
    # from both indices instead; morphism_id (not the outer object_id key)
    # is what deduplicates here.
    morphisms_by_id: dict[str, MemoryMorphism] = {}
    for morphism_list in (*outgoing.values(), *incoming.values()):
        for m in morphism_list:
            if m.morphism_id in all_morphism_ids:
                morphisms_by_id[m.morphism_id] = m

    objects = tuple(sorted((ClosureObject(id=obj_id) for obj_id in all_object_ids), key=lambda o: o.id))
    morphisms = tuple(
        sorted(
            (
                ClosureMorphism(
                    id=m.morphism_id,
                    source=m.dom.object_id,
                    target=m.cod.object_id,
                    kind=_morphism_kind(m),
                )
                for m in morphisms_by_id.values()
            ),
            key=lambda m: m.id,
        )
    )

    pullback_roots = tuple(
        sorted(obj_id for obj_id in pullback_objects if obj_id != query_id and obj_id not in outgoing)
    )
    pushout_wavefront = tuple(
        sorted(obj_id for obj_id in pushout_objects if obj_id != query_id and obj_id not in incoming)
    )

    derived_compositions = _derived_compositions(category, query_id, outgoing)
    is_minimal_self_contained = _verify_minimal_self_contained(objects, morphisms, query_id)

    return SemanticClosure(
        query_object_id=query_id,
        objects=objects,
        morphisms=morphisms,
        derived_compositions=derived_compositions,
        pullback_roots=pullback_roots,
        pushout_wavefront=pushout_wavefront,
        is_minimal_self_contained=is_minimal_self_contained,
    )


__all__ = [
    "ClosureMorphism",
    "ClosureObject",
    "DerivedComposition",
    "SemanticClosure",
    "compute_closure",
]
