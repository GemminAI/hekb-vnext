"""Memory Semantics Layer — HEKB vNext Stage 2.

Built strictly on top of Stage 1's `core/` (imported, never modified).
Implements Branching, ContextualRecall, Reconsolidation (as derivation),
and Provenance Preservation using only Object/Morphism/Composition/
Category primitives already verified by Stage 1 — no Functor, Pushout,
Colimit, Adjunction, or Natural Transformation is used or implied anywhere
in this package (Guardrail).

Does not import, reference, or depend on `sensos`, `nvs-kernel`, or
`GemminAI/hekb`, and does not reuse any code from the decommissioned
`hekb_v3`. Does not connect to CS336 M0 or to HEX001 (both explicitly
deferred past this stage).
"""

from .branching import BRANCH_OP, branch, is_branch_edge
from .provenance import (
    AmbiguousProvenanceError,
    ProvenanceCycleError,
    provenance_path,
)
from .recall import ActiveSubgraph, recall, recall_by_ids
from .reconsolidation import RECONSOLIDATION_OP, is_reconsolidation_edge, reconsolidate

__all__ = [
    "BRANCH_OP",
    "branch",
    "is_branch_edge",
    "RECONSOLIDATION_OP",
    "reconsolidate",
    "is_reconsolidation_edge",
    "ActiveSubgraph",
    "recall",
    "recall_by_ids",
    "provenance_path",
    "AmbiguousProvenanceError",
    "ProvenanceCycleError",
]
