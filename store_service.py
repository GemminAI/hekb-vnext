"""HEKB vNext Experience Store — FastAPI wrapper (My_LLM_Builder_App_SPEC.md T7).

Thin HTTP wrapper around the vendored HEKB vNext categorical core
(`vendor/hekb_vnext/{core,memory}`, copied verbatim from
`/Users/tomonam3/Projects/experiments/cs336/hekb_vnext`, which itself has no
persistence or HTTP layer of its own). This module owns everything that
package does not: content-addressed on-disk storage under
`<data_dir>/{objects,relations,lineages,meta}/`, the FastAPI routes, and
process startup (host/port/data-dir).

Three-layer data model (SPEC-HEKB-003_v2 §2.1):

- Layer 1 (hashed core) — `objects/<object_id>.json`'s `payload`: the
  reserved-key `ExperiencePayload` structure whose canonical JSON bytes
  define `object_id`. No mutable value may appear here.
- Layer 2 (lineage bookkeeping) — `objects/<object_id>.json`'s top level:
  `object_id, lineage_id, parent_id, morphism_id, op, context, created_at`.
  This service's own bookkeeping concept, not part of the vendored core
  (which defines Object/Morphism/Category only). Posting an experience with
  no `parent_id` starts a new lineage rooted at that experience's own
  content-addressed object_id; posting with a `parent_id` calls the
  vendored `branch()` to record a dom->cod morphism from parent to child
  and joins the parent's lineage.
- Layer 3 (mutable sidecar) — `meta/<object_id>.json`: `recall_count`,
  `last_recalled_at`, `archived`, `provenance`. Rebuildable; never affects
  `object_id`.

Lineage is a tree, not a linear chain (SPEC-HEKB-002_v2 §2.2): the same
`parent_id` may have multiple children (branching). `lineages/<id>.json`'s
`object_ids` array is an insertion-order log, not an ancestor chain — its
last entry is always the most recently created object in that lineage and
is therefore always a leaf at query time (nothing has been posted with it
as parent yet), but ancestor recall must not simply slice this array.
Instead `GET /experience/recall` reconstructs a vendored `Category` from
every stored parent->child edge and calls
`vendor/hekb_vnext/memory/provenance.py`'s `provenance_path` to walk the
real dom->cod chain back to the root.

Content addressing is delegated entirely to `vendor.hekb_vnext.core.object.
MemoryObject`: an experience's `object_id` is a SHA-256 hash of its own
payload's canonical bytes, so identical payloads always resolve to the same
object_id regardless of when or how many times they are posted (POST
/experience is idempotent for a given (payload, parent_id) pair).

Governance (SPEC-HEKB-002_v2 §3.1): `POST /experience` requires a valid
`X-Audit-Signature` header — a P-256 ECDSA signature (DER, base64 or hex)
over the payload's own canonical bytes (the `object_id` preimage),
verified against a public key loaded from `HEKB_AUDIT_PUBLIC_KEY_PATH`.

`mlba-core` (`/Users/tomonam3/Projects/mlba-core`) was investigated as the
literal integration target the spec names, and has no signature/ledger
code at all — its only `extern "C"` exports are the two geometry distance
functions in `ffi.rs`, and `lib.rs`'s own comment defers "ledger / license
/ cloud" to later. The real P-256 sign/verify logic lives in
`MLBAKit/Security/{AuditLedger,SecureEnclaveManager,LedgerVerifier}.swift`,
Swift-only, backed by Secure Enclave hardware keys — `LedgerVerifier.swift`
itself documents choosing Swift over the originally-specified Rust
placement for the same pragmatic reasons. Verification here is therefore
implemented natively in Python (`cryptography`), not via any mlba-core FFI
call, mirroring `LedgerVerifier`'s own algorithm (canonical bytes -> ECDSA
P-256 verify against a public key only — verification never needs the
private/Secure-Enclave key). `HEKB_AUDIT_PUBLIC_KEY_PATH` accepts either a
PEM SPKI file or a raw 65-byte X9.63 uncompressed point (the exact format
`SecKeyCopyExternalRepresentation` produces), so a real key exported from
`SecureEnclaveManager.publicKeyData()` can be dropped in directly. Until
that env var is set to a valid key, verification always fails closed and
every `POST /experience` is rejected with `403` — no bypass exists.

Out of scope for this revision (see SPEC-HEKB-002_v2 §5 / §7): governing
`POST /experience/{id}/recall` or `/archive` the same way (spec scopes
this to `POST /experience` only), ephemeral-port + PID-watchdog process
isolation, and `GET /search`. `meta.provenance` is still always `null` —
this module does not itself write to the Ledger, only verifies against it.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "hekb_vnext"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.category import Category  # noqa: E402
from core.morphism import MemoryMorphism  # noqa: E402
from core.object import MemoryObject, canonical_bytes  # noqa: E402
from memory.branching import branch  # noqa: E402
from memory.provenance import (  # noqa: E402
    AmbiguousProvenanceError,
    ProvenanceCycleError,
    provenance_path,
)
from auth.local_auth import BearerTokenGuard, generate_bearer_token  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8300
DEFAULT_DATA_DIR = "./experience"
DEFAULT_RECALL_DEPTH = 5
AUDIT_PUBLIC_KEY_ENV = "HEKB_AUDIT_PUBLIC_KEY_PATH"
DEFAULT_APP_SUPPORT_DIR = str(Path.home() / "Library" / "Application Support" / "MyLLMBuilder")

# SPEC-HEKB-REFACTOR-2026-v2.0 section 5.3 (Step 2): a NEW transport-auth
# layer, separate from the existing X-Audit-Signature payload governance
# below. Applies to every route including /health -- "unauthenticated local
# HTTP request -> 401" is stated without carving out health checks, and a
# local ephemeral token costs nothing to check on every call. `_bearer_guard`
# starts with token=None (fails closed on every request) until `main()` (or
# a test) generates/loads a real token and assigns `_bearer_guard.token`.
_bearer_guard = BearerTokenGuard(token=None)


def _require_bearer_token(authorization: str | None = Header(None)) -> None:
    _bearer_guard(authorization)


def _data_dir() -> Path:
    return Path(os.environ.get("HEKB_DATA_DIR", DEFAULT_DATA_DIR)).resolve()


def _objects_dir() -> Path:
    return _data_dir() / "objects"


def _relations_dir() -> Path:
    return _data_dir() / "relations"


def _lineages_dir() -> Path:
    return _data_dir() / "lineages"


def _meta_dir() -> Path:
    return _data_dir() / "meta"


def _ensure_layout() -> None:
    for d in (_objects_dir(), _relations_dir(), _lineages_dir(), _meta_dir()):
        d.mkdir(parents=True, exist_ok=True)


def _object_path(object_id: str) -> Path:
    return _objects_dir() / f"{object_id}.json"


def _relation_path(morphism_id: str) -> Path:
    return _relations_dir() / f"{morphism_id}.json"


def _lineage_path(lineage_id: str) -> Path:
    return _lineages_dir() / f"{lineage_id}.json"


def _meta_path(object_id: str) -> Path:
    return _meta_dir() / f"{object_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _default_sidecar(object_id: str) -> dict[str, Any]:
    """The rebuildable Layer-3 sidecar (SPEC-HEKB-003_v2 §2.1): loss of this
    file is not catastrophic — `recall_count` resets to 0 and `provenance`
    to `None` (signature verification is out of scope here; see module
    docstring)."""
    return {
        "object_id": object_id,
        "recall_count": 0,
        "last_recalled_at": None,
        "archived": False,
        "provenance": None,
    }


class ExperiencePayload(BaseModel):
    """Layer 1 (hashed core). `object_id` is the SHA-256 of this model's own
    canonical JSON bytes (`canonical_dict()`), so no mutable value may live
    here. The reserved key set is exactly these seven fields
    (SPEC-HEKB-003_v2 §2.1) — anything else must go inside `ext`, which is
    itself opaque and not schema-checked."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    schema_: str = Field(..., alias="schema", description="Payload schema tag, e.g. 'hekb.experience/1'.")
    raw_input: Any = Field(..., description="The raw, unresolved input.")
    resolved_meaning: Any = Field(None, description="The resolved/disambiguated meaning.")
    action: str | None = Field(None, description="The action taken as a result, if any.")
    title: str | None = Field(None, description="Short human-readable label for graph/topology display.")
    tags: list[str] = Field(default_factory=list, description="Free-form classification tags.")
    ext: dict[str, Any] = Field(default_factory=dict, description="Opaque, unindexed extension bag.")

    def canonical_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


class ExperienceIn(BaseModel):
    payload: ExperiencePayload = Field(..., description="Layer-1 reserved-key payload.")
    parent_id: str | None = Field(
        None, description="object_id of the parent experience to branch from; omit to start a new lineage."
    )
    op: str = Field("experience", description="Edge tag recorded on the branch morphism when parent_id is given.")
    context: Any = Field(None, description="Opaque context recorded on the branch morphism.")


class ExperienceOut(BaseModel):
    object_id: str
    lineage_id: str
    parent_id: str | None
    morphism_id: str | None
    op: str
    context: Any
    payload: dict[str, Any]
    created_at: str


class RecallOut(BaseModel):
    lineage_id: str | None
    count: int
    experiences: list[ExperienceOut]


class SidecarOut(BaseModel):
    object_id: str
    recall_count: int
    last_recalled_at: str | None
    archived: bool
    provenance: dict[str, Any] | None = None


class GraphNodeOut(BaseModel):
    object_id: str
    title: str | None
    tags: list[str]
    recall_count: int
    archived: bool
    created_at: str


class GraphEdgeOut(BaseModel):
    source: str
    target: str
    relation: str = "lineage_parent"


class GraphTopologyOut(BaseModel):
    nodes: list[GraphNodeOut]
    edges: list[GraphEdgeOut]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _ensure_layout()
    yield


app = FastAPI(
    title="hekb-vnext",
    description="HEKB vNext Experience Store — FastAPI wrapper around the vendored HEKB vNext core (T7).",
    lifespan=_lifespan,
    dependencies=[Depends(_require_bearer_token)],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hekb-vnext"}


def _load_audit_public_key() -> ec.EllipticCurvePublicKey:
    """Loads the governance public key from `HEKB_AUDIT_PUBLIC_KEY_PATH`.
    Fails closed (raises `403`, never a bypass) if the env var is unset, the
    file is unreadable, or its contents aren't a P-256 public key — a
    missing/misconfigured key must never be silently treated as
    'verification not required'."""
    path_str = os.environ.get(AUDIT_PUBLIC_KEY_ENV)
    if not path_str:
        raise HTTPException(
            status_code=403,
            detail=f"governance verification unavailable: {AUDIT_PUBLIC_KEY_ENV} is not configured",
        )

    try:
        data = Path(path_str).read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=403,
            detail=f"governance verification unavailable: cannot read public key at {path_str!r}: {exc}",
        ) from exc

    try:
        if data.lstrip().startswith(b"-----BEGIN"):
            key = serialization.load_pem_public_key(data)
        else:
            key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), data)
    except Exception as exc:  # noqa: BLE001 — any parse failure is a 403, not a 500
        raise HTTPException(
            status_code=403,
            detail=f"governance verification unavailable: malformed public key at {path_str!r}: {exc}",
        ) from exc

    if not isinstance(key, ec.EllipticCurvePublicKey) or key.curve.name != "secp256r1":
        raise HTTPException(
            status_code=403,
            detail="governance verification unavailable: configured key is not a P-256 (secp256r1) public key",
        )
    return key


_HEX_SIGNATURE_RE = re.compile(r"^[0-9a-fA-F]+$")


def _decode_signature(header_value: str) -> bytes:
    """Hex is checked first, deliberately: hex digits are a strict subset
    of the base64 alphabet, so a hex string whose length happens to be a
    multiple of 4 is also syntactically valid (garbage) base64 — trying
    base64 first would silently misdecode a fraction of legitimate hex
    signatures instead of rejecting or accepting them correctly. A real
    base64 signature has a negligible chance of consisting entirely of
    hex-alphabet characters, so this ordering has no practical ambiguity
    the other way."""
    stripped = header_value.strip()
    if _HEX_SIGNATURE_RE.fullmatch(stripped) and len(stripped) % 2 == 0:
        return bytes.fromhex(stripped)
    try:
        return base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=403, detail="X-Audit-Signature is neither valid hex nor valid base64"
        ) from None


def _verify_audit_signature(payload_dict: dict[str, Any], signature_header: str | None) -> None:
    """SPEC-HEKB-002_v2 §3.1: signs/verifies the payload's own canonical
    bytes (the `object_id` preimage), matching `SecKeyVerifySignature(...,
    .ecdsaSignatureMessageX962SHA256, ...)`'s pairing on the Swift side —
    SHA-256 is applied internally by `ec.ECDSA(hashes.SHA256())`, not by
    the caller. Called before any read or write in `create_experience`."""
    if not signature_header:
        raise HTTPException(status_code=403, detail="X-Audit-Signature header is required")

    public_key = _load_audit_public_key()
    signature_bytes = _decode_signature(signature_header)
    message = canonical_bytes(payload_dict)

    try:
        public_key.verify(signature_bytes, message, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature as exc:
        raise HTTPException(status_code=403, detail="X-Audit-Signature verification failed") from exc


@app.post("/experience", response_model=ExperienceOut, status_code=201)
def create_experience(
    body: ExperienceIn,
    x_audit_signature: str | None = Header(None, alias="X-Audit-Signature"),
) -> ExperienceOut:
    payload_dict = body.payload.canonical_dict()
    _verify_audit_signature(payload_dict, x_audit_signature)

    _ensure_layout()
    object_id = MemoryObject(payload=payload_dict).object_id
    existing = _read_json(_object_path(object_id))
    if existing is not None:
        if existing.get("parent_id") != body.parent_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"object {object_id} already exists with parent_id={existing.get('parent_id')!r}; "
                    "a content-addressed object cannot be re-parented "
                    f"(requested parent_id={body.parent_id!r})"
                ),
            )
        return ExperienceOut(**existing)

    morphism_id: str | None = None
    if body.parent_id is None:
        lineage_id = object_id
    else:
        parent_record = _read_json(_object_path(body.parent_id))
        if parent_record is None:
            raise HTTPException(status_code=404, detail=f"parent_id {body.parent_id!r} not found")
        lineage_id = parent_record["lineage_id"]
        parent_object = MemoryObject(payload=parent_record["payload"])
        edge = branch(Category(), parent_object, new_payload=payload_dict, op=body.op, context=body.context)
        object_id = edge.cod.object_id
        morphism_id = edge.morphism_id
        _write_json_atomic(
            _relation_path(morphism_id),
            {"morphism_id": morphism_id, "dom": edge.dom.object_id, "cod": edge.cod.object_id, "label": edge.label},
        )

    record = {
        "object_id": object_id,
        "lineage_id": lineage_id,
        "parent_id": body.parent_id,
        "morphism_id": morphism_id,
        "op": body.op,
        "context": body.context,
        "payload": payload_dict,
        "created_at": _now_iso(),
    }
    _write_json_atomic(_object_path(object_id), record)

    lineage = _read_json(_lineage_path(lineage_id)) or {"lineage_id": lineage_id, "object_ids": []}
    lineage["object_ids"].append(object_id)
    _write_json_atomic(_lineage_path(lineage_id), lineage)

    _write_json_atomic(_meta_path(object_id), _default_sidecar(object_id))

    return ExperienceOut(**record)


def _build_category() -> Category:
    """Reconstructs a vendored `Category` by replaying every stored
    parent->child edge as a `MemoryMorphism`, reproducing exactly what
    `branch()` registered at write time (same dom/cod payload, same
    `{"op", "context"}` label) — so `morphism_id`s recompute identically and
    `provenance_path` walks the real graph, not a copy of it."""
    category = Category()
    for path in _objects_dir().glob("*.json"):
        record = _read_json(path)
        if record is None or record.get("parent_id") is None:
            continue
        parent_record = _read_json(_object_path(record["parent_id"]))
        if parent_record is None:
            continue
        dom = MemoryObject(payload=parent_record["payload"])
        cod = MemoryObject(payload=record["payload"])
        edge = MemoryMorphism(dom=dom, cod=cod, label={"op": record["op"], "context": record["context"]})
        category.add_morphism(edge)
    return category


def _ancestor_chain(target_record: dict[str, Any], depth: int) -> list[dict[str, Any]]:
    """The ordered ancestor chain ending at `target_record`, oldest first,
    truncated to `depth` ancestor hops (target itself counts as hop 0, so
    at most `depth + 1` records are returned) — per SPEC-HEKB-002_v2 §3.2."""
    category = _build_category()
    target_obj = MemoryObject(payload=target_record["payload"])
    try:
        chain = provenance_path(category, target_obj)
    except (AmbiguousProvenanceError, ProvenanceCycleError) as exc:
        raise HTTPException(status_code=500, detail=f"provenance graph inconsistent: {exc}") from exc

    if chain:
        ids = [chain[0].dom.object_id] + [m.cod.object_id for m in chain]
    else:
        ids = [target_obj.object_id]

    ids = ids[-(depth + 1) :]

    records: list[dict[str, Any]] = []
    for oid in ids:
        record = _read_json(_object_path(oid))
        if record is None:
            raise HTTPException(status_code=500, detail=f"ancestor object {oid!r} missing from objects/")
        records.append(record)
    return records


def _latest_leaf_object_id(lineage_id: str) -> str:
    """The most recently created object in `lineage_id`. This is always a
    leaf at query time: nothing can have been posted with it as parent_id
    before it itself was created."""
    lineage = _read_json(_lineage_path(lineage_id))
    if lineage is None or not lineage.get("object_ids"):
        raise HTTPException(status_code=404, detail=f"lineage_id {lineage_id!r} not found")
    return lineage["object_ids"][-1]


@app.get("/experience/recall", response_model=RecallOut)
def recall_experience(
    object_id: str | None = Query(None, description="Return exactly this experience, by content-addressed id."),
    lineage_id: str | None = Query(
        None, description="Ancestor chain (root..leaf) ending at this lineage's latest leaf node."
    ),
    from_object_id: str | None = Query(None, description="Ancestor chain (root..node) ending at this specific node."),
    depth: int = Query(
        DEFAULT_RECALL_DEPTH,
        ge=0,
        description="Ancestor hops to walk back from the start node (the node itself counts as hop 0).",
    ),
    limit: int = Query(
        50, ge=1, le=1000, description="Debug fallback: most recent writes across all lineages, no filter applied."
    ),
) -> RecallOut:
    _ensure_layout()

    if object_id is not None:
        record = _read_json(_object_path(object_id))
        if record is None:
            raise HTTPException(status_code=404, detail=f"object_id {object_id!r} not found")
        return RecallOut(lineage_id=record["lineage_id"], count=1, experiences=[ExperienceOut(**record)])

    if from_object_id is not None:
        target_record = _read_json(_object_path(from_object_id))
        if target_record is None:
            raise HTTPException(status_code=404, detail=f"from_object_id {from_object_id!r} not found")
        records = _ancestor_chain(target_record, depth)
        return RecallOut(
            lineage_id=target_record["lineage_id"],
            count=len(records),
            experiences=[ExperienceOut(**r) for r in records],
        )

    if lineage_id is not None:
        leaf_id = _latest_leaf_object_id(lineage_id)
        target_record = _read_json(_object_path(leaf_id))
        if target_record is None:
            raise HTTPException(status_code=500, detail=f"leaf object {leaf_id!r} missing from objects/")
        records = _ancestor_chain(target_record, depth)
        return RecallOut(
            lineage_id=lineage_id,
            count=len(records),
            experiences=[ExperienceOut(**r) for r in records],
        )

    paths = sorted(_objects_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    records = [ExperienceOut(**_read_json(p)) for p in paths]
    return RecallOut(lineage_id=None, count=len(records), experiences=records)


@app.get("/graph/topology", response_model=GraphTopologyOut)
def graph_topology(
    lineage_id: str | None = Query(None, description="Restrict nodes/edges to this lineage; omit for the whole store."),
    include_archived: bool = Query(
        False, description="Include nodes whose sidecar has archived=true. Excluded nodes' edges are dropped too."
    ),
) -> GraphTopologyOut:
    """Visual Graph node/edge feed (SPEC-HEKB-003_v2 §3.2). Pure read: never
    updates `recall_count` or any other sidecar/object field, and performs
    no write of any kind — reads `objects/` and `meta/` only."""
    _ensure_layout()

    included_ids: set[str] = set()
    records_by_id: dict[str, dict[str, Any]] = {}
    nodes: list[GraphNodeOut] = []

    for path in _objects_dir().glob("*.json"):
        record = _read_json(path)
        if record is None:
            continue
        if lineage_id is not None and record["lineage_id"] != lineage_id:
            continue
        sidecar = _read_json(_meta_path(record["object_id"])) or _default_sidecar(record["object_id"])
        if sidecar["archived"] and not include_archived:
            continue

        records_by_id[record["object_id"]] = record
        included_ids.add(record["object_id"])
        payload = record["payload"]
        nodes.append(
            GraphNodeOut(
                object_id=record["object_id"],
                title=payload.get("title"),
                tags=payload.get("tags", []),
                recall_count=sidecar["recall_count"],
                archived=sidecar["archived"],
                created_at=record["created_at"],
            )
        )

    nodes.sort(key=lambda n: n.created_at)

    edges = [
        GraphEdgeOut(source=record["parent_id"], target=object_id_)
        for object_id_, record in records_by_id.items()
        if record.get("parent_id") is not None and record["parent_id"] in included_ids
    ]
    edges.sort(key=lambda e: (e.source, e.target))

    return GraphTopologyOut(nodes=nodes, edges=edges)


@app.post("/experience/{object_id}/recall", response_model=SidecarOut)
def touch_recall(object_id: str) -> SidecarOut:
    """The only endpoint that increments `recall_count` (SPEC-HEKB-003_v2
    §3.4). Writes exclusively to `meta/<object_id>.json` — `objects/` is
    never touched, so `object_id` cannot change as a side effect of being
    recalled."""
    _ensure_layout()
    if _read_json(_object_path(object_id)) is None:
        raise HTTPException(status_code=404, detail=f"object_id {object_id!r} not found")

    sidecar = _read_json(_meta_path(object_id)) or _default_sidecar(object_id)
    sidecar["recall_count"] = sidecar.get("recall_count", 0) + 1
    sidecar["last_recalled_at"] = _now_iso()
    _write_json_atomic(_meta_path(object_id), sidecar)
    return SidecarOut(**sidecar)


@app.post("/experience/{object_id}/archive", response_model=SidecarOut)
def archive_experience(object_id: str) -> SidecarOut:
    """Soft-delete only (SPEC-HEKB-003_v2 §3.5 / SPEC-ARCH-IMPACT-001_v2
    §0.1): flips the sidecar's `archived` flag. No physical delete is
    implemented anywhere in this service."""
    _ensure_layout()
    if _read_json(_object_path(object_id)) is None:
        raise HTTPException(status_code=404, detail=f"object_id {object_id!r} not found")

    sidecar = _read_json(_meta_path(object_id)) or _default_sidecar(object_id)
    sidecar["archived"] = True
    _write_json_atomic(_meta_path(object_id), sidecar)
    return SidecarOut(**sidecar)


# --- SPEC-HEKB-REFACTOR-2026-v2.0 Step 5: post-hoc recalibration API -------
# New: no `/trajectories/*` route existed in this service before this change.

from index.recalibrate import get_job, new_job_id, run_recalibration_job  # noqa: E402


def _trajectories_root() -> Path:
    return _data_dir() / "trajectories"


def _index_db_path() -> Path:
    return _data_dir() / "indexes" / "hekb_vnext_index.db"


class RecalibrateRequest(BaseModel):
    """[OBSERVED DIFFERENCE, disclosed]: SPEC-HEKB-REFACTOR-2026-v2.0
    section 4.2's own worked example nests these fields under a
    `quantizer_config` object. This task's own acceptance-criteria wording
    ("リクエストパラメータ: codebook_id, k_target (デフォルト: 5), dry_run")
    asks for them flat instead -- flattened here per that explicit request.
    `n_pca`/`gamma` are still accepted (the quantizer needs them) but with
    defaults, so a minimal {codebook_id, dry_run} request also works."""
    codebook_id: str
    k_target: int = Field(5, gt=0)
    n_pca: int = Field(32, gt=0)
    gamma: float = Field(2.0, gt=1.0)
    dry_run: bool = True


class RecalibrateAccepted(BaseModel):
    job_id: str
    status: str
    estimated_trajectories: int


@app.post("/trajectories/recalibrate", response_model=RecalibrateAccepted, status_code=202)
async def recalibrate_trajectories(body: RecalibrateRequest) -> RecalibrateAccepted:
    import asyncio

    from index.recalibrate import iter_stored_trajectory_ids

    trajectories_root = _trajectories_root()
    estimated = len(iter_stored_trajectory_ids(trajectories_root))
    job_id = new_job_id()

    asyncio.create_task(run_recalibration_job(
        job_id=job_id,
        trajectories_root=trajectories_root,
        index_db_path=_index_db_path(),
        n_pca=body.n_pca,
        gamma=body.gamma,
        k_target=body.k_target,
        codebook_id=body.codebook_id,
        dry_run=body.dry_run,
    ))
    return RecalibrateAccepted(job_id=job_id, status="PROCESSING", estimated_trajectories=estimated)


@app.get("/trajectories/recalibrate/{job_id}")
def get_recalibration_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"unknown job_id {job_id!r}")
    if job.status == "FAILED":
        raise HTTPException(status_code=500, detail=f"recalibration job {job_id} failed: {job.error}")
    if job.status == "PROCESSING":
        return {"job_id": job_id, "status": "PROCESSING"}
    return job.result


def main() -> None:
    import asyncio

    import uvicorn

    from auth.uds_guard import UDSGuardConfig, serve_uds_guard

    parser = argparse.ArgumentParser(description="HEKB vNext Experience Store")
    parser.add_argument("--host", default=os.environ.get("HEKB_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HEKB_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--data-dir", default=os.environ.get("HEKB_DATA_DIR", DEFAULT_DATA_DIR))
    parser.add_argument("--app-support-dir", default=os.environ.get("HEKB_APP_SUPPORT_DIR", DEFAULT_APP_SUPPORT_DIR))
    parser.add_argument("--no-uds", action="store_true", help="Skip starting the hekb.sock UDS guard listener.")
    args = parser.parse_args()

    os.environ["HEKB_DATA_DIR"] = args.data_dir

    app_support_dir = Path(args.app_support_dir)
    token_path = app_support_dir / "hekb.token"
    token = generate_bearer_token(token_path)
    _bearer_guard.token = token
    print(f"[hekb-vnext] TCP loopback Bearer token written to {token_path} (mode 0600)")

    async def _run() -> None:
        config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
        server = uvicorn.Server(config)
        tasks = [asyncio.create_task(server.serve())]

        if not args.no_uds:
            socket_path = app_support_dir / "hekb.sock"
            guard_config = UDSGuardConfig(
                socket_path=socket_path, backend_host=args.host, backend_port=args.port,
            )
            await serve_uds_guard(guard_config)
            print(f"[hekb-vnext] UDS guard listening on {socket_path} (mode 0600, LOCAL_PEERCRED-authorized)")

        await asyncio.gather(*tasks)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
