# hekb-vnext — HEKB vNext Experience Store (FastAPI wrapper)

Thin FastAPI/uvicorn HTTP wrapper around the HEKB vNext categorical core,
built per `My_LLM_Builder_App_SPEC.md` §4.1 (engine launch) and §7 T7
(`ExperienceStoreClient`). Intended to be launched as a local child process
(or container) by the macOS app via `MLBAKit`, over `127.0.0.1` only.

## What this is, and isn't

- The vendored code under `vendor/hekb_vnext/{core,memory}` is a **verbatim
  copy** of `/Users/tomonam3/Projects/experiments/cs336/hekb_vnext`'s `core/`
  and `memory/` packages — pure Python, no persistence, no HTTP, no
  third-party dependencies. It defines `MemoryObject` (content-addressed,
  immutable), `MemoryMorphism`/`Category` (registry + composition), and the
  Memory-Layer operations (`branch`, `reconsolidate`, `recall`,
  `provenance_path`) exactly as that source project left them. Nothing in
  `vendor/` is modified.
- `store_service.py` is new: it adds the HTTP layer, on-disk persistence,
  and the *lineage* bookkeeping concept described below. None of that exists
  in the source project, which is an in-memory-only library.

## Lineage model (this service's own design decision)

The vendored core has no notion of "lineage" — only objects, morphisms, and
a `Category` registry. This wrapper defines one, matching the pattern
`adapter/hex_bridge.py` already uses in the source project (a linear chain
of `branch()` calls, root → step → step → ...):

- `POST /experience` with no `parent_id` starts a **new lineage**: the
  experience becomes its own lineage root, and `lineage_id == object_id`.
- `POST /experience` with a `parent_id` calls the vendored `branch()` to
  record a `parent -> child` morphism, and the child joins the parent's
  lineage (`lineage_id` is inherited).
- Because `object_id` is a content hash of the payload alone, re-posting an
  identical `(payload, parent_id)` pair is idempotent and returns the
  existing record. Re-posting the same payload under a *different*
  `parent_id` is rejected with `409 Conflict` — a content-addressed object
  cannot be re-parented, mirroring the vendored
  `provenance.AmbiguousProvenanceError` discipline.

## Three-layer data model

`object_id` is the SHA-256 of **only** the Layer-1 payload's canonical
bytes, so no mutable value may ever live in `payload`:

- **Layer 1 (hashed core)** — `objects/<object_id>.json`'s `payload`: a
  reserved-key structure (`schema`, `raw_input`, `resolved_meaning`,
  `action`, `title`, `tags`, `ext`). Extra top-level keys are rejected —
  put anything else inside `ext`.
- **Layer 2 (lineage bookkeeping)** — `objects/<object_id>.json`'s top
  level: `object_id, lineage_id, parent_id, morphism_id, op, context,
  created_at`.
- **Layer 3 (mutable sidecar)** — `meta/<object_id>.json`: `recall_count`,
  `last_recalled_at`, `archived`, `provenance`. Rebuildable; never affects
  `object_id`.

Lineage is a **tree**, not a linear chain: the same `parent_id` may branch
into multiple children. `GET /experience/recall` and `GET /graph/topology`
both walk the real dom→cod edges (via the vendored `provenance_path`), not
`lineages/<id>.json`'s insertion-order log.

## Governance: `POST /experience` requires a signed request

Every `POST /experience` must carry an `X-Audit-Signature` header — a
P-256 ECDSA signature (DER, base64 or hex) over the payload's own
canonical bytes (the `object_id` preimage) — verified against the public
key at `HEKB_AUDIT_PUBLIC_KEY_PATH` (PEM SPKI or a raw 65-byte X9.63
uncompressed point, the format `SecKeyCopyExternalRepresentation`
produces). **If that env var is unset, unreadable, or the signature is
missing/invalid, every write is rejected with `403` and nothing is written
to `objects/`, `relations/`, `lineages/`, or `meta/` — this fails closed by
default, not open.** See `store_service.py`'s module docstring for why
this is implemented in Python rather than via `mlba-core` (that crate has
no signature/ledger code today). `tools/dev_audit_signer.py` generates a
throwaway dev keypair and signs a payload file for local testing — it is
explicitly **not** the real Secure Enclave key.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "service": "hekb-vnext"}` |
| `POST /experience` | Persist one experience (requires `X-Audit-Signature`); returns its content-addressed `object_id` and `lineage_id`. |
| `GET /experience/recall?object_id=` | Exact experience by id. |
| `GET /experience/recall?lineage_id=&depth=` | Ancestor chain (root→leaf) ending at that lineage's latest leaf, `depth` hops back (default 5). |
| `GET /experience/recall?from_object_id=&depth=` | Ancestor chain ending at a specific node. |
| `GET /experience/recall?limit=` | Debug fallback: most recent writes across all lineages, no filter. |
| `GET /graph/topology?lineage_id=&include_archived=` | Visual-graph nodes/edges (`relation: "lineage_parent"`, parent→child). Pure read — never touches `recall_count`. |
| `POST /experience/{object_id}/recall` | Bumps the sidecar's `recall_count`/`last_recalled_at`. The only endpoint that does. |
| `POST /experience/{object_id}/archive` | Sets the sidecar's `archived` flag. Soft-delete only — no physical delete exists anywhere in this service. |
| `GET /openapi.json` | FastAPI auto-generated schema. |

### `POST /experience` request body

```json
{
  "payload": {
    "schema": "hekb.experience/1",
    "raw_input": "100",
    "resolved_meaning": "100 Fahrenheit",
    "action": "unit_conversion_fahrenheit",
    "title": "100°F scalar unit disambiguation",
    "tags": ["CUMR001"],
    "ext": {}
  },
  "parent_id": null,
  "op": "experience",
  "context": null
}
```

### `POST /experience` response (`201`)

```json
{
  "object_id": "…sha256 hex…",
  "lineage_id": "…sha256 hex…",
  "parent_id": null,
  "morphism_id": null,
  "op": "experience",
  "context": null,
  "payload": { "schema": "hekb.experience/1", "raw_input": "100", "...": "..." },
  "created_at": "2026-09-04T12:00:00Z"
}
```

## On-disk layout

Persisted under `HEKB_DATA_DIR` (default `./experience`, mounted from the
host via `docker-compose.yml`):

```
experience/
├── objects/<object_id>.json      # Layer 1 (payload) + Layer 2 (lineage bookkeeping)
├── relations/<morphism_id>.json  # one vendored MemoryMorphism per parent->child branch edge
├── lineages/<lineage_id>.json    # {"lineage_id": ..., "object_ids": [insertion order, NOT an ancestor chain]}
└── meta/<object_id>.json         # Layer 3: recall_count, last_recalled_at, archived, provenance
```

## Running locally (no Docker)

```bash
pip install -r requirements.txt
python store_service.py --host 127.0.0.1 --port 8300 --data-dir ./experience
```

`--host`/`--port`/`--data-dir` each fall back to the `HEKB_HOST` /
`HEKB_PORT` / `HEKB_DATA_DIR` environment variables, then to
`127.0.0.1` / `8300` / `./experience`.

## Running in Docker

```bash
docker build -t hekb-vnext:latest .
docker compose up -d
curl http://127.0.0.1:8300/health
```

Inside the container the process binds `0.0.0.0:8300` (required for the
Docker bridge network to reach it at all) but `docker-compose.yml` only
publishes it on `127.0.0.1:8300` on the host, preserving the product's
loopback-only network posture (`My_LLM_Builder_App_SPEC.md` §3.3) end to
end. A native (non-Docker) launch by `MLBAKit`/`EngineManager` should keep
`--host 127.0.0.1` (the default).

## Smoke test

`POST /experience` needs a signed request (see Governance above). For
local testing, generate a throwaway dev keypair and point
`HEKB_AUDIT_PUBLIC_KEY_PATH` at its public half before starting the
server:

```bash
python3 tools/dev_audit_signer.py keygen ./devkeys
HEKB_AUDIT_PUBLIC_KEY_PATH=./devkeys/dev_public_key.raw \
  python store_service.py --host 127.0.0.1 --port 8300 --data-dir ./experience
```

Then, in another shell:

```bash
curl -s http://127.0.0.1:8300/health

cat > /tmp/smoke_payload.json <<'EOF'
{"schema": "hekb.experience/1", "raw_input": "hello", "resolved_meaning": null, "action": null, "title": "smoke_test", "tags": [], "ext": {}}
EOF
SIG=$(python3 tools/dev_audit_signer.py sign ./devkeys/dev_private_key.pem /tmp/smoke_payload.json)

ROOT=$(curl -s -X POST http://127.0.0.1:8300/experience \
  -H 'Content-Type: application/json' \
  -H "X-Audit-Signature: $SIG" \
  -d "{\"payload\": $(cat /tmp/smoke_payload.json)}")
echo "$ROOT"
LINEAGE_ID=$(echo "$ROOT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lineage_id"])')

curl -s "http://127.0.0.1:8300/experience/recall?lineage_id=$LINEAGE_ID&depth=5"
curl -s "http://127.0.0.1:8300/graph/topology?lineage_id=$LINEAGE_ID"
```

`./devkeys/` holds a private key — never commit it (it isn't the real
Secure Enclave key, but treat it like a secret anyway).
