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

## API

| Endpoint | Description |
|---|---|
| `GET /health` | `{"status": "ok", "service": "hekb-vnext"}` |
| `POST /experience` | Persist one experience; returns its content-addressed `object_id` and `lineage_id`. |
| `GET /experience/recall?lineage_id=&limit=` | Ordered chain of experiences in a lineage, most recent first. |
| `GET /experience/recall?object_id=` | Exact experience by id. |
| `GET /experience/recall?limit=` | Most recently written experiences across all lineages (no filter). |
| `GET /openapi.json` | FastAPI auto-generated schema. |

### `POST /experience` request body

```json
{
  "payload": { "any": "JSON-compatible value" },
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
  "payload": { "any": "JSON-compatible value" },
  "created_at": "2026-09-04T12:00:00Z"
}
```

## On-disk layout

Persisted under `HEKB_DATA_DIR` (default `./experience`, mounted from the
host via `docker-compose.yml`):

```
experience/
├── objects/<object_id>.json      # content-addressed experience record (payload + lineage bookkeeping)
├── relations/<morphism_id>.json  # one vendored MemoryMorphism per parent->child branch edge
└── lineages/<lineage_id>.json    # {"lineage_id": ..., "object_ids": [root, ..., latest]}
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

```bash
curl -s http://127.0.0.1:8300/health

ROOT=$(curl -s -X POST http://127.0.0.1:8300/experience \
  -H 'Content-Type: application/json' \
  -d '{"payload": {"kind": "smoke_test", "text": "hello"}}')
echo "$ROOT"
LINEAGE_ID=$(echo "$ROOT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["lineage_id"])')

curl -s "http://127.0.0.1:8300/experience/recall?lineage_id=$LINEAGE_ID&limit=10"
```
