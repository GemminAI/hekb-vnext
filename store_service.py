"""HEKB vNext Experience Store — FastAPI wrapper (My_LLM_Builder_App_SPEC.md T7).

Thin HTTP wrapper around the vendored HEKB vNext categorical core
(`vendor/hekb_vnext/{core,memory}`, copied verbatim from
`/Users/tomonam3/Projects/experiments/cs336/hekb_vnext`, which itself has no
persistence or HTTP layer of its own). This module owns everything that
package does not: content-addressed on-disk storage under
`<data_dir>/{objects,relations,lineages}/`, the FastAPI routes, and process
startup (host/port/data-dir).

Lineage model: a "lineage" is this service's own bookkeeping concept, not
part of the vendored core (which defines Object/Morphism/Category only).
Posting an experience with no `parent_id` starts a new lineage rooted at
that experience's own content-addressed object_id; posting with a
`parent_id` calls the vendored `branch()` to record a dom->cod morphism
from parent to child and joins the parent's lineage. This mirrors the
vendored `adapter/hex_bridge.py`'s own "linear, traceable provenance chain"
pattern (root -> step -> step -> ...).

Content addressing is delegated entirely to `vendor.hekb_vnext.core.object.
MemoryObject`: an experience's `object_id` is a SHA-256 hash of its own
payload's canonical bytes, so identical payloads always resolve to the same
object_id regardless of when or how many times they are posted (POST
/experience is idempotent for a given (payload, parent_id) pair).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "hekb_vnext"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from core.category import Category  # noqa: E402
from core.object import MemoryObject  # noqa: E402
from memory.branching import branch  # noqa: E402

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8300
DEFAULT_DATA_DIR = "./experience"


def _data_dir() -> Path:
    return Path(os.environ.get("HEKB_DATA_DIR", DEFAULT_DATA_DIR)).resolve()


def _objects_dir() -> Path:
    return _data_dir() / "objects"


def _relations_dir() -> Path:
    return _data_dir() / "relations"


def _lineages_dir() -> Path:
    return _data_dir() / "lineages"


def _ensure_layout() -> None:
    for d in (_objects_dir(), _relations_dir(), _lineages_dir()):
        d.mkdir(parents=True, exist_ok=True)


def _object_path(object_id: str) -> Path:
    return _objects_dir() / f"{object_id}.json"


def _relation_path(morphism_id: str) -> Path:
    return _relations_dir() / f"{morphism_id}.json"


def _lineage_path(lineage_id: str) -> Path:
    return _lineages_dir() / f"{lineage_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


class ExperienceIn(BaseModel):
    payload: Any = Field(..., description="Content-addressed experience payload (any JSON-compatible value).")
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
    payload: Any
    created_at: str


class RecallOut(BaseModel):
    lineage_id: str | None
    count: int
    experiences: list[ExperienceOut]


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _ensure_layout()
    yield


app = FastAPI(
    title="hekb-vnext",
    description="HEKB vNext Experience Store — FastAPI wrapper around the vendored HEKB vNext core (T7).",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hekb-vnext"}


@app.post("/experience", response_model=ExperienceOut, status_code=201)
def create_experience(body: ExperienceIn) -> ExperienceOut:
    _ensure_layout()

    object_id = MemoryObject(payload=body.payload).object_id
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
        edge = branch(Category(), parent_object, new_payload=body.payload, op=body.op, context=body.context)
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
        "payload": body.payload,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _write_json_atomic(_object_path(object_id), record)

    lineage = _read_json(_lineage_path(lineage_id)) or {"lineage_id": lineage_id, "object_ids": []}
    lineage["object_ids"].append(object_id)
    _write_json_atomic(_lineage_path(lineage_id), lineage)

    return ExperienceOut(**record)


@app.get("/experience/recall", response_model=RecallOut)
def recall_experience(
    lineage_id: str | None = Query(None, description="Return the chain of experiences for this lineage."),
    object_id: str | None = Query(None, description="Return exactly this experience, by content-addressed id."),
    limit: int = Query(50, ge=1, le=1000, description="Max experiences to return, most recent first."),
) -> RecallOut:
    _ensure_layout()

    if object_id is not None:
        record = _read_json(_object_path(object_id))
        if record is None:
            raise HTTPException(status_code=404, detail=f"object_id {object_id!r} not found")
        return RecallOut(lineage_id=record["lineage_id"], count=1, experiences=[ExperienceOut(**record)])

    if lineage_id is not None:
        lineage = _read_json(_lineage_path(lineage_id))
        if lineage is None:
            raise HTTPException(status_code=404, detail=f"lineage_id {lineage_id!r} not found")
        ids = list(reversed(lineage["object_ids"]))[:limit]
        records = [ExperienceOut(**_read_json(_object_path(oid))) for oid in ids]
        return RecallOut(lineage_id=lineage_id, count=len(records), experiences=records)

    paths = sorted(_objects_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    records = [ExperienceOut(**_read_json(p)) for p in paths]
    return RecallOut(lineage_id=None, count=len(records), experiences=records)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="HEKB vNext Experience Store")
    parser.add_argument("--host", default=os.environ.get("HEKB_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.environ.get("HEKB_PORT", str(DEFAULT_PORT))))
    parser.add_argument("--data-dir", default=os.environ.get("HEKB_DATA_DIR", DEFAULT_DATA_DIR))
    args = parser.parse_args()

    os.environ["HEKB_DATA_DIR"] = args.data_dir
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
