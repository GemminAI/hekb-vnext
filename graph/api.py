"""Read-only HTTP surface of the graph layer (additive; mounted by ``store_service``)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from graph.trace import GraphQuery


def make_router(data_dir: Callable[[], Path]) -> APIRouter:
    router = APIRouter(prefix="/graph", tags=["graph"])

    @router.get("/q2")
    def q2(
        token: int = Query(..., ge=0),
        session: str | None = Query(None),
        object_id: str | None = Query(None),
        port: str = Query("P18", pattern=r"^P[0-9]{2}$"),
    ) -> dict[str, Any]:
        """Port value -> causal-log record -> source text range and token id -> hext/1.0, in one call."""
        if (session is None) == (object_id is None):
            raise HTTPException(status_code=422, detail="give exactly one of session, object_id")
        try:
            return GraphQuery(data_dir()).q2(session=session, object_id=object_id, token=token, port=port)
        except (KeyError, IndexError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @router.get("/catalog")
    def catalog_view(space_id: str | None = None, policy_set: str | None = None) -> dict[str, Any]:
        index = GraphQuery(data_dir()).catalog
        ids: set[str] | None = None
        if space_id is not None:
            ids = set(index["by_space_id"].get(space_id, []))
        if policy_set is not None:
            members = set(index["by_policy_set"].get(policy_set, []))
            ids = members if ids is None else ids & members
        return {"object_ids": sorted(ids) if ids is not None else None, "graph_heads": index["graph_heads"]}

    return router
