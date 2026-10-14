"""Catalog index over the graph: by space_id, policy_set, session and content_id.

Derived data: ``rebuild`` reads only the node/edge logs; the file can be deleted at any time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graph.store import GraphStore


def rebuild(store: GraphStore) -> dict[str, Any]:
    by_space: dict[str, list[str]] = {}
    by_session: dict[str, str] = {}
    by_content: dict[str, list[str]] = {}
    for node in store.nodes():
        if node["node_type"] != "trajectory":
            continue
        a = node["attrs"]
        by_space.setdefault(a["space_id"], []).append(node["node_id"])
        by_session.setdefault(a["session_id"], node["node_id"])
        by_content.setdefault(a["content_id"], []).append(node["node_id"])
    by_policy: dict[str, list[str]] = {}
    for edge in store.edges():
        if edge["type"] == "measured_under":
            by_policy.setdefault(edge["dst"], []).append(edge["src"])
    index = {
        "schema": "hekbv3.graph_catalog/1",
        "graph_heads": {"nodes": store.nodes_log.head, "edges": store.edges_log.head},
        "by_space_id": {k: sorted(v) for k, v in sorted(by_space.items())},
        "by_policy_set": {k: sorted(v) for k, v in sorted(by_policy.items())},
        "by_session": dict(sorted(by_session.items())),
        "by_content_id": {k: sorted(v) for k, v in sorted(by_content.items())},
    }
    return index


def path_of(data_dir: Path) -> Path:
    return Path(data_dir) / "graph" / "catalog.json"


def write(store: GraphStore, data_dir: Path) -> dict[str, Any]:
    index = rebuild(store)
    path_of(data_dir).write_text(json.dumps(index, sort_keys=True, indent=1))
    return index


def load(store: GraphStore, data_dir: Path) -> dict[str, Any]:
    """The stored catalog when it matches the graph heads, otherwise a rebuilt one (written back)."""
    path = path_of(data_dir)
    if path.exists():
        index: dict[str, Any] = json.loads(path.read_text())
        if index.get("graph_heads") == {"nodes": store.nodes_log.head, "edges": store.edges_log.head}:
            return index
    return write(store, data_dir)
