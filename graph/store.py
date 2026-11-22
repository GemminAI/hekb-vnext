"""``GraphStore``: nodes and edges of one data dir, as two append-only hash-chained logs.

Nothing is ever rewritten or removed (D-18): a correction is a new node or edge. Adding the same
node (same ``node_id``) or the same edge (same type, src, dst, attrs) again is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graph.chainlog import ChainLog, canonical
from graph.model import validate_edge, validate_node


class GraphStore:
    def __init__(self, data_dir: Path, *, v01: bool = True) -> None:
        self.root = Path(data_dir) / "graph"
        self.v01 = v01
        self.nodes_log = ChainLog(self.root / "nodes.jsonl")
        self.edges_log = ChainLog(self.root / "edges.jsonl")
        self._node_ids: set[str] = {r["node_id"] for r in self.nodes_log.records()}
        self._edge_keys: set[str] = {self._edge_key(r) for r in self.edges_log.records()}

    @staticmethod
    def _edge_key(edge: dict[str, Any]) -> str:
        return canonical([edge["type"], edge["src"], edge["dst"], edge.get("attrs", {})])

    def has_node(self, node_id: str) -> bool:
        return node_id in self._node_ids

    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        kind: str = "observation",
        verification: str = "unverified",
        attrs: dict[str, Any] | None = None,
    ) -> bool:
        node = {
            "node_id": node_id,
            "node_type": node_type,
            "kind": kind,
            "verification": verification,
            "attrs": attrs or {},
        }
        validate_node(node, v01=self.v01)
        if node_id in self._node_ids:
            return False
        self.nodes_log.append(node)
        self._node_ids.add(node_id)
        return True

    def add_edge(self, type_: str, src: str, dst: str, attrs: dict[str, Any] | None = None) -> bool:
        edge = {"type": type_, "src": src, "dst": dst, "attrs": attrs or {}}
        validate_edge(edge)
        key = self._edge_key(edge)
        if key in self._edge_keys:
            return False
        self.edges_log.append(edge)
        self._edge_keys.add(key)
        return True

    def nodes(self) -> list[dict[str, Any]]:
        return list(self.nodes_log.records())

    def edges(self) -> list[dict[str, Any]]:
        return list(self.edges_log.records())

    def sync(self) -> None:
        self.nodes_log.sync()
        self.edges_log.sync()
