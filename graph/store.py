"""``GraphStore``: nodes and edges of one data dir, as two append-only hash-chained logs.

Nothing is ever rewritten or removed (D-18): a correction is a new node or edge. Adding the same
node (same ``node_id`` and the same record) or the same edge (same type, src, dst, attrs) again is
a no-op. Writing a node under an existing ``node_id`` with a different record raises
``graph.guard.CollisionError``; every write goes through ``graph.guard.put`` (one lock per data dir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graph.chainlog import ChainLog
from graph.guard import GraphLock, Ledger, PeerNodes, edge_key, node_key, put
from graph.model import validate_edge, validate_node


class GraphStore:
    def __init__(self, data_dir: Path, *, v01: bool = True) -> None:
        self.root = Path(data_dir) / "graph"
        self.v01 = v01
        self._lock = GraphLock.of(self.root)
        with self._lock:
            self.nodes_log = ChainLog(self.root / "nodes.jsonl")
            self.edges_log = ChainLog(self.root / "edges.jsonl")
            self._nodes = Ledger(self.nodes_log, node_key, "node")
            self._edges = Ledger(self.edges_log, edge_key, "edge")
        # the HEXT projection layer's own node log: same data dir, same ids (D11)
        self._layer_nodes = PeerNodes(self.root / "hext_projection" / "nodes.jsonl")

    def has_node(self, node_id: str) -> bool:
        with self._lock:
            self._nodes.refresh()
            return self._nodes.lookup(node_id) is not None

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
        return put(self._lock, self._nodes, node, peers=[self._layer_nodes])

    def add_edge(self, type_: str, src: str, dst: str, attrs: dict[str, Any] | None = None) -> bool:
        edge = {"type": type_, "src": src, "dst": dst, "attrs": attrs or {}}
        validate_edge(edge)
        return put(self._lock, self._edges, edge)

    def nodes(self) -> list[dict[str, Any]]:
        return list(self.nodes_log.records())

    def edges(self) -> list[dict[str, Any]]:
        return list(self.edges_log.records())

    def sync(self) -> None:
        self.nodes_log.sync()
        self.edges_log.sync()
