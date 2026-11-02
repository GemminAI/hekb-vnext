"""``GraphStore``: nodes and edges of one data dir, as two append-only hash-chained logs.

Nothing is ever rewritten or removed (D-18): a correction is a new node or edge. Adding the same
node (same ``node_id`` and the same record) or the same edge (same type, src, dst, attrs) again is
a no-op. Writing a node under an existing ``node_id`` with a different record raises
``graph.guard.CollisionError``; every write goes through ``graph.guard.put`` (one lock per data dir).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from graph import axis, ctp
from graph.chainlog import ChainLog
from graph.guard import GraphLock, Ledger, PeerNodes, edge_key, node_key, put
from graph.kindmap import check_stored_kind, d70_class
from graph.model import validate_edge, validate_node


class GraphStore:
    def __init__(self, data_dir: Path, *, v01: bool = True, blob_dir: Path | None = None) -> None:
        self.root = Path(data_dir) / "graph"
        self.v01 = v01
        #: where the blobs are (the dir that holds ``blobs/aa/bb/<sha256>``); a ``step_output`` is refused without it
        self.blob_dir = Path(blob_dir) if blob_dir is not None else None
        self._lock = GraphLock.of(self.root)
        self._types: dict[str, str] = {}  # node_id -> node_type, derived from the node log
        self._axis = axis.AxisIndex()  # axis_verification records and who replaced whom, derived from the logs
        self._recs: dict[str, dict[str, Any]] = {}  # CTP records and HEXT documents of the main log, for rule lookups
        with self._lock:
            self.nodes_log = ChainLog(self.root / "nodes.jsonl")
            self.edges_log = ChainLog(self.root / "edges.jsonl")
            self._nodes = Ledger(self.nodes_log, node_key, "node", self._observe_node)
            self._edges = Ledger(self.edges_log, edge_key, "edge", self._axis.observe_edge)
        # the HEXT projection layer's own node log: same data dir, same ids (D11)
        self._layer_nodes = PeerNodes(self.root / "hext_projection" / "nodes.jsonl")

    def _observe_node(self, record: dict[str, Any]) -> None:
        self._types[record["node_id"]] = record["node_type"]
        self._axis.observe_node(record)
        if record["node_type"] in ctp.CTP_NODE_TYPES or record["node_type"] == "hext_document":
            self._recs[record["node_id"]] = record

    def _node_type_of(self, node_id: str) -> str | None:
        self._nodes.refresh()
        return self._types.get(node_id) or self._layer_nodes.node_type(node_id)

    def _record_hash_of(self, node_id: str) -> str | None:
        self._nodes.refresh()
        return self._nodes.lookup(node_id) or self._layer_nodes.lookup(node_id)

    # the lookups the CTP rules use (graph/ctp.py Context); the main log first, then the projection layer
    def node_type(self, node_id: str) -> str | None:
        return self._node_type_of(node_id)

    def attrs(self, node_id: str) -> dict[str, Any] | None:
        self._nodes.refresh()
        rec = self._recs.get(node_id)
        return rec["attrs"] if rec is not None else self._layer_nodes.attrs(node_id)

    def record_hash(self, node_id: str) -> str | None:
        return self._record_hash_of(node_id)

    def blob_root(self) -> Path | None:
        return self.blob_dir

    def _check_node(self, record: dict[str, Any]) -> None:
        check_stored_kind(record)
        if record["node_type"] == axis.NODE_TYPE:
            axis.check_node(record, self._record_hash_of)
        if record["node_type"] in ctp.CTP_NODE_TYPES:
            if self.v01:
                raise ctp.CtpError(f"{record['node_type']}: CTP records are written with GraphStore(v01=False)")
            ctp.check_node(record, self)

    def _check_edge(self, record: dict[str, Any]) -> None:
        self._nodes.refresh()
        axis.check_edge(record, self._axis, self._node_type_of)
        ctp.check_edge(record, self)

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
        return put(self._lock, self._nodes, node, peers=[self._layer_nodes], check=self._check_node)

    def add_edge(self, type_: str, src: str, dst: str, attrs: dict[str, Any] | None = None) -> bool:
        edge = {"type": type_, "src": src, "dst": dst, "attrs": attrs or {}}
        validate_edge(edge)
        return put(self._lock, self._edges, edge, check=self._check_edge)

    def nodes(self) -> list[dict[str, Any]]:
        return list(self.nodes_log.records())

    def edges(self) -> list[dict[str, Any]]:
        return list(self.edges_log.records())

    def sync(self) -> None:
        self.nodes_log.sync()
        self.edges_log.sync()

    # ---- axis_verification (graph/axis.py) ------------------------------------------------------------
    def write_axis_verification(
        self,
        *,
        axis_name: str,
        value: str,
        method: Any,
        target: str,
        producer: dict[str, str],
        detail_ref: str | None = None,
    ) -> str:
        """Record one axis' check of ``target`` (a stored node) and tie it to the target. Returns the node_id.

        Writes the node and its ``verifies`` edge; both are idempotent, so repeating the call is harmless.
        Needs ``GraphStore(v01=False)``: the v0.1 mode refuses every kind but observation.
        """
        with self._lock:
            target_hash = self._record_hash_of(target)
            if target_hash is None:
                raise axis.AxisVerificationError(f"target {target!r} is not a stored node")
            target_class = d70_class({"node_id": target, "node_type": self._node_type_of(target), "kind": None})
            if target_class not in axis.VERIFIABLE_CLASSES:
                raise axis.AxisVerificationError(f"target {target!r} is classified {target_class!r}; only {axis.VERIFIABLE_CLASSES} can be verified")
            node = axis.build_node(
                axis=axis_name, value=value, method=method, detail_ref=detail_ref,
                target=target, target_record_hash=target_hash, producer=producer,
            )
            self.add_node(node["node_id"], node["node_type"], kind=node["kind"], verification=node["verification"], attrs=node["attrs"])
            self.add_edge("verifies", node["node_id"], target)
            return str(node["node_id"])

    def replace_axis_verification(self, new_id: str, old_id: str, *, correction: bool = False) -> bool:
        """``new_id`` supersedes (a newer method or evaluator) or, with ``correction``, corrects ``old_id``."""
        return self.add_edge("corrects" if correction else "supersedes", new_id, old_id)

    def axis_state(self, target: str, axis_name: str) -> axis.AxisState:
        """The state of (target, axis): ``state`` (a value, ``pending`` or ``conflict``), ``basis`` (``no_record`` or ``recorded``) and ``records`` (graph/axis.py)."""
        with self._lock:
            self._nodes.refresh()
            self._edges.refresh()
            return axis.current_state(self._axis, target, axis_name)

    def axis_consistency(self) -> axis.ConsistencyReport:
        """axis_verification records without their ``verifies`` edge, and records on a replacement cycle."""
        with self._lock:
            self._nodes.refresh()
            self._edges.refresh()
            return axis.consistency_report(self._axis)

    # ---- CTP episode records (graph/ctp.py) ------------------------------------------------------------
    def build_ctp_node(
        self,
        node_type: str,
        fields: dict[str, Any],
        *,
        producer: dict[str, str],
        input_nodes: tuple[str, ...] = (),
        input_facts: dict[str, Any] | None = None,
        refs: dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> dict[str, Any]:
        """The node record for a CTP type; its input nodes must be stored (their record hashes go into input_hash)."""
        with self._lock:
            return ctp.build_node(
                node_type, fields, producer=producer, hash_of=self._record_hash_of,
                input_nodes=input_nodes, input_facts=input_facts, refs=refs, observed_at=observed_at,
            )

    def write_ctp_node(self, node_type: str, fields: dict[str, Any], **kwargs: Any) -> str:
        """Build and write one CTP node (needs ``GraphStore(v01=False)``). Returns its node_id; repeating is idempotent."""
        with self._lock:
            node = self.build_ctp_node(node_type, fields, **kwargs)
            self.add_node(node["node_id"], node["node_type"], kind=node["kind"], verification=node["verification"], attrs=node["attrs"])
            return str(node["node_id"])

    def find_outcomes(
        self,
        *,
        two_axis: str | tuple[str, ...] | list[str] | None = None,
        bundle_id: str | tuple[str, ...] | list[str] | None = None,
        violation_type: str | tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Outcome records by two_axis, the episode's bundle_id and violation_type (any combination; a tuple is "any of")."""
        with self._lock:
            self._nodes.refresh()
            return ctp.find_outcomes(self._recs, two_axis=two_axis, bundle_id=bundle_id, violation_type=violation_type)

    def nondeterminism(self) -> list[ctp.NondeterminismGroup]:
        """Judgements with the same identity, inputs and evaluator but another result (reported, never refused)."""
        with self._lock:
            self._nodes.refresh()
            return ctp.nondeterministic_groups(self._recs)
