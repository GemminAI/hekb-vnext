"""Q2: token value -> causal log -> source text -> HEXT, in one call."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from graph import catalog
from graph.blobs import read_blob
from graph.store import GraphStore

def _flat(x: Any) -> list[float]:
    if isinstance(x, bool):
        return [float(x)]
    if isinstance(x, (int, float)):
        return [float(x)]
    if isinstance(x, list):
        return [v for e in x for v in _flat(e)]
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in _flat(x[k])]
    return []


def _l2(values: list[float]) -> float:
    return float(math.sqrt(sum(v * v for v in values)))


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if len(values) else None


#: port -> scalar of its result. The same definition as ``nvs_edge.board.MAP`` (the table PORT38-
#: TRACE001 used), written out here because this layer must not import the kernel. A port that is
#: not in the table, or whose rule fails, falls back to the L2 norm of every number in the result.
SCALAR_OF: dict[str, Callable[[dict[str, Any]], Any]] = {
    "P01": lambda r: r["context_distance"],
    "P02": lambda r: r["top_eigenvalue"],
    "P03": lambda r: r["mean_step"],
    "P04": lambda r: r["residual"],
    "P06": lambda r: r["reduced_magnitude"],
    "P07": lambda r: r["reduced"],
    "P08": lambda r: r["accumulated_state"],
    "P09": lambda r: r["decay_variance"],
    "P10": lambda r: r["accumulated_shift"],
    "P13": lambda r: r["variance_of_reductions"],
    "P14": lambda r: r["projection"],
    "P15": lambda r: None,
    "P16": lambda r: r["force_magnitude"],
    "P18": lambda r: r["v_norm"],
    "P19": lambda r: r["distance"],
    "P20": lambda r: r["aggregate"],
    "P21": lambda r: r["distance"],
    "P22": lambda r: r["integrated_state"],
    "P23": lambda r: r["gated"],
    "P24": lambda r: r["eigenvector"][0],
    "P25": lambda r: r["nearest_distance"],
    "P26": lambda r: _mean(r["sample_magnitudes"]),
    "P27": lambda r: r["hole_count"],
    "P28": lambda r: _l2(r["mean_vector"]),
    "P29": lambda r: r["binned"],
    "P30": lambda r: r["tracked"],
    "P31": lambda r: r["reduced"],
    "P32": lambda r: r["summary"],
    "P33": lambda r: r["smallest_eigenvalue"],
    "P35": lambda r: _mean(r["alignment"]),
    "P36": lambda r: _mean(r["convolved"]),
    "P38": lambda r: r["top_magnitude"],
}


def port_scalar(port: str, result: dict[str, Any] | None) -> float:
    if result is None:
        return float("nan")
    try:
        if port in SCALAR_OF:
            v = SCALAR_OF[port](result)
            return float("nan") if v is None else float(v)
    except Exception:
        pass
    flat = _flat(result)
    return float(np.linalg.norm(flat)) if flat else float("nan")


class GraphQuery:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self.store = GraphStore(self.data_dir)
        self.catalog = catalog.load(self.store, self.data_dir)
        self._nodes = {n["node_id"]: n for n in self.store.nodes()}
        self._blob_cache: dict[str, Any] = {}
        self._out: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for e in self.store.edges():
            self._out.setdefault((e["src"], e["type"]), []).append(e)

    def _json_blob(self, sha: str) -> Any:
        if sha not in self._blob_cache:
            if len(self._blob_cache) >= 4:
                self._blob_cache.pop(next(iter(self._blob_cache)))
            self._blob_cache[sha] = json.loads(read_blob(self.data_dir, sha))
        return self._blob_cache[sha]

    def trajectory_of(self, session: str | None = None, object_id: str | None = None) -> str:
        if object_id is not None:
            return object_id
        if session is None or session not in self.catalog["by_session"]:
            raise KeyError(f"session {session!r} is not in the graph")
        return str(self.catalog["by_session"][session])

    def q2(
        self, *, session: str | None = None, object_id: str | None = None, token: int, port: str = "P18"
    ) -> dict[str, Any]:
        traj = self.trajectory_of(session, object_id)
        cl_edge = self._out[(traj, "has_causal_log")][0]
        st_edge = self._out[(traj, "has_source_text")][0]
        ps_edge = self._out[(traj, "measured_under")][0]
        records = self._json_blob(cl_edge["dst"].removeprefix("blob:"))
        st = self._json_blob(st_edge["dst"].removeprefix("blob:"))
        if not 0 <= token < len(records):
            raise IndexError(f"token {token} out of range 0..{len(records) - 1}")
        record = records[token]
        hext_edges = [
            e for e in self._out[(cl_edge["dst"], "token_hext")] if e["attrs"]["token_index"] == token
        ]
        hext = self._nodes[hext_edges[0]["dst"]]
        c0, c1 = st["token_char_ranges"][token]
        obs = record["ports"][port]
        return {
            "trajectory": traj,
            "session_id": self._nodes[traj]["attrs"]["session_id"],
            "token_index": token,
            "port": {
                "id": port,
                "scalar": port_scalar(port, obs.get("result")),
                "observation": obs,
            },
            "causal_log": {
                "node": cl_edge["dst"],
                "record": record,
                "ports": record["ports"],
                "cycle_raw": record["cycle_raw"],
            },
            "source_text": {
                "node": st_edge["dst"],
                "token_id": st["token_ids"][token],
                "char_range": [c0, c1],
                "byte_range": st["token_byte_ranges"][token],
                "text": st["text"][c0:c1],
            },
            "hext": {
                "node": hext["node_id"],
                "object_id": hext["node_id"].removeprefix("hext:"),
                "schema_version": hext["attrs"].get("schema_version"),
                "risk": hext["attrs"].get("risk"),
                "tier": hext["attrs"].get("tier"),
                "verification": hext["verification"],
            },
            "policy_set": {"node": ps_edge["dst"], **self._nodes[ps_edge["dst"]]["attrs"]},
            "provenance": {
                "kinds": {
                    "trajectory": self._nodes[traj]["verification"],
                    "causal_log": self._nodes[cl_edge["dst"]]["verification"],
                    "source_text": self._nodes[st_edge["dst"]]["verification"],
                    "hext": hext["verification"],
                },
                "edges": [cl_edge["type"], st_edge["type"], "token_hext", ps_edge["type"]],
            },
        }

    def find_by_value(
        self, port: str, value: float, *, session: str, rel_tol: float = 1e-9
    ) -> list[int]:
        traj = self.trajectory_of(session)
        cl = self._out[(traj, "has_causal_log")][0]["dst"].removeprefix("blob:")
        records = self._json_blob(cl)
        out = []
        for t, r in enumerate(records):
            v = port_scalar(port, r["ports"][port].get("result"))
            if not math.isnan(v) and math.isclose(v, value, rel_tol=rel_tol, abs_tol=1e-12):
                out.append(t)
        return out
