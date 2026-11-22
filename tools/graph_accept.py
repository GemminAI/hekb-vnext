#!/usr/bin/env python3
"""HEKB-GRAPH002 acceptance: Q2 results against TRACE001's table (written by an independent route:
hekb HTTP API + a separate decoder), catalog rebuild, and unchanged-source hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import catalog  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from graph.trace import GraphQuery  # noqa: E402

EVID = Path.home() / "vaults/20260124/experiment-evidence"
TRACE = EVID / "PORT38-TRACE001"
PORTS = [f"P{n:02d}" for n in range(1, 39)]


def tree_hash(root: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
            n += 1
    return h.hexdigest(), n


def close(a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9)


def acceptance(data_dir: Path) -> dict[str, Any]:
    q = GraphQuery(data_dir)
    out: dict[str, Any] = {"cases": 0, "tokens": 0, "mismatch": [], "ports_compared": 0}
    with (TRACE / "summary.csv").open() as handle:
        cases = list(csv.DictReader(handle))
    for case in cases:
        with (TRACE / f"trace_{case['case']}.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        out["cases"] += 1
        for row in rows:
            t = int(row["idx"])
            r = q.q2(session=case["session"], token=t, port="P18")
            out["tokens"] += 1
            probs = []
            if r["source_text"]["text"] != row["token_text"] and r["source_text"]["text"].encode(
                "utf-8", "replace"
            ) != row["token_text"].encode("utf-8", "replace"):
                probs.append(("text", r["source_text"]["text"], row["token_text"]))
            if r["hext"]["object_id"][:12] != row["hext_object_id"]:
                probs.append(("hext", r["hext"]["object_id"][:12], row["hext_object_id"]))
            if f"{r['hext']['risk']:.4g}" != row["hext_risk"]:
                probs.append(("risk", r["hext"]["risk"], row["hext_risk"]))
            if r["hext"]["tier"] != row["hext_tier"]:
                probs.append(("tier", r["hext"]["tier"], row["hext_tier"]))
            t_ns = float(r["causal_log"]["record"]["ports"]["P01"]["timestamp_ns"])
            if not close(t_ns / 1e9, float(row["event_time_s"])):
                probs.append(("time", None, row["event_time_s"]))
            for port in PORTS:
                key = f"val_{port}"
                if key not in row:
                    continue
                obs = r["causal_log"]["ports"].get(port)
                if obs is None:
                    continue
                from graph.trace import port_scalar

                got = port_scalar(port, obs.get("result"))
                out["ports_compared"] += 1
                if not close(got, float(row[key])):
                    probs.append((port, got, row[key]))
            if probs:
                out["mismatch"].append({"case": case["case"], "token": t, "problems": probs[:4]})
    out["mismatch_count"] = len(out["mismatch"])
    out["mismatch"] = out["mismatch"][:20]
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, Any] = {}
    result["q2_vs_trace001"] = acceptance(args.data)
    # catalog: delete and rebuild -> same content
    store = GraphStore(args.data)
    before = json.loads(catalog.path_of(args.data).read_text())
    catalog.path_of(args.data).unlink()
    after = catalog.load(store, args.data)
    result["catalog_rebuild_equal"] = before == after
    first = GraphQuery(args.data).q2(session="p11-jailbreak-1629", token=5)
    result["q2_same_after_rebuild"] = first["causal_log"]["record"] == GraphQuery(args.data).q2(
        session="p11-jailbreak-1629", token=5
    )["causal_log"]["record"]
    # existing values unchanged: every copied file equals its source byte for byte
    sel = json.loads(args.selection.read_text())
    same = total = 0
    for s in sel["sessions"]:
        oid = sel["object_id"][s]
        obj = json.loads((args.src / "objects" / f"{oid}.json").read_text())["payload"]["ext"]["hekbv3"]
        files = [f"objects/{oid}.json", f"meta/{oid}.json", f"lineages/{oid}.json"]
        for sha in (obj["causal_log"]["blob_sha256"], obj["ports"]["blob_sha256"]):
            files.append(f"blobs/{sha[:2]}/{sha[2:4]}/{sha}")
        for f in files:
            a, b = args.src / f, args.data / f
            if a.exists():
                total += 1
                same += int(b.exists() and a.read_bytes() == b.read_bytes())
    result["files_identical_to_source"] = {"same": same, "total": total}
    result["source_tree"] = dict(zip(("sha256", "files"), tree_hash(args.src), strict=True))
    args.out.write_text(json.dumps(result, indent=1, default=str))
    brief = dict(result)
    brief["q2_vs_trace001"] = {x: y for x, y in result["q2_vs_trace001"].items() if x != "mismatch"}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
