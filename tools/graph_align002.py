#!/usr/bin/env python3
# ruff: noqa: E501
"""GRAPH-ALIGN001 follow-up: (1) freeze the Phase 2 artifacts by sha256, (2) record the connection checks,
(3) token containment Verifications in a COPY of the graph (graph_data_v2), (4) causal-align-token/2 records
with a diff against the Phase 2 (v1) records. Existing artifacts are read only; outputs are new files/dirs
under GRAPH-ALIGN001/.

    python3 tools/graph_align002.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.alignment_v2 import diff_v1_v2, to_v2  # noqa: E402
from graph.containment import containing_documents, intervals, range_verification  # noqa: E402
from graph.hext_projection import HextProjectionLayer  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from graph.verification import write_verification  # noqa: E402

EV = Path.home() / "vaults/20260124/experiment-evidence"
A2 = EV / "CAUSAL-ALIGN002"
G1 = EV / "GRAPH-ALIGN001"
GEN = EV / "HEXT-V2-GEN001"
DATA = Path("/Volumes/SSD1TB/HEKBv3/exp_hekbv3_causal48_v2")


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # ---- 1. freeze manifest -----------------------------------------------------------------------------
    files = sorted(
        [p for p in A2.glob("*") if p.is_file()]
        + [p for p in G1.glob("*") if p.is_file() and not p.name.startswith("phase2_") and p.name not in ("phase2_freeze_manifest.json",)]
        + [p for p in (G1 / "graph_data").rglob("*") if p.is_file()]
        + [p for p in GEN.rglob("*") if p.is_file()]
    )
    manifest = {"note": "sha256 of the artifacts the Phase 2 connection checks and the v2 extension were built on; "
                        "nothing listed here was modified",
                "files": {str(p.relative_to(EV)): sha_file(p) for p in files}}
    (G1 / "phase2_freeze_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))

    # ---- inputs -----------------------------------------------------------------------------------------
    layer = HextProjectionLayer(G1 / "graph_data")
    pnodes = {n["node_id"]: n for n in layer.nodes()}
    pedges = layer.edges()
    ivs = intervals(pedges)
    doc_meta = {k: {"state_hash": v["attrs"]["state_hash"], "hext_object_id": v["attrs"]["hext_object_id"],
                    "blob_sha256": v["attrs"]["blob_sha256"]} for k, v in pnodes.items() if v["node_type"] == "hext_document"}
    objs = {}
    for f in (DATA / "objects").glob("*.json"):
        o = json.loads(f.read_text())
        objs[o["object_id"]] = o
    ref_edges = [e for e in pedges if e["type"] == "observation_ref"]
    ver_resolve: dict[tuple[str, int], dict[str, Any]] = {}
    for line in (G1 / "verifications.jsonl").read_text().splitlines():
        v = json.loads(line)
        if v["rule"]["id"] == "R-VER-REF-RESOLVE-01":
            ver_resolve[(v["detail"]["session"], v["detail"]["token_index"])] = v

    # ---- 3. containment verifications into a COPY of the graph ------------------------------------------
    v2dir = G1 / "graph_data_v2"
    if v2dir.exists():
        shutil.rmtree(v2dir)
    shutil.copytree(G1 / "graph_data", v2dir)
    store = GraphStore(v2dir, v01=False)
    containment_status: Counter[str] = Counter()
    ver_ids: dict[tuple[str, str], str] = {}
    for e in sorted(ref_edges, key=lambda x: (x["src"], x["dst"], x["attrs"]["ref"])):
        traj = objs[e["dst"]]
        ext = traj["payload"]["ext"]["hekbv3"]
        ver = range_verification(
            doc_node=e["src"], trajectory=e["dst"], n_steps=ext["causal_log"]["n_steps"],
            seq_from=e["attrs"]["seq_from"], seq_to=e["attrs"]["seq_to"], ref=e["attrs"]["ref"],
            doc_sha256=doc_meta[e["src"]]["blob_sha256"], causal_log_sha256=ext["causal_log"]["blob_sha256"])
        write_verification(store, ver, known_ids=set(pnodes))
        containment_status[ver["status"]] += 1
        ver_ids[(e["src"], e["dst"])] = ver["verification_id"]
    store.sync()

    # ---- 4. v2 token records and the diff -----------------------------------------------------------------
    out_path = G1 / "phase2_alignments_v2.jsonl"
    n = 0
    diffs: list[dict[str, Any]] = []
    cover: Counter[int] = Counter()
    status_same = Counter()
    with (A2 / "phase2_new_alignments.jsonl").open() as src, out_path.open("w") as dst:
        next(src)  # meta line of the Phase 2 file
        dst.write(json.dumps({"_meta": {"schema": "causal-align-token/2", "from": "CAUSAL-ALIGN002/phase2_new_alignments.jsonl",
                                        "changes": "hext_ref -> hext10_reference; parties_present renamed; hext_v2_documents, "
                                                   "schema, alignment_id_v1 added; alignment_id recomputed"}}, sort_keys=True) + "\n")
        for line in src:
            v1 = json.loads(line)
            traj = v1["object_id"]
            cont = containing_documents(ivs, traj, v1["token_index"])
            rv = ver_resolve[(v1["session"], v1["token_index"])]
            v2 = to_v2(v1, containing=cont, resolution=rv["status"], resolution_verification=rv["verification_id"],
                       doc_meta=doc_meta)
            for c in v2["hext_v2_documents"]:
                c["range_verification"] = ver_ids.get((c["hextdoc"], traj))
            bad = diff_v1_v2(v1, v2)
            if bad and len(diffs) < 20:
                diffs.append({"session": v1["session"], "token": v1["token_index"], "diff": bad})
            cover[len(cont)] += 1
            status_same["identity_equal"] += int(v1["identity_status"] == v2["identity_status"])
            status_same["temporal_equal"] += int(v1["temporal_status"] == v2["temporal_status"])
            status_same["semantic_equal"] += int(v1["semantic_status"] == v2["semantic_status"])
            dst.write(json.dumps(v2, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    # ---- 2. connection report ----------------------------------------------------------------------------
    corr = {r["session_id"]: r for r in json.loads((GEN / "correspondence.json").read_text())}
    conn: Counter[str] = Counter()
    by_sess = {e["attrs"]["session_id"]: e for e in ref_edges}
    for sid, r in corr.items():
        d = pnodes.get("hextdoc:" + r["state_hash"])
        conn["doc_node_exists"] += int(d is not None)
        conn["blob_sha256_equal"] += int(bool(d) and d["attrs"]["blob_sha256"] == r["blob_sha256"])
        conn["hekb_object_id_equal"] += int(bool(d) and d["attrs"]["hekb_object_id"] == r["hekb_object_id_v2"])
        conn["obsref_dst_is_v2_object"] += int(by_sess[sid]["dst"] == r["hekb_object_id_v2"])
    report = {
        "phase2_connection_checks_over_sessions": dict(conn),
        "token_records": n, "diff_vs_v1_unexpected": len(diffs), "diff_examples": diffs,
        "status_unchanged": dict(status_same),
        "tokens_by_number_of_containing_documents": {str(k): v for k, v in sorted(cover.items())},
        "containment_verifications": dict(containment_status),
        "v2_graph": {"nodes": len(store.nodes_log), "edges": len(store.edges_log),
                     "heads": {"nodes": store.nodes_log.head, "edges": store.edges_log.head}},
        "v2_records_sha256": sha_file(out_path),
    }
    (G1 / "phase2_connection_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("token_records", "diff_vs_v1_unexpected", "status_unchanged",
                                             "tokens_by_number_of_containing_documents", "containment_verifications")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
