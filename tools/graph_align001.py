#!/usr/bin/env python3
# ruff: noqa: E501
"""GRAPH-ALIGN001: a read-only prototype that adds Verification records to the existing graph layer for the 48
sessions of exp_hekbv3_causal48_v2 (CAUSAL-ALIGN002). Nothing in the source data dirs, the raw logs or the HEXT
documents is written; the graph is written to a NEW data dir under the output folder.

    python3 tools/graph_align001.py --out DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path.home() / "GemminAI/hext/reference/python"))

from hext2.errors import HextError  # noqa: E402
from hext2.verify import verify as hext2_verify  # noqa: E402

from graph.hext_projection import HextProjectionLayer  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from graph.verification import make_verification, unresolved_reverse_index, write_verification  # noqa: E402

ROOT = Path("/Volumes/SSD1TB/HEKBv3")
DATA = ROOT / "exp_hekbv3_causal48_v2"
RAW = ROOT / "trj003_48/results_vm"
EV = Path.home() / "vaults/20260124/experiment-evidence"
GEN = EV / "HEXT-V2-GEN001"


def tree_digest(root: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
            n += 1
    return h.hexdigest(), n


def build(out_graph_dir: Path) -> dict[str, Any]:
    store = GraphStore(out_graph_dir, v01=False)
    objs = {}
    for f in (DATA / "objects").glob("*.json"):
        o = json.loads(f.read_text())
        objs[o["payload"]["title"]] = (f, o)
    corr = {r["session_id"]: r for r in json.loads((GEN / "correspondence.json").read_text())}
    sessions = sorted(objs)
    # scope of the reference-resolution rule (stated in every UNRESOLVED record)
    scope_ids: set[str] = set()
    for sid in sessions:
        _f, o = objs[sid]
        ext = o["payload"]["ext"]["hekbv3"]
        scope_ids |= {o["object_id"], o.get("lineage_id") or "", ext["causal_log"]["blob_sha256"], ext["ports"]["blob_sha256"]}
        c = corr[sid]
        scope_ids |= {c["document_object_id"], c["state_hash"], c["state_hash"].removeprefix("sha256:"), c["blob_sha256"]}
    scope = {"stored_objects": len(sessions), "hext_v2_documents": len(corr), "ids_in_scope": len(scope_ids),
             "kinds": "object_id, lineage_id, causal/ports blob sha256, HEXT v2 object.id/state_hash/blob sha256"}
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    # ---- HEXT v2 documents -> graph projection (derived layer, separate logs) -------------------------
    docs_in: list[tuple[bytes, dict[str, Any]]] = []
    for sid in sessions:
        b = (GEN / "docs" / f"{sid}.hext2.json").read_bytes()
        try:
            lvl = hext2_verify(b, level=2).level
        except HextError:
            lvl = -1
        docs_in.append((b, {"hekb_object_id": objs[sid][1]["object_id"], "validated_level": lvl, "epistemic_status": "derived"}))
    session_to_traj = {sid: objs[sid][1]["object_id"] for sid in sessions}
    layer = HextProjectionLayer(out_graph_dir)
    proj = layer.rebuild(docs_in, resolve_session=session_to_traj.get)
    proj_ids = {n["node_id"] for n in layer.nodes()}

    def emit(ver: dict[str, Any]) -> None:
        write_verification(store, ver, known_ids=proj_ids)
        records.append(ver)
        counts[f"{ver['rule']['id']}:{ver['status']}"] += 1

    for sid in sessions:
        f, o = objs[sid]
        oid = o["object_id"]
        ext = o["payload"]["ext"]["hekbv3"]
        cl_sha = ext["causal_log"]["blob_sha256"]
        cl_bytes = (DATA / "blobs" / cl_sha[:2] / cl_sha[2:4] / cl_sha).read_bytes()
        recs = json.loads(cl_bytes)
        n = len(recs)
        raw_path = next(RAW.glob(f"sessions/*_{sid}.jsonl.gz"))
        import gzip

        raw_bytes = raw_path.read_bytes()
        raw_sha = hashlib.sha256(raw_bytes).hexdigest()
        lines = [ln for ln in gzip.decompress(raw_bytes).decode("utf-8").splitlines() if ln]
        c = corr[sid]
        doc_bytes = (GEN / "docs" / f"{sid}.hext2.json").read_bytes()
        doc = json.loads(doc_bytes)
        doc_sha = hashlib.sha256(doc_bytes).hexdigest()
        # ---- nodes: observed (as stored) and derived (the generated HEXT document) -------------------
        store.add_node(oid, "trajectory", kind="observation", verification="unverified",
                       attrs={"session_id": sid, "space_id": ext["space"]["space_id"], "n_steps": n, "epistemic_status": "observed"})
        cl_node, raw_node, doc_node = f"blob:{cl_sha}", f"raw:{raw_sha}", f"hextdoc:{c['state_hash']}"
        store.add_node(cl_node, "causal_log", kind="observation", verification="unverified",
                       attrs={"n_steps": n, "schema": ext["causal_log"].get("schema"), "epistemic_status": "observed"})
        store.add_node(raw_node, "raw_log", kind="observation", verification="unverified",
                       attrs={"session_id": sid, "file": raw_path.name, "epistemic_status": "observed"})
        assert doc_node in proj_ids  # the document node comes from the projection layer, not from this store
        # ---- R-VER-RAW-01 --------------------------------------------------------------------------------
        ok_file = raw_sha == ext["causal_log"]["source_file"]["sha256"]
        mism = 0
        for i, (rec, line) in enumerate(zip(recs, lines, strict=True), start=1):
            ref = (rec.get("provenance") or {}).get("raw_ref") or {}
            if ref.get("file_sha256") != raw_sha or ref.get("line_number") != i or ref.get("record_sha256") != hashlib.sha256(line.encode()).hexdigest():
                mism += 1
        emit(make_verification(
            rule_id="R-VER-RAW-01", subject_a=cl_node, subject_b=raw_node,
            status="VERIFIED" if ok_file and mism == 0 and len(lines) == n else "FAILED",
            verified_means=f"raw file sha256 equals stored source_file sha256; {n - mism}/{n} records: raw-line sha256 and line number equal provenance.raw_ref",
            inputs=[{"role": "raw_log", "sha256": raw_sha}, {"role": "causal_log", "sha256": cl_sha}],
            detail={"session": sid, "records": n, "mismatched": mism, "raw_lines": len(lines)},
            reason=None if ok_file and mism == 0 else f"{mism} mismatched records, file sha equal={ok_file}"))
        # ---- R-VER-DOC-TRAJ-01 --------------------------------------------------------------------------
        try:
            level = hext2_verify(doc_bytes, level=2).level
            lv_err = None
        except HextError as exc:
            level, lv_err = -1, exc.code
        m = doc.get("measurement", {})
        want_ref = f"urn:gmm:nvs:session:{sid}#seq=0-{n - 1}"
        same = (m.get("observation_refs") == [want_ref]) and m.get("space_id") == ext["space"]["space_id"] and level == 2
        emit(make_verification(
            rule_id="R-VER-DOC-TRAJ-01", subject_a=doc_node, subject_b=oid,
            status="VERIFIED" if same else "FAILED",
            verified_means="session id and token range in observation_refs, and space_id, equal the stored trajectory's; document verified at level 2 by the reference verifier",
            inputs=[{"role": "hext_document", "sha256": doc_sha}, {"role": "causal_log", "sha256": cl_sha}],
            detail={"session": sid, "observation_refs": m.get("observation_refs"), "document_level": level},
            reason=None if same else f"mismatch or level {level} ({lv_err})"))
        # ---- R-VER-DOC-CL-01 ----------------------------------------------------------------------------
        has_ref = "causal_log_ref" in m
        emit(make_verification(
            rule_id="R-VER-DOC-CL-01", subject_a=doc_node, subject_b=cl_node,
            status="NOT_EVALUATED" if not has_ref else "FAILED",
            verified_means="", inputs=[{"role": "hext_document", "sha256": doc_sha}, {"role": "causal_log", "sha256": cl_sha}],
            detail={"session": sid},
            reason="the document has no measurement.causal_log_ref (omitted: no session-level definition; per-token BoardRecord refs do not equal BLAKE2b of the stored cycle_provenance)" if not has_ref
            else "rule not implemented for documents that carry causal_log_ref"))
        # ---- per-token hext/1.0 references ---------------------------------------------------------------
        for rec in recs:
            t = rec["token_index"]
            h = (rec["cycle_raw"].get("hext") or {}).get("object_id")
            hid = f"hext:{h}"
            store.add_node(hid, "hext_kernel", kind="observation", verification="unverified",
                           attrs={"schema_version": (rec["cycle_raw"].get("hext") or {}).get("schema_version"),
                                  "source": "cycle_raw.hext (reference only)", "epistemic_status": "observed"})
            cp = ((rec.get("provenance") or {}).get("cycle_provenance") or {}).get("value") or [{}]
            root = ((rec.get("ports") or {}).get("P01") or {}).get("raw", {}).get("parent_event_id")
            ident = h is not None and h == root and h == cp[0].get("hext_object_id")
            emit(make_verification(
                rule_id="R-VER-ID-HEXT10-01", subject_a=cl_node, subject_b=hid,
                status="VERIFIED" if ident else "FAILED",
                verified_means="hext object id equals the P01 chain root and the cycle_provenance hext id of the same record (identifier match only)",
                inputs=[{"role": "causal_log", "sha256": cl_sha}],
                detail={"session": sid, "token_index": t, "target_id": h},
                reason=None if ident else "identifier mismatch"))
            resolved = h in scope_ids
            emit(make_verification(
                rule_id="R-VER-REF-RESOLVE-01", subject_a=cl_node, subject_b=hid,
                status="VERIFIED" if resolved else "UNRESOLVED",
                verified_means="the id equals a stored document identifier within the search scope" if resolved else "",
                inputs=[{"role": "causal_log", "sha256": cl_sha}],
                detail={"session": sid, "token_index": t, "target_id": h, "search_scope": scope},
                reason=None if resolved else "no stored document with this id could be confirmed in the search scope (not a statement that none exists)"))
    # unresolved observation_refs: no edge was made; the failure is a Verification record in the main store
    for u in layer.unresolved():
        if u["kind"] == "observation_ref":
            emit(make_verification(
                rule_id="R-VER-OBSREF-RESOLVE-01", subject_a=u["document"], subject_b=None, status="UNRESOLVED",
                verified_means="", inputs=[{"role": "hext_document", "sha256": u["document"].split(":")[-1]}],
                detail={"ref": u["ref"], "session": u.get("session_id")}, reason=u["reason"]))
    store.sync()
    return {"store": store, "records": records, "counts": dict(counts), "scope": scope, "sessions": len(sessions),
            "projection": proj, "layer": layer, "docs_in": docs_in, "session_to_traj": session_to_traj}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EV / "GRAPH-ALIGN001")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    before = tree_digest(DATA)
    gdir = out / "graph_data"
    if (gdir / "graph" / "nodes.jsonl").exists():
        shutil.rmtree(gdir)  # our own previous output only
    res = build(gdir)
    store: GraphStore = res["store"]
    rev = unresolved_reverse_index(store)
    (out / "verifications.jsonl").write_text("\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in res["records"]) + "\n")
    (out / "unresolved_reverse_index.json").write_text(json.dumps(rev, sort_keys=True, indent=0))
    # reproducibility: build twice more into scratch dirs and compare chain heads
    heads = []
    for k in (1, 2):
        d = out / f"_repro{k}"
        r = build(d)
        heads.append((r["store"].nodes_log.head, r["store"].edges_log.head))
        shutil.rmtree(d)
    # rebuild the projection from the same documents: the HEXT-derived part is identical and the
    # Verification records of the main store are untouched
    layer: HextProjectionLayer = res["layer"]
    verifs_before = sorted(n["node_id"] for n in store.nodes() if n["node_type"] == "verification")
    heads_before = layer.manifest()["heads"]
    layer.rebuild(res["docs_in"], resolve_session=res["session_to_traj"].get)
    verifs_after = sorted(n["node_id"] for n in GraphStore(gdir, v01=False).nodes() if n["node_type"] == "verification")
    rebuild_check = {"projection_heads_equal": layer.manifest()["heads"] == heads_before,
                     "verification_records_before": len(verifs_before), "verification_records_after": len(verifs_after),
                     "verification_ids_equal": verifs_before == verifs_after}
    after = tree_digest(DATA)
    rep = {
        "sessions": res["sessions"], "verification_records": len(res["records"]), "counts_by_rule_status": res["counts"],
        "search_scope": res["scope"],
        "graph": {"nodes": len(store.nodes_log), "edges": len(store.edges_log),
                  "heads": {"nodes": store.nodes_log.head, "edges": store.edges_log.head}},
        "reproducible_heads_equal": heads[0] == heads[1] == (store.nodes_log.head, store.edges_log.head),
        "unresolved_targets": len(rev), "unresolved_references": sum(len(v) for v in rev.values()),
        "source_dir_digest": {"before": before[0], "after": after[0], "files": before[1], "unchanged": before == after},
        "hext_projection": {k: res["projection"][k] for k in ("documents", "nodes", "edges", "unresolved", "heads")},
        "projection_rebuild_check": rebuild_check,
        "note": "Verification relates existing records; it asserts neither sameness of meaning nor causality.",
    }
    (out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("sessions", "verification_records", "counts_by_rule_status", "unresolved_references", "reproducible_heads_equal")}, indent=1))
    print("source unchanged:", rep["source_dir_digest"]["unchanged"], rep["graph"]["nodes"], rep["graph"]["edges"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
