#!/usr/bin/env python3
# ruff: noqa: E501
"""Record the baseline of GRAPH-ALIGN001/graph_data_v2 and check that it can be rebuilt from its inputs.

    python3 tools/graph_v2_rebuild.py record    # write graph_data_v2_baseline.json from the current artifact
    python3 tools/graph_v2_rebuild.py check     # rebuild into a scratch dir (the artifact is not touched), compare

The scratch rebuild is deleted afterwards; the existing artifact is never overwritten by this tool.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.rebuild_check import CRITERIA, SCHEMA, compare, digest_of, digests  # noqa: E402
from graph.verification import RULES, rule_ref  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EV = Path.home() / "vaults/20260124/experiment-evidence"
G1, A2, GEN = EV / "GRAPH-ALIGN001", EV / "CAUSAL-ALIGN002", EV / "HEXT-V2-GEN001"
BASELINE = G1 / "graph_data_v2_baseline.json"


def input_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for p in (G1 / "graph_data").rglob("*"):
        if p.is_file():
            files[f"GRAPH-ALIGN001/{p.relative_to(G1)}"] = p
    files["GRAPH-ALIGN001/verifications.jsonl"] = G1 / "verifications.jsonl"
    files["CAUSAL-ALIGN002/phase2_new_alignments.jsonl"] = A2 / "phase2_new_alignments.jsonl"
    for p in GEN.rglob("*"):
        if p.is_file() and p.name in ("correspondence.json",) or p.parent.name == "docs":
            files[f"HEXT-V2-GEN001/{p.relative_to(GEN)}"] = p
    return files


def output_files(graph_dir: Path, records: Path, report: Path) -> dict[str, Path]:
    files = {f"graph_data_v2/{p.relative_to(graph_dir)}": p for p in graph_dir.rglob("*") if p.is_file()}
    files["phase2_alignments_v2.jsonl"] = records
    files["phase2_connection_report.json"] = report
    return files


def build_identity() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "graph", "tools"], cwd=REPO,
                                capture_output=True, text=True).stdout.strip())
    return {"tool": "tools/graph_align002.py", "git_commit": commit, "working_tree_dirty": dirty,
            "rules": {i: rule_ref(i)["sha256"] for i in sorted(RULES)}, "python": platform.python_version(),
            "PYTHONHASHSEED": "0", "alignment_schema": "causal-align-token/2"}


def record() -> int:
    inputs = digests(input_files())
    outputs = digests(output_files(G1 / "graph_data_v2", G1 / "phase2_alignments_v2.jsonl", G1 / "phase2_connection_report.json"))
    rep = json.loads((G1 / "phase2_connection_report.json").read_text())
    BASELINE.write_text(json.dumps({
        "schema": SCHEMA, "artifact": "GRAPH-ALIGN001/graph_data_v2 (+ phase2_alignments_v2.jsonl, phase2_connection_report.json)",
        "generated_by": "tools/graph_align002.py (a generated artifact: it can be deleted and rebuilt from the inputs)",
        "build": build_identity(), "inputs": inputs, "inputs_digest": digest_of(inputs),
        "outputs": outputs, "outputs_digest": digest_of(outputs), "heads": rep["v2_graph"]["heads"],
        "counts": {"nodes": rep["v2_graph"]["nodes"], "edges": rep["v2_graph"]["edges"], "token_records": rep["token_records"]},
        "criteria": CRITERIA,
    }, indent=1, sort_keys=True))
    print("baseline recorded:", len(inputs), "inputs,", len(outputs), "outputs")
    return 0


def check() -> int:
    base = json.loads(BASELINE.read_text())
    scratch = G1 / "_rebuild_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir()
    graph, rec, rep = scratch / "graph_data_v2", scratch / "phase2_alignments_v2.jsonl", scratch / "phase2_connection_report.json"
    env = dict(os.environ, PYTHONHASHSEED="0")
    subprocess.run([sys.executable, str(REPO / "tools/graph_align002.py"), "--graph-out", str(graph), "--records-out", str(rec),
                    "--report-out", str(rep), "--no-freeze"], check=True, capture_output=True, env=env, cwd=REPO)
    res = compare(base, digests(input_files()), digests(output_files(graph, rec, rep)), build_identity())
    # the check itself must not have touched the artifact
    res["artifact_untouched"] = digests(output_files(G1 / "graph_data_v2", G1 / "phase2_alignments_v2.jsonl",
                                                     G1 / "phase2_connection_report.json")) == base["outputs"]
    shutil.rmtree(scratch)
    (G1 / "graph_data_v2_rebuild_check.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps({k: res[k] for k in ("verdict", "artifact_untouched") if k in res}))
    return 0 if res["verdict"] == "REBUILD_EQUAL" else 1


if __name__ == "__main__":
    raise SystemExit({"record": record, "check": check}[sys.argv[1]]())
