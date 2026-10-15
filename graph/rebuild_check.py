"""Baseline and comparison for a generated artifact (graph_data_v2): what is compared before and after a rebuild.

Verdicts (``compare``):
  REBUILD_EQUAL              inputs identical AND every output file byte-identical (so chain heads, counts and
                             all records are identical)
  INPUTS_CHANGED             an input differs from the baseline: no statement is made about the outputs
                             (they are expected to differ); the changed inputs are listed
  OUTPUT_DIFFERS             inputs identical but an output differs: non-determinism or a code/rule change; the
                             differing files are listed, and the build identity (code commit, rule sha256s) is
                             reported next to the baseline's so a code change can be told from non-determinism
  OUTPUT_FILES_MISMATCH      the set of output files differs (missing or extra)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "graph-data-v2-baseline/1"
CRITERIA = {
    "C1_inputs": "sha256 of every input file equals the baseline (graph_data logs and projection, "
                 "CAUSAL-ALIGN002/phase2_new_alignments.jsonl, verifications.jsonl, HEXT-V2-GEN001 correspondence and docs)",
    "C2_build_identity": "recorded, not required equal: tool git commit, rule registry sha256s, python, PYTHONHASHSEED. "
                         "Used only to explain a C3 difference",
    "C3_outputs": "the set of output files and the sha256 of each (graph_data_v2 logs, projection logs, manifest, "
                  "phase2_alignments_v2.jsonl, phase2_connection_report.json) are equal; byte equality implies equal "
                  "chain heads, counts and records",
    "C4_invariants": "graph_data_v2 nodes/edges logs begin with the unmodified graph_data logs (append-only); every "
                     "v2 token record differs from v1 only by the declared renames/additions; every token is "
                     "contained by exactly one document",
}


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digests(files: dict[str, Path]) -> dict[str, str]:
    return {k: sha_file(p) for k, p in sorted(files.items())}


def digest_of(d: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def compare(baseline: dict[str, Any], inputs: dict[str, str], outputs: dict[str, str], build: dict[str, Any]) -> dict[str, Any]:
    changed = sorted(k for k in set(baseline["inputs"]) | set(inputs) if baseline["inputs"].get(k) != inputs.get(k))
    res: dict[str, Any] = {"build_baseline": baseline["build"], "build_now": build}
    if changed:
        return {**res, "verdict": "INPUTS_CHANGED", "changed_inputs": changed[:50], "n_changed_inputs": len(changed)}
    if set(baseline["outputs"]) != set(outputs):
        return {**res, "verdict": "OUTPUT_FILES_MISMATCH",
                "missing": sorted(set(baseline["outputs"]) - set(outputs)),
                "extra": sorted(set(outputs) - set(baseline["outputs"]))}
    diff = sorted(k for k in outputs if outputs[k] != baseline["outputs"][k])
    if diff:
        same_build = baseline["build"].get("git_commit") == build.get("git_commit") and \
            baseline["build"].get("rules") == build.get("rules")
        return {**res, "verdict": "OUTPUT_DIFFERS", "differing_files": diff,
                "same_code_and_rules": same_build,
                "reading": "non-determinism" if same_build else "code or rule change (not necessarily a defect)"}
    return {**res, "verdict": "REBUILD_EQUAL", "files_compared": len(outputs)}
