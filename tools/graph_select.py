#!/usr/bin/env python3
"""HEKB-GRAPH002: record the 48-session selection *before* ingest.

Rule (fixed here, in this file, before any session is imported):
  1. every session of PORT38-TRACE001's summary.csv (37, all ``anomaly_dev``);
  2. +11 non-Locked benign sessions (benign_fit / benign_dev) from the INFO001 regeneration summary,
     category-balanced: group by ``stratum`` in MSD0001 meta.json, strata in sorted order, within a
     stratum the session with the smallest sha256("GRAPH002|" + session_id) not yet taken, one per
     stratum per round, until 11 are taken;
  3. sessions whose split label ends in ``_locked`` are never candidates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

EVID = Path.home() / "vaults/20260124/experiment-evidence"
MSD = Path("/Volumes/SSD1TB/EXP-MSD0001 v1.1")
N_EXTRA = 11


def select() -> dict[str, Any]:
    assignment = json.loads((MSD / "splits.json").read_text())["assignment"]
    objects = json.loads((EVID / "PORT38-INFO001/regeneration_summary.json").read_text())["object_ids"]
    with (EVID / "PORT38-TRACE001/summary.csv").open() as handle:
        trace = [r["session"] for r in csv.DictReader(handle)]
    candidates: dict[str, list[str]] = {}
    for s in sorted(objects):
        if assignment.get(s) in ("benign_fit", "benign_dev"):
            stratum = json.loads((MSD / "sessions" / s / "meta.json").read_text())["stratum"]
            candidates.setdefault(stratum, []).append(s)
    key = lambda s: hashlib.sha256(f"GRAPH002|{s}".encode()).hexdigest()  # noqa: E731
    pools = {k: sorted(v, key=key) for k, v in sorted(candidates.items())}
    extra: list[str] = []
    while len(extra) < N_EXTRA and any(pools.values()):
        for stratum in pools:
            if pools[stratum] and len(extra) < N_EXTRA:
                extra.append(pools[stratum].pop(0))
    sessions = trace + extra
    assert len(sessions) == len(set(sessions)) == 48
    assert not any(str(assignment.get(s, "")).endswith("_locked") for s in sessions)
    return {
        "rule": __doc__,
        "trace001": trace,
        "extras": extra,
        "sessions": sessions,
        "split": {s: assignment[s] for s in sessions},
        "object_id": {s: objects[s] for s in sessions},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=EVID / "HEKB-GRAPH002/selection.json")
    out = parser.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)
    sel = select()
    out.write_text(json.dumps(sel, indent=1, ensure_ascii=False))
    print(len(sel["sessions"]), "sessions ->", out)
