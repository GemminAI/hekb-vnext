"""MSD0001 v1.1 reading for stage 1. Locked sessions are excluded BY ID before anything is opened."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MSD = Path("/Volumes/SSD1TB/EXP-MSD0001 v1.1")


def load_records(msd: Path = MSD) -> dict[str, Any]:
    assignment: dict[str, str] = json.loads((msd / "splits.json").read_text())["assignment"]
    observed = json.loads((msd / "observed_labels.json").read_text())
    locked = sorted(s for s, a in assignment.items() if str(a).endswith("_locked"))
    recs: list[dict[str, Any]] = []
    system_prompt = None
    for session, split in sorted(assignment.items()):
        if str(split).endswith("_locked"):
            continue  # not opened
        meta = json.loads((msd / "sessions" / session / "meta.json").read_text())
        task = meta.get("task")
        if system_prompt is None:
            system_prompt = meta["system_prompt"]
        if not task:
            continue
        verdict = ((observed.get(session) or {}).get("machine") or {}).get("task") or {}
        recs.append(
            {
                "session": session, "split": split, "side": "fit" if split == "benign_fit" else "dev",
                "prompt_id": meta["prompt_id"], "prompt_sha256": meta["prompt_sha256"],
                "kind": task["kind"], "difficulty": task["difficulty"], "params": task["params"],
                "prompt": task["prompt"], "expected": task["expected"],
                "verdict_correct": verdict.get("correct"), "extracted": verdict.get("extracted"),
                "n_steps": meta["n_steps"],
            }
        )
    return {"records": recs, "locked_count": len(locked), "system_prompt": system_prompt, "assignment": assignment}
