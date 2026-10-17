#!/usr/bin/env python3
"""EXP-D71-ORACLE001 harness (PREREG.md + PREREG_AMEND1.md). Preparation only; generation runs on the Pod after Tomo's OK.

Only the three observations of the context differ from stage 3: the user text of every target is the rendered,
sha256-pinned text produced on the Mac by ``select_amend1/select_amend1.py`` (summarize_v1, system prompt, sampling
and the seed formula ``stage3.seed_for(prompt_id, version, sample)`` are the stage-3 ones). The Mac packs those
texts into a bundle (``prep``); the Pod never builds contexts.

Modes (PREREG §8 order; each gate refuses to run if the previous one has not passed):
  prep      Mac: pack select_amend1/ into a bundle (targets_block{1,2}.jsonl + manifest.json). Refuses if the leak scan has residuals.
  dry-run   no GPU, no generation: re-verify the bundle (sha256 of every user text, seeds, leak rules, counts) and write the design.
  bridge    Pod: regenerate the stage-3 B / C' rows with idx % 50 == 0 (same input and seed) and compare token ids.
  block1    Pod: confirmatory block (684 generations). Needs b0 PASS and bridge PASS.
  block2    Pod: exploratory block (759). Needs block1 finished.
  resample  Pod: idx % 50 == 0 second generation for a block (mismatches recorded, never stop).
  manifest  sha256 of every file under --out.
D71-B0 is ``python -m d71.stage3 b0`` (unchanged). Stop conditions: stage 3's (run_generations) plus the gates above.
Credentials are never read here except through the environment by the loader (HF_TOKEN); values are never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from d71 import stage3

SAMPLES = stage3.SAMPLES
ORDER_SEED = stage3.ORDER_SEED  # block randomisation within an item, same generator as stage 3
EXPECTED_TARGETS = {"block1": 684, "block2": 759}  # PREREG_AMEND1 §4 (confirmatory 76 items x 3 arms x R=3)
BRIDGE_CONDS = ("B", "Cprime")
BRIDGE_EVERY = 50
EXIT_GATE, EXIT_BRIDGE = 10, 9
BUNDLE_FIELDS = ("arm", "version", "prompt_id", "sample", "sources", "obs_tokens", "obs_tokens_Cprime", "context_tokens",
                 "context_sha256", "user")


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---- leak rules (leak_rules.md (a)(b)(c)); third implementation, applied to the packed texts ------------------
_PIECES = re.compile(r'\("(.*?)"\)')
_INTS = re.compile(r"(?<![\w.])\d+(?![\w.])")
_WEEK = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def leak_hits(text: str, expected: str, removed: str, kind: str) -> list[str]:
    body = text.split("\n", 1)[1]
    pieces = [p.strip() for p in _PIECES.findall(text)]
    hits: set[str] = set()
    for tgt in (expected, removed):
        d = re.sub(r"\D", "", tgt)
        if len(d) >= 2 and any(re.search(r"\d", p) and any(d[i:i + 2] in p for i in range(len(d) - 1)) for p in pieces):
            hits.add("a")
        if tgt.isdigit() and tgt in _INTS.findall(body):
            hits.add("b")
    if kind == "day_of_week" and any(w in p.lower() or w[:3] in p.lower() for p in pieces for w in _WEEK):
        hits.add("c")
    return sorted(hits)


# ---- bundle (Mac) -------------------------------------------------------------------------------------------
def build_bundle(select_dir: Path, contexts_all: Path, out: Path, system_prompt: Path) -> dict[str, Any]:
    scan = _read_json(select_dir / "leak_scan.json")
    if scan["n_residual"] != 0:
        raise SystemExit(f"STOP: leak scan has {scan['n_residual']} residual observations")
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    counts: dict[str, int] = {}
    meta: dict[str, dict[str, Any]] = {}
    for block in ("block1", "block2"):
        sel = _read_json(select_dir / f"selection_{block}.json")
        if sel["unbuildable_targets"]:
            raise SystemExit(f"STOP: {block} has unbuildable targets")
        rows = []
        for t in sel["targets"]:
            name = f"{t['arm']}__{t['prompt_id']}__{t['version']}__s{t['sample']}.json"
            rec = _read_json(select_dir / "contexts" / block / name)
            pid = t["prompt_id"]
            if pid not in meta:
                meta[pid] = _read_json(contexts_all / pid / "context_set.json")
            cs = meta[pid]
            rows.append({**{k: rec[k] for k in BUNDLE_FIELDS}, "block": block, "kind": cs["kind"],
                         "difficulty": cs["difficulty"], "removed": (cs["deficient"] or {}).get("removed"),
                         "expected": cs["expected"] if rec["version"] == "sufficient" else None,
                         "leak_E": str(cs["expected"]), "leak_R": str((cs["deficient"] or {}).get("removed_literal", "")),
                         "obs_texts": rec["obs_texts"]})
        rows.sort(key=lambda r: (r["prompt_id"], r["version"], r["arm"], int(r["sample"])))
        path = out / f"targets_{block}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[path.name] = _sha_file(path)
        counts[block] = len(rows)
    sp = out / "system_prompt.txt"
    shutil.copyfile(system_prompt, sp)
    files[sp.name] = _sha_file(sp)
    manifest = {"files": files, "counts": counts, "leak_scanned_targets": scan["scanned_targets"], "leak_n_residual": 0,
                "excluded_cells": _read_json(select_dir / "excluded_cells.json"),
                "selection_sha256": {f"selection_{b}.json": _sha_file(select_dir / f"selection_{b}.json")
                                     for b in ("block1", "block2")},
                "version_fingerprint": _version_fingerprint(), "created_utc": datetime.now(timezone.utc).isoformat()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def _version_fingerprint() -> dict[str, str]:
    from d71.summarize import version_fingerprint

    return version_fingerprint()


def load_bundle(bundle: Path) -> dict[str, list[dict[str, Any]]]:
    """Verifies every pinned file and every user text. Any mismatch is a stop condition (context sha256, leak)."""
    man = _read_json(bundle / "manifest.json")
    if man["leak_n_residual"] != 0:
        raise RuntimeError("STOP: bundle manifest records leak residuals")
    for name, want in man["files"].items():
        if _sha_file(bundle / name) != want:
            raise RuntimeError(f"STOP: bundle file differs from manifest: {name}")
    out: dict[str, list[dict[str, Any]]] = {}
    for block in ("block1", "block2"):
        rows = [json.loads(ln) for ln in (bundle / f"targets_{block}.jsonl").read_text().splitlines() if ln.strip()]
        for r in rows:
            if stage3.sha(r["user"]) != r["context_sha256"]:
                raise RuntimeError(f"STOP: context sha256 mismatch: {r['arm']} {r['prompt_id']} {r['version']} s{r['sample']}")
            hits = [h for t in r["obs_texts"] for h in leak_hits(t, r["leak_E"], r["leak_R"], r["kind"])]
            if hits:
                raise RuntimeError(f"STOP: leak residual in {r['arm']} {r['prompt_id']} {r['version']} s{r['sample']}: {hits}")
        out[block] = rows
    return out


def build_design(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One shuffled block per item (both versions, all arms, all samples), items in sha256 order; stage-3 style."""
    rng = random.Random(ORDER_SEED)
    by_item: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_item.setdefault(r["prompt_id"], []).append(r)
    design: list[dict[str, Any]] = []
    for pid in sorted(by_item, key=lambda p: stage3.sha("ORACLE001|order|" + p)):
        block = [{
            "prompt_id": pid, "kind": r["kind"], "difficulty": r["difficulty"], "version": r["version"],
            "cond": r["arm"], "sample": int(r["sample"]), "seed": stage3.seed_for(pid, r["version"], int(r["sample"])),
            "user": r["user"], "context_sha256": r["context_sha256"], "context_tokens": r["context_tokens"],
            "sources": r["sources"], "removed": r["removed"], "expected": r["expected"],
        } for r in sorted(by_item[pid], key=lambda x: (x["version"], x["arm"], int(x["sample"])))]
        rng.shuffle(block)
        design.extend(block)
    for i, d in enumerate(design):
        d["idx"] = i
    return design


def dry_run(bundle: Path, out: Path | None = None) -> dict[str, Any]:
    blocks = load_bundle(bundle)
    summary: dict[str, Any] = {"ok": True, "version_fingerprint": _version_fingerprint()}
    for block, rows in blocks.items():
        design = build_design(rows)
        if len(design) != EXPECTED_TARGETS[block]:
            raise RuntimeError(f"STOP: {block} has {len(design)} targets, expected {EXPECTED_TARGETS[block]}")
        keys = {(d["cond"], d["prompt_id"], d["version"], d["sample"]) for d in design}
        if len(keys) != len(design):
            raise RuntimeError(f"STOP: duplicate targets in {block}")
        arms: dict[str, int] = {}
        for d in design:
            arms[f"{d['cond']}:{d['version']}"] = arms.get(f"{d['cond']}:{d['version']}", 0) + 1
        summary[block] = {"n": len(design), "items": len({d["prompt_id"] for d in design}), "arms": arms,
                          "resample_rows": sum(1 for d in design if d["idx"] % stage3.RESAMPLE_EVERY == 0),
                          "design_sha256": stage3.sha(json.dumps([{k: v for k, v in d.items() if k != "user"} for d in design]))}
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "dry_run.json").write_text(json.dumps(summary, indent=1))
    return summary


# ---- gates and bridge ---------------------------------------------------------------------------------------
def gate_ok(gates: Path, name: str) -> bool:
    p = gates / name
    try:
        if name == "b0":
            return bool(_read_json(p / "b0_result.json")["pass"])
        if name == "bridge":
            return bool(_read_json(p / "bridge_result.json")["pass"])
        return _read_json(p / "run_status.json")["code"] == 0  # block1 / block2
    except (OSError, KeyError, ValueError):
        return False


REQUIRES = {"block1": ("b0", "bridge"), "block2": ("b0", "bridge", "block1")}


def bridge_design(contexts: dict[str, dict[str, Any]], selection: dict[str, Any]) -> list[dict[str, Any]]:
    design = stage3.build_design(selection["main"], contexts, conds=stage3.CONDS, samples=SAMPLES)
    return [d for d in design if d["idx"] % BRIDGE_EVERY == 0 and d["cond"] in BRIDGE_CONDS]


def run_bridge(design: list[dict[str, Any]], stage3_gens: Path, provider: Any, out: Path, system_prompt: str) -> dict[str, Any]:
    stored = {}
    for ln in stage3_gens.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)
            stored[r["idx"]] = r
    out.mkdir(parents=True, exist_ok=True)
    mism: list[int] = []
    rows = []
    for d in design:
        s = stored.get(d["idx"])
        if s is None or (s["prompt_id"], s["cond"], s["version"], s["sample"]) != (d["prompt_id"], d["cond"], d["version"], d["sample"]):
            raise RuntimeError(f"STOP: stage-3 design does not reproduce row {d['idx']}")  # stored run differs from design
        r = provider.generate_chat([{"role": "system", "content": system_prompt}, {"role": "user", "content": d["user"]}],
                                   max_tokens=stage3.MAX_TOKENS, temperature=0.7, top_k=50, top_p=0.95, seed=d["seed"])
        same = bool(r.failure_kind not in ("generate_failed", "load_failed") and r.token_ids == s["token_ids"])
        rows.append({"idx": d["idx"], "prompt_id": d["prompt_id"], "cond": d["cond"], "version": d["version"],
                     "sample": d["sample"], "token_ids_identical": same, "n_stored": len(s["token_ids"]),
                     "n_new": len(r.token_ids), "warnings": provider.last_warnings, "error": r.error})
        if not same:
            mism.append(d["idx"])
    (out / "bridge_rows.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    res = {"pass": not mism, "n": len(rows), "n_mismatch": len(mism), "mismatch_idx": mism}
    (out / "bridge_result.json").write_text(json.dumps(res, indent=1))
    return res


def warnings_extra(provider: Any, d: dict[str, Any]) -> dict[str, Any]:
    w = provider.last_warnings or {}
    return {"determinism_warning_count": w.get("count", 0), "determinism_warning_ops": w.get("ops", {})}


def run_block(rows: list[dict[str, Any]], block: str, provider: Any, out: Path, gates: Path, system_prompt: str,
              *, mem_fraction: Any = lambda: 0.0) -> int:
    for g in REQUIRES[block]:
        if not gate_ok(gates, g):
            print(f"STOP: gate '{g}' has not passed ({gates / g})", flush=True)
            return EXIT_GATE
    design = build_design(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "design.json").write_text(json.dumps([{k: v for k, v in d.items() if k != "user"} for d in design]))
    code = stage3.run_generations(design, provider, out, system_prompt, mem_fraction=mem_fraction, row_extra=warnings_extra)
    (out / "run_status.json").write_text(json.dumps({"code": code, "utc": datetime.now(timezone.utc).isoformat()}))
    return code


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over tested functions
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prep", "dry-run", "bridge", "block1", "block2", "resample", "manifest"])
    ap.add_argument("--block", choices=["block1", "block2"])
    ap.add_argument("--bundle", type=Path)
    ap.add_argument("--select-dir", type=Path)
    ap.add_argument("--contexts-all", type=Path)
    ap.add_argument("--system-prompt-file", type=Path)
    ap.add_argument("--stage3-contexts", type=Path)
    ap.add_argument("--stage3-selection", type=Path)
    ap.add_argument("--stage3-generations", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--gates", type=Path, help="directory holding b0/, bridge/, block1/ outputs (default: --out's parent)")
    ap.add_argument("--expect-gpu", default=None)
    a = ap.parse_args(argv)
    if a.mode == "manifest":
        print(json.dumps({"n_files": stage3.write_manifest(a.out)["n_files"]}))
        return 0
    if a.mode == "prep":
        print(json.dumps(build_bundle(a.select_dir, a.contexts_all, a.out, a.system_prompt_file)["counts"]))
        return 0
    if a.mode == "dry-run":
        print(json.dumps(dry_run(a.bundle, a.out), indent=1))
        return 0
    from d71 import hf_provider

    hf_provider.deterministic_env()
    import torch

    settings = hf_provider.deterministic_torch(torch)  # warn_only=True (D-75), counts and op names recorded per generation
    mode = a.block if a.mode == "resample" else a.mode
    blocks = load_bundle(a.bundle) if a.mode != "bridge" else {}
    system_prompt = (a.bundle / "system_prompt.txt").read_text() if a.bundle else a.system_prompt_file.read_text()
    provider = hf_provider.Llama31HFProvider()
    provider._ensure_loaded()
    env = stage3.environment(provider)
    env["deterministic_settings"] = settings
    env["version_fingerprint"] = _version_fingerprint()
    if a.expect_gpu and a.expect_gpu not in str(env["gpu_name"]):
        print(f"STOP: GPU {env['gpu_name']!r} does not match {a.expect_gpu!r}")
        return 7
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_env.json").write_text(json.dumps(env, indent=1))
    gates = a.gates or a.out.parent
    if a.mode == "bridge":
        ids = _read_json(a.stage3_selection)["main"]
        ctx = stage3.load_contexts(a.stage3_contexts, ids)
        res = run_bridge(bridge_design(ctx, _read_json(a.stage3_selection)), a.stage3_generations, provider, a.out, system_prompt)
        print("BRIDGE", json.dumps(res))
        return 0 if res["pass"] else EXIT_BRIDGE
    if a.mode == "resample":
        summ = stage3.run_resample(build_design(blocks[mode]), provider, a.out, system_prompt)
        (a.out / "resample_summary.json").write_text(json.dumps(summ, indent=1))
        print("RESAMPLE", json.dumps(summ))
        return 0
    return run_block(blocks[mode], mode, provider, a.out, gates, system_prompt, mem_fraction=stage3.gpu_mem_fraction)


if __name__ == "__main__":
    raise SystemExit(main())
