#!/usr/bin/env python3
"""EXP-D71-CONTEXT-001 stage 3 (RunPod, CUDA bf16). Preparation harness; NOT executed on the Mac.

Modes (v0.2-B 4.1; every mode needs the Pod gates of that document, which are Tomo's to open):
  select    Mac: choose items from the rebuilt contexts (deterministic) and write selection.json.
  b0        D71-B0: 5 items x 2 runs, same input and seed; token ids must match exactly, layer-32 max abs diff <= 1e-3.
  calibrate 20 calibration items, condition A only (both versions), R=3 -> delta_T source.
  main      N main items x {sufficient, deficient} x {A,B,C,C',D} x R=3.
  manifest  sha256 of every file under --out (before carrying anything off the Pod).

Credentials are never read here except through the environment by the loader (HF_TOKEN); values are never printed.
Stop conditions: GPU memory > 90%, one generation > 10 min, 3 consecutive failed rows, any generate_failed or
non-deterministic-op error (never silenced), context sha256 differs from selection.json, GPU model differs from
--expect-gpu.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONDS = ("A", "B", "C", "Cprime", "D")
SAMPLES = (0, 1, 2)  # R = 3 (v0.2-B 1)
MAX_TOKENS = 768
N_CALIBRATION = 20
N_B0 = 5
ORDER_SEED = 20261015
GPU_MEM_LIMIT = 0.90
GEN_TIMEOUT_S = 600.0
CONSECUTIVE_FAIL_LIMIT = 3
HIDDEN_TOL = 1e-3


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def seed_for(prompt_id: str, version: str, sample: int) -> int:
    """Identical to ``d71.stage2.seed_for`` (stage 2 and stage 3 share the seed values)."""
    return int(hashlib.sha256(f"D71|seed|{prompt_id}|{version}|{sample}".encode()).hexdigest()[:8], 16)


def select_items(
    contexts: dict[str, dict[str, Any]], stage2_items: set[str]
) -> dict[str, list[str]]:
    """Deterministic: no pi_digits, no stage-2 item; order by sha256('D71|s3|'+prompt_id).
    First N_CALIBRATION -> calibration (condition A only); first N_B0 of the main list -> D71-B0 inputs."""
    eligible = sorted(
        (p for p, c in contexts.items() if c["kind"] != "pi_digits" and p not in stage2_items),
        key=lambda p: sha("D71|s3|" + p),
    )
    calibration = eligible[:N_CALIBRATION]
    main = eligible[N_CALIBRATION:]
    return {"calibration": calibration, "main": main, "b0": main[:N_B0]}


def build_design(
    ids: list[str], contexts: dict[str, dict[str, Any]], *, conds: tuple[str, ...], samples: tuple[int, ...]
) -> list[dict[str, Any]]:
    rng = random.Random(ORDER_SEED)
    design: list[dict[str, Any]] = []
    for pid in ids:
        cs = contexts[pid]
        block = []
        for version in ("sufficient", "deficient"):
            for cond in conds:
                c = cs["conditions"][f"{version}:{cond}"]
                for sample in samples:
                    block.append({
                        "prompt_id": pid, "kind": cs["kind"], "difficulty": cs["difficulty"],
                        "version": version, "cond": cond, "sample": sample,
                        "seed": seed_for(pid, version, sample), "user": c["user"],
                        "context_sha256": c["sha256"], "context_tokens": c["tokens"],
                        "removed": (cs["deficient"] or {}).get("removed") if version == "deficient" else None,
                        "expected": cs["expected"] if version == "sufficient" else None,
                    })
        rng.shuffle(block)  # block randomisation within an item (v0.2-B 1 "実行順")
        design.extend(block)
    for i, d in enumerate(design):
        d["idx"] = i
    return design


def load_contexts(root: Path, ids: list[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for p in ids:
        cs = json.loads((root / p / "context_set.json").read_text())
        for key, c in cs["conditions"].items():
            if sha(c["user"]) != c["sha256"]:
                raise RuntimeError(f"context sha256 mismatch: {p} {key}")  # stop condition (4)
        out[p] = cs
    return out


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    for run in (a, b):  # a failed generation is a FAIL, never an (empty == empty) match
        if run.get("ok") is False or not run["token_ids"] or run["hidden"] is None:
            return {"token_ids_identical": False, "hidden_max_abs_diff": None, "pass": False,
                    "reason": "generation_failed", "error": run.get("error")}

    same_ids = a["token_ids"] == b["token_ids"]
    diff: float | None = None
    if a["hidden"] is not None and b["hidden"] is not None and a["hidden"].shape == b["hidden"].shape:
        diff = float(np.max(np.abs(a["hidden"] - b["hidden"]))) if a["hidden"].size else 0.0
    return {"token_ids_identical": same_ids, "hidden_max_abs_diff": diff,
            "pass": bool(same_ids and diff is not None and diff <= HIDDEN_TOL)}


def environment(provider: Any) -> dict[str, Any]:  # pragma: no cover - Pod only
    def run(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    import torch
    import transformers

    return {
        "utc": datetime.now(timezone.utc).isoformat(), "pod_id": os.environ.get("RUNPOD_POD_ID"),
        "hostname": os.uname().nodename, "torch": torch.__version__, "transformers": transformers.__version__,
        "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
        "harness_commit": run(["git", "-C", str(Path(__file__).resolve().parent.parent), "rev-parse", "HEAD"]),
        "backend": provider.describe(),
        "attn_implementation": getattr(getattr(getattr(provider._backend, "model", None), "config", None),
                                       "_attn_implementation", None),
    }


def gpu_mem_fraction() -> float:  # pragma: no cover - Pod only
    import torch

    free, total = torch.cuda.mem_get_info()
    return 1.0 - free / total


def run_generations(
    design: list[dict[str, Any]], provider: Any, out: Path, system_prompt: str,
    *, mem_fraction: Any = lambda: 0.0, save_hidden: bool = True,
) -> int:
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    (out / "hidden").mkdir(exist_ok=True)
    runs = out / "generations.jsonl"
    done = {json.loads(ln)["idx"] for ln in runs.read_text().splitlines() if ln.strip()} if runs.exists() else set()
    consecutive_fail = 0
    for d in design:
        if d["idx"] in done:
            continue
        frac = mem_fraction()
        if frac > GPU_MEM_LIMIT:
            print(f"STOP: GPU memory {frac:.0%}", flush=True)
            return 3
        started = datetime.now(timezone.utc).isoformat()
        r = provider.generate_chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": d["user"]}],
            max_tokens=MAX_TOKENS, temperature=0.7, top_k=50, top_p=0.95, seed=d["seed"],
        )
        ended = datetime.now(timezone.utc).isoformat()
        hidden = provider.last_hidden
        hpath = hsha = None
        if save_hidden and hidden is not None and r.failure_kind not in ("generate_failed", "load_failed"):
            hpath = f"hidden/{d['idx']:06d}.npy"
            np.save(out / hpath, hidden.astype(np.float32))
            hsha = hashlib.sha256((out / hpath).read_bytes()).hexdigest()
        row = {
            **{k: v for k, v in d.items() if k != "user"},
            "text": r.text, "token_ids": r.token_ids, "completion_tokens": r.completion_tokens,
            "finish_reason": r.finish_reason, "failure_kind": r.failure_kind, "error": r.error, "ok": r.ok,
            "generate_seconds": r.generate_seconds, "load_seconds": r.load_seconds,
            "gpu_memory_fraction": round(frac, 4), "started_utc": started, "ended_utc": ended,
            "sampler": r.sampler, "chat_template_sha256": r.chat_template_sha256, "model_ref": r.model_ref,
            "revision": r.revision, "device": r.device, "hidden_file": hpath, "hidden_sha256": hsha,
            "hidden_shape": list(hidden.shape) if hidden is not None else None,
            "logprob_chosen": provider.last_logprob.tolist() if provider.last_logprob is not None else None,
            "entropy": provider.last_entropy.tolist() if provider.last_entropy is not None else None,
        }
        with runs.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        if r.failure_kind in ("generate_failed", "load_failed"):
            print("STOP:", r.failure_kind, r.error, flush=True)  # includes non-deterministic-op errors
            return 4
        consecutive_fail = consecutive_fail + 1 if not r.ok else 0
        if consecutive_fail >= CONSECUTIVE_FAIL_LIMIT:
            print(f"STOP: {consecutive_fail} consecutive failed rows", flush=True)
            return 5
        if r.generate_seconds and r.generate_seconds > GEN_TIMEOUT_S:
            print("STOP: one generation exceeded 10 minutes", flush=True)
            return 6
    print("DONE", flush=True)
    return 0


def write_manifest(out: Path) -> dict[str, Any]:
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(out))] = {"bytes": p.stat().st_size,
                                              "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    m = {"n_files": len(files), "files": files}
    (out / "manifest.json").write_text(json.dumps(m, indent=1))
    return m


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI over tested functions
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "b0", "calibrate", "main", "manifest"])
    ap.add_argument("--contexts", type=Path, help="contexts_all directory (Mac-built, sha256-checked)")
    ap.add_argument("--selection", type=Path)
    ap.add_argument("--stage2-selection", type=Path)
    ap.add_argument("--system-prompt-file", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--expect-gpu", default=None, help="substring the GPU name must contain (e.g. A100)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.mode == "manifest":
        print(json.dumps({"n_files": write_manifest(a.out)["n_files"]}))
        return 0
    if a.mode == "select":
        ctx = {p.name: json.loads((p / "context_set.json").read_text()) for p in sorted(a.contexts.iterdir())}
        s2 = {i["prompt_id"] for i in json.loads(a.stage2_selection.read_text())["items"]}
        picked = select_items(ctx, s2)
        sel: dict[str, Any] = dict(picked)
        sel["rule"] = ("no pi_digits, no stage-2 item; ascending sha256('D71|s3|'+prompt_id); first 20 = "
                       "calibration (condition A only), rest = main; b0 = first 5 of main")
        sel["context_sha256"] = {p: {k: c["sha256"] for k, c in ctx[p]["conditions"].items()}
                                 for p in picked["calibration"] + picked["main"]}
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "selection.json").write_text(json.dumps(sel, indent=1))
        print({k: len(v) for k, v in picked.items()})
        return 0
    from d71 import hf_provider

    hf_provider.deterministic_env()
    import torch

    settings = hf_provider.deterministic_torch(torch)
    sel = json.loads(a.selection.read_text())
    ids = {"b0": sel["b0"], "calibrate": sel["calibration"], "main": sel["main"]}[a.mode]
    ctx = load_contexts(a.contexts, ids)
    for p in ids:  # selection.json pins the sha256 recorded on the Mac
        for k, c in ctx[p]["conditions"].items():
            if sel["context_sha256"][p][k] != c["sha256"]:
                raise RuntimeError(f"context differs from selection.json: {p} {k}")
    system_prompt = a.system_prompt_file.read_text()
    provider = hf_provider.Llama31HFProvider()
    provider._ensure_loaded()
    env = environment(provider)
    env["deterministic_settings"] = settings
    if a.expect_gpu and a.expect_gpu not in str(env["gpu_name"]):
        print(f"STOP: GPU {env['gpu_name']!r} does not match {a.expect_gpu!r}")
        return 7
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "run_env.json").write_text(json.dumps(env, indent=1))
    if a.mode == "b0":
        first: dict[str, dict[str, Any]] = {}
        for d0 in build_design(ids, ctx, conds=("A", "C"), samples=(0,)):
            first.setdefault(d0["prompt_id"], d0)  # one generation per B0 item (5), each run twice
        design = list(first.values())
        results = []
        for d in design:
            runs = []
            for _ in range(2):
                r = provider.generate_chat([{"role": "system", "content": system_prompt},
                                            {"role": "user", "content": d["user"]}], max_tokens=MAX_TOKENS,
                                           seed=d["seed"])
                runs.append({"token_ids": r.token_ids, "hidden": provider.last_hidden, "ok": r.failure_kind
                             not in ("generate_failed", "load_failed"), "error": r.error,
                             "warnings": provider.last_warnings})
            results.append({"idx": d["idx"], "prompt_id": d["prompt_id"], **compare_runs(*runs),
                            "warnings": [x["warnings"] for x in runs]})
        verdict = all(x["pass"] for x in results)
        env["determinism_warnings"] = [{"idx": x["idx"], "runs": x["warnings"]} for x in results]
        (a.out / "run_env.json").write_text(json.dumps(env, indent=1))
        (a.out / "b0_result.json").write_text(json.dumps({"pass": verdict, "tolerance": HIDDEN_TOL,
                                                          "results": results}, indent=1))
        print("D71-B0", "PASS" if verdict else "FAIL")
        return 0 if verdict else 8
    conds = ("A",) if a.mode == "calibrate" else CONDS
    design = build_design(ids, ctx, conds=conds, samples=SAMPLES)
    if a.limit:
        design = design[: a.limit]
    (a.out / "design.json").write_text(json.dumps([{k: v for k, v in d.items() if k != "user"} for d in design]))
    return run_generations(design, provider, a.out, system_prompt, mem_fraction=gpu_mem_fraction)


if __name__ == "__main__":
    raise SystemExit(main())
