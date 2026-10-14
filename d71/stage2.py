#!/usr/bin/env python3
"""EXP-D71-CONTEXT-001 stage 2 (Mac, MLX 4bit): 40 items x {sufficient, deficient} x {A,B,C,C',D} x 2 samples.

Goes through ``sensos_inference.Llama31MLXProvider`` (sensos commit cb155004). Hidden states are not taken.
Stop conditions (v0.2-A 4): memory > 80%, one generation > 10 minutes, Locked touched (never read here).

    python3 -m d71.stage2 --out DIR [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVID = Path.home() / "vaults/20260124/experiment-evidence/EXP-D71-CONTEXT-001"
WORK = Path("/Volumes/SSD1TB/d71_work")
MODEL_DIR = WORK / "models/Meta-Llama-3.1-8B-Instruct-4bit"
REPO = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
REVISION = "241a666dad6cb93c8ff213d39a7f34a36bf26db4"
MSD = Path("/Volumes/SSD1TB/EXP-MSD0001 v1.1")
CONDS = ("A", "B", "C", "Cprime", "D")
SAMPLES = (0, 1)
MEMORY_LIMIT = 0.80
GEN_TIMEOUT_S = 600.0
MAX_TOKENS = 768


def memory_used_fraction() -> float:
    total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    out = subprocess.check_output(["vm_stat"]).decode()
    page = int(out.split("page size of ")[1].split(" bytes")[0])
    pages: dict[str, int] = {}
    for line in out.splitlines()[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            digits = value.strip().rstrip(".")
            if digits.isdigit():
                pages[key.strip()] = int(digits)
    used = pages.get("Pages active", 0) + pages.get("Pages wired down", 0) + pages.get("Pages occupied by compressor", 0)
    return used * page / total


class StopRun(RuntimeError):
    pass


def seed_for(prompt_id: str, version: str, sample: int) -> int:
    return int(hashlib.sha256(f"D71|seed|{prompt_id}|{version}|{sample}".encode()).hexdigest()[:8], 16)


def build_design(stage1: Path) -> list[dict[str, Any]]:
    sel = json.loads((stage1 / "selection40.json").read_text())["items"]
    design: list[dict[str, Any]] = []
    rng = random.Random(20261014)
    for it in sel:
        cs = json.loads((stage1 / "contexts" / it["prompt_id"] / "context_set.json").read_text())
        block = []
        for version in ("sufficient", "deficient"):
            for cond in CONDS:
                for sample in SAMPLES:
                    c = cs["conditions"][f"{version}:{cond}"]
                    block.append({
                        "prompt_id": it["prompt_id"], "kind": cs["kind"], "difficulty": cs["difficulty"],
                        "version": version, "cond": cond, "sample": sample,
                        "seed": seed_for(it["prompt_id"], version, sample),
                        "user": c["user"], "context_sha256": c["sha256"], "context_tokens": c["tokens"],
                        "removed": (cs["deficient"] or {}).get("removed") if version == "deficient" else None,
                        "expected": cs["expected"] if version == "sufficient" else None,
                    })
        rng.shuffle(block)
        design.extend(block)
    for i, d in enumerate(design):
        d["idx"] = i
    return design


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EVID / "stage2")
    ap.add_argument("--stage1", type=Path, default=EVID / "stage1")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    design = build_design(args.stage1)
    (args.out / "design.json").write_text(json.dumps(
        [{k: v for k, v in d.items() if k not in ("user",)} for d in design], indent=0))
    meta = json.loads((MSD / "sessions/p11-recall-0932/meta.json").read_text())
    system_prompt = meta["system_prompt"]
    ctx0 = json.loads((args.stage1 / "contexts" / design[0]["prompt_id"] / "context_set.json").read_text())
    assert hashlib.sha256(system_prompt.encode()).hexdigest() == ctx0["system_prompt_sha256"]

    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from sensos_inference.providers.llama31_mlx import Llama31MLXProvider

    state = {"t0": 0.0}

    def guarded_stream(model: Any, tokenizer: Any, prompt: Any, max_tokens: int, sampler: Any) -> Any:
        state["t0"] = time.perf_counter()
        for n, resp in enumerate(stream_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)):
            if time.perf_counter() - state["t0"] > GEN_TIMEOUT_S:
                raise TimeoutError(f"generation exceeded {GEN_TIMEOUT_S:.0f} s")
            if n % 32 == 0 and memory_used_fraction() > MEMORY_LIMIT:
                raise MemoryError("memory use above 80%")
            yield resp

    provider = Llama31MLXProvider(REPO, revision=REVISION, loader=lambda _mid: load(str(MODEL_DIR)), stream=guarded_stream)
    runs = args.out / "generations.jsonl"
    done = set()
    if runs.exists():
        done = {json.loads(line)["idx"] for line in runs.read_text().splitlines() if line.strip()}
    todo = [d for d in design if d["idx"] not in done]
    if args.limit is not None:
        todo = todo[: args.limit]
    peak_frac = 0.0
    t_run = time.time()
    first10: list[dict[str, float]] = []
    for d in todo:
        frac = memory_used_fraction()
        if frac > MEMORY_LIMIT:
            print(f"STOP: memory use {frac:.0%}", flush=True)
            return 3
        started = datetime.now(timezone.utc).isoformat()
        r = provider.generate_chat(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": d["user"]}],
            max_tokens=MAX_TOKENS, temperature=0.7, top_k=50, top_p=0.95, seed=d["seed"],
        )
        ended = datetime.now(timezone.utc).isoformat()
        frac = memory_used_fraction()
        peak_frac = max(peak_frac, frac)
        row = {
            **{k: v for k, v in d.items() if k != "user"},
            "text": r.text, "token_ids": r.token_ids, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "finish_reason": r.finish_reason,
            "failure_kind": r.failure_kind, "error": r.error, "ok": r.ok,
            "generate_seconds": r.generate_seconds, "load_seconds": r.load_seconds,
            "tokens_per_second": (r.completion_tokens / r.generate_seconds) if r.generate_seconds and r.completion_tokens else None,
            "memory_used_fraction": round(frac, 4), "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
            "started_utc": started, "ended_utc": ended, "sampler": r.sampler,
            "chat_template_sha256": r.chat_template_sha256, "model_ref": r.model_ref, "revision": r.revision,
            "device": r.device,
        }
        with runs.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if len(first10) < 10:
            first10.append({"idx": d["idx"], "tokens": r.completion_tokens or 0, "seconds": r.generate_seconds or 0,
                            "prompt_tokens": r.prompt_tokens or 0, "mem": frac, "peak_gb": mx.get_peak_memory() / 1e9})
            if len(first10) == 10:
                (args.out / "first10.json").write_text(json.dumps(first10, indent=1))
                print("FIRST10", json.dumps(first10), flush=True)
        if r.failure_kind in ("generate_failed", "load_failed", "eos_missing") or (r.error and "exceeded" in r.error):
            print("STOP:", r.failure_kind, r.error, flush=True)
            return 4
        n_done = len(done) + todo.index(d) + 1
        if n_done % 20 == 0:
            print(f"{n_done}/{len(design)} elapsed {time.time() - t_run:.0f}s mem {frac:.0%}", flush=True)
    print("DONE", f"{time.time() - t_run:.0f}s", f"peak mem {peak_frac:.0%}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
