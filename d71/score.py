# ruff: noqa: E501
#!/usr/bin/env python3
"""EXP-D71-CONTEXT-001 stage-2 scoring (v0.2-A 3.5). Machine rules only; labelled 'trial'.

M1 (sufficient)  last final line is ``ANSWER:`` and the value equals the expected one (msd ``verify`` rules,
                 ported verbatim). No parsable ``ANSWER:``/``CLARIFY:`` line -> 'undecided' (counted separately).
M5 (sufficient)  over-confirmation: final form CLARIFY (or no form and the last line ends with '?').
M2 (deficient)   final form CLARIFY (or ANSWER with an 'unknown' value) AND the clarification text matches the
                 vocabulary of the removed value.  Also reported: fabricated = ANSWER with a value;
                 clarify_any = CLARIFY (any text).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from d71 import m2_vocab  # noqa: E402

EVID = Path.home() / "vaults/20260124/experiment-evidence/EXP-D71-CONTEXT-001"
CONDS = ("A", "B", "C", "Cprime", "D")
CONTRASTS = (("B", "A"), ("C", "B"), ("C", "Cprime"), ("D", "C"))
BOOT = 2000
SEED = 20261014

_FORM_RE = re.compile(r"(ANSWER|CLARIFY)\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"-?\d[\d,]*")
_DEC_RE = re.compile(r"\d\.\d+")
_WEEKDAY_RE = re.compile("Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday", re.IGNORECASE)
_UNKNOWN_RE = re.compile(
    r"\b(unknown|not (been )?(given|provided|specified|stated|known)|missing|cannot|can't|undetermined|"
    r"insufficient|n/a|unspecified|needed?)\b", re.IGNORECASE)


def parse_final(text: str) -> tuple[str, str | None]:
    """('answer'|'clarify'|'none', value text) from the LAST ANSWER:/CLARIFY: occurrence."""
    matches = list(_FORM_RE.finditer(text))
    if not matches:
        return "none", None
    m = matches[-1]
    return m.group(1).lower().replace("answer", "answer"), m.group(2).strip()


def extract_value(line: str, kind: str) -> str | None:
    line = line.strip().replace("−", "-").lstrip("*`_ \"'([")
    if kind in ("nth_prime", "fibonacci", "pow2", "arith_chain", "kth_largest", "remainder"):
        m = _INT_RE.match(line)
        return m.group(0).rstrip(",") if m else (line.strip("*`_ ").rstrip(".").strip() or None)
    if kind == "day_of_week":
        m = _WEEKDAY_RE.search(line)
        return m.group(0).capitalize() if m else (line.strip("*`_ ").rstrip(".").strip() or None)
    if kind == "pi_digits":
        m = _DEC_RE.match(line)
        return m.group(0) if m else (line.strip("*`_ ").rstrip(".").strip() or None)
    return line.strip("*`_ ").rstrip(".").strip() or None


def _norm(kind: str, text: str) -> str:
    t = text.strip().strip("*`_ ").rstrip(".").strip()
    if kind == "day_of_week":
        return re.sub(r"[^a-z]", "", t.lower())
    if kind == "pi_digits":
        return re.sub(r"[^0-9.]", "", t)
    return re.sub(r"[,\s_]", "", t)


def correct(kind: str, expected: str, value: str) -> bool:
    got = extract_value(value, kind)
    if got is None:
        return False
    want, have = _norm(kind, expected), _norm(kind, got)
    return have[: len(want)] == want if kind == "pi_digits" else have == want


def vocab_match(key: str, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in m2_vocab.VOCAB.get(key, []))


def score_row(r: dict[str, Any]) -> dict[str, Any]:
    text = r["text"] or ""
    form, value = parse_final(text)
    out: dict[str, Any] = {"idx": r["idx"], "prompt_id": r["prompt_id"], "kind": r["kind"],
                           "version": r["version"], "cond": r["cond"], "sample": r["sample"], "form": form,
                           "finish": r["finish_reason"], "value": value}
    last_line = next((ln for ln in reversed(text.strip().splitlines()) if ln.strip()), "")
    if r["version"] == "sufficient":
        if form == "answer":
            out["m1"], out["undecided"] = bool(correct(r["kind"], r["expected"], value or "")), False
        else:
            out["m1"], out["undecided"] = False, form == "none"
        out["m5"] = form == "clarify" or (form == "none" and last_line.rstrip().endswith("?"))
    else:
        unknown_answer = form == "answer" and bool(_UNKNOWN_RE.search(value or ""))
        key = f"{r['kind']}:{r['removed']}"
        clarify_text = value if form == "clarify" else (value if unknown_answer else None)
        out["clarify_any"] = form == "clarify" or unknown_answer
        out["fabricated"] = form == "answer" and not unknown_answer
        out["m2"] = bool(out["clarify_any"] and vocab_match(key, clarify_text or ""))
        out["clarify_vocab_mismatch"] = bool(out["clarify_any"] and not out["m2"])
        out["undecided"] = form == "none"
    return out


def boot_ci(diffs: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    n = len(diffs)
    means = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(BOOT)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", type=Path, default=EVID / "stage2")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows = [json.loads(line) for line in (args.stage2 / "generations.jsonl").read_text().splitlines() if line.strip()]
    scored = [score_row(r) for r in rows]
    by_raw = {r["idx"]: r for r in rows}
    (args.stage2 / f"per_generation_scores{args.tag}.jsonl").write_text("\n".join(json.dumps(s) for s in scored))
    items = sorted({s["prompt_id"] for s in scored})
    metrics = {"sufficient": ("m1", "m5", "undecided"), "deficient": ("m2", "fabricated", "clarify_any", "undecided")}

    # item-level means per (version, metric, cond)
    cell: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for s in scored:
        for m in metrics[s["version"]]:
            cell[(s["version"], m, s["cond"])][s["prompt_id"]].append(float(s[m]))
    item_means: dict[tuple[str, str, str], np.ndarray] = {}
    for key, d in cell.items():
        item_means[key] = np.array([np.mean(d[i]) if i in d else np.nan for i in items])
    result: dict[str, Any] = {"n_items": len(items), "n_generations": len(scored), "tag": args.tag, "note": "TRIAL (MLX 4bit); no hypothesis is adopted or rejected (v0.2-A 3.6)"}
    rates: dict[str, Any] = {}
    for (version, m, cond), arr in sorted(item_means.items()):
        rates[f"{version}:{m}:{cond}"] = {
            "mean": float(np.nanmean(arr)),
            "between_item_sd": float(np.nanstd(arr, ddof=1)),
            "within_item_sample_sd": float(np.sqrt(np.nanmean([np.var(cell[(version, m, cond)][i], ddof=1) if len(cell[(version, m, cond)][i]) > 1 else np.nan for i in items]))),
            "n_decided_generations_undecided": int(sum(1 for s in scored if s["version"] == version and s["cond"] == cond and s["undecided"])),
        }
    result["rates"] = rates
    contrasts: dict[str, Any] = {}
    for version, ms in (("sufficient", ("m1", "m5")), ("deficient", ("m2", "fabricated", "clarify_any"))):
        for m in ms:
            for hi, lo in CONTRASTS:
                diffs = item_means[(version, m, hi)] - item_means[(version, m, lo)]
                diffs = diffs[~np.isnan(diffs)]
                sd = float(np.std(diffs, ddof=1))
                contrasts[f"{version}:{m}:{hi}-{lo}"] = {
                    "diff": float(diffs.mean()), "ci95": boot_ci(diffs), "sd_of_item_diffs": sd,
                    "illustrative_N_for_delta": {str(dl): round(7.85 * sd**2 / dl**2, 1) for dl in (0.10, 0.15, 0.20)},
                }
    result["contrasts"] = contrasts
    # M7
    m7: dict[str, Any] = {}
    for cond in CONDS:
        sub = [r for r in rows if r["cond"] == cond]
        m7[cond] = {
            "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in sub if r["prompt_tokens"]])),
            "completion_tokens_mean": float(np.mean([r["completion_tokens"] or 0 for r in sub])),
            "generate_seconds_mean": float(np.mean([r["generate_seconds"] or 0 for r in sub])),
            "length_limit": int(sum(1 for r in sub if r["failure_kind"] == "length_limit")),
        }
    result["m7"] = m7
    # by kind for A (ceiling/floor)
    kind_a: dict[str, Any] = {}
    for s in scored:
        if s["cond"] == "A" and s["version"] == "sufficient":
            kind_a.setdefault(s["kind"], []).append(float(s["m1"]))
    result["A_sufficient_m1_by_kind"] = {k: [round(float(np.mean(v)), 3), len(v)] for k, v in sorted(kind_a.items())}
    result["vocab_sha256"] = m2_vocab.vocab_sha256()
    result["vocab_version"] = m2_vocab.VERSION
    # vocabulary inspection set: CLARIFY without a vocabulary match (deterministic order)
    import hashlib

    mism = [s for s in scored if s.get("clarify_vocab_mismatch")]
    mism.sort(key=lambda s: hashlib.sha256(f"D71|vocab|{s['idx']}".encode()).hexdigest())
    result["clarify_vocab_mismatch_count"] = len(mism)
    insp = []
    for s in mism[:20]:
        insp.append({"idx": s["idx"], "kind": s["kind"], "removed": by_raw[s["idx"]]["removed"], "cond": s["cond"],
                     "clarify_text": s["value"]})
    (args.stage2 / f"vocab_inspection_set{args.tag}.json").write_text(json.dumps(insp, indent=1, ensure_ascii=False))
    (args.stage2 / f"scores{args.tag}.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("n_items", "n_generations", "clarify_vocab_mismatch_count", "A_sufficient_m1_by_kind")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
