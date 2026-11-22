"""Condition contexts A, B, C, C', D (v0.2-A 2.4, 2.7, 2.8). No function here receives the expected answer."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

K = 3
CPRIME_RULE = (
    "D71-Cprime rule v1\n"
    "1. candidates = library sessions (one per prompt_id: the smallest sha256(session_id)) whose kind differs "
    "from the item's kind; same prompt_id/prompt_sha256 never included.\n"
    "2. h = sha256('D71-Cprime|' + prompt_id). kind choice: the candidate kinds sorted lexicographically; start "
    "index = int(h,16) mod number_of_kinds.\n"
    "3. within a kind order by sha256(h + '|' + session_id) ascending; take 3; if the kind has fewer than 3, "
    "continue with the next kind (cyclic, lexicographic).\n"
    "4. if the observation section token count differs from C's by more than 10%, start from the next kind "
    "(cyclic) and repeat 2-3; if every start fails the item has no C' (dropped, counted).\n"
)
CPRIME_RULE_SHA256 = hashlib.sha256(CPRIME_RULE.encode()).hexdigest()

RETRIEVAL_CONFIG: dict[str, Any] = {
    "version": "retrieval_config_v1",
    "method": "deterministic rule search: same kind, ascending parameter distance, tie sha256(session_id)",
    "embedding_model": None,
    "k": K,
    "chunk": "one session",
    "order": "distance ascending (kept in the context)",
    "population": "fit library, one session per prompt_id, derivability screen of 2.7",
    "summarizer": "summarize_v1",
    "cprime_rule_sha256": CPRIME_RULE_SHA256,
}


def retrieval_config_sha256(extra: dict[str, Any] | None = None) -> str:
    cfg = dict(RETRIEVAL_CONFIG)
    cfg.update(extra or {})
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ItemPublic:
    """What the context builder may see of an evaluation item (no expected answer)."""

    prompt_id: str
    prompt_sha256: str
    kind: str
    difficulty: str
    params: dict[str, Any]
    body_sufficient: str
    body_deficient: str | None


def _dist(kind: str, p: dict[str, Any], q: dict[str, Any]) -> float:
    import datetime

    if kind in ("nth_prime", "pow2"):
        return abs(p["n"] - q["n"])
    if kind == "pi_digits":
        return abs(p["decimals"] - q["decimals"])
    if kind == "day_of_week":
        a = datetime.date.fromisoformat(p["date"])
        b = datetime.date.fromisoformat(q["date"])
        return abs((a - b).days)
    if kind == "arith_chain":
        return sum(abs(math.log10(p[k]) - math.log10(q[k])) for k in "abcde")
    if kind == "kth_largest":
        return abs(p["k"] - q["k"]) + 2 * abs(len(p["numbers"]) - len(q["numbers"]))
    if kind == "remainder":
        return sum(abs(math.log(p[k]) - math.log(q[k])) for k in "abm")
    raise ValueError(kind)


def derivable(kind: str, lib: dict[str, Any], ev: dict[str, Any]) -> bool:
    """True if the library case could let the evaluation answer be read off (2.7-2): excluded."""
    import datetime

    if lib == ev:
        return True
    if kind in ("nth_prime", "pow2"):
        return abs(lib["n"] - ev["n"]) <= 2
    if kind == "pi_digits":
        return lib["decimals"] >= ev["decimals"]
    if kind == "day_of_week":
        a = datetime.date.fromisoformat(lib["date"])
        b = datetime.date.fromisoformat(ev["date"])
        return a.year == b.year and abs((a - b).days) <= 7
    if kind == "kth_largest":
        return sorted(lib["numbers"]) == sorted(ev["numbers"])
    if kind == "arith_chain":
        return all(lib[k] == ev[k] for k in "abcde")
    if kind == "remainder":
        return all(lib[k] == ev[k] for k in "abm")
    return False


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def retrieve_b(item: ItemPublic, library: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    cands = [
        e for e in library
        if e["kind"] == item.kind and e["prompt_id"] != item.prompt_id
        and e["prompt_sha256"] != item.prompt_sha256 and not derivable(item.kind, e["params"], item.params)
    ]
    cands.sort(key=lambda e: (_dist(item.kind, e["params"], item.params), _h(e["session"])))
    return cands[:K] if len(cands) >= K else None


def select_cprime(
    item: ItemPublic, library: list[dict[str, Any]], count_tokens: Any, c_obs_text: str, obs_of: Any
) -> tuple[list[dict[str, Any]] | None, str]:
    h = _h("D71-Cprime|" + item.prompt_id)
    others = [e for e in library if e["kind"] != item.kind and e["prompt_id"] != item.prompt_id]
    kinds = sorted({e["kind"] for e in others})
    if not kinds:
        return None, "no other kind"
    target = count_tokens(c_obs_text)
    for shift in range(len(kinds)):
        start = (int(h, 16) + shift) % len(kinds)
        order = kinds[start:] + kinds[:start]
        chosen: list[dict[str, Any]] = []
        for kind in order:
            pool = sorted([e for e in others if e["kind"] == kind], key=lambda e: _h(h + "|" + e["session"]))
            for e in pool:
                if len(chosen) < K:
                    chosen.append(e)
            if len(chosen) >= K:
                break
        if len(chosen) < K:
            continue
        text = obs_section(chosen, obs_of)
        n = count_tokens(text)
        if abs(n - target) <= 0.10 * target:
            return chosen, f"ok shift={shift} tokens={n} vs {target}"
    return None, "length-mismatch for every start kind"


def cases_section(cases: list[dict[str, Any]]) -> str:
    lines = ["Verified past cases (different problems; each answer was checked by code):"]
    for i, c in enumerate(cases, 1):
        lines += [
            f"[{i}] session {c['session']} ({c['kind']})",
            f"Problem: {c['body']}",
            f"Model's final answer: {c['extracted']}",
            "Machine check: correct.",
        ]
    return "\n".join(lines)


def obs_section(cases: list[dict[str, Any]], obs_of: Any) -> str:
    return "Observations of earlier runs (measured values only):\n" + "\n".join(obs_of(c["session"]) for c in cases)


def assemble(version: str, body: str, parts: list[str]) -> str:
    from d71.tasks import FORMAT_V2

    ctx = "\n\n".join(p for p in parts if p)
    q = f"Question: {body} {FORMAT_V2}"
    return (ctx + "\n\n" + q) if ctx else q


def leak_literal(text: str, expected: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(expected)}(?![\d])", text) is not None
