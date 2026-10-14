"""Verifiable tasks of MSD0001 v1.1 (EXP-TURF-WAR ``msd/tasks.py``, frozen tag msd0001-v1.1-frozen):
prompt rendering, the sufficient/deficient versions of an item, and the stage-1 checks (v0.2-A 2.3).

The sufficient prompt is re-rendered from ``params`` and compared with the stored prompt; the deficient
prompt replaces ONE answer-determining value by a reference to a value "given earlier" (never given).
"""

from __future__ import annotations

import datetime
import hashlib
import re
from typing import Any

ANSWER_INSTRUCTION = (
    "Keep your reasoning concise (at most 200 words). Your response MUST end with a final line in exactly "
    "this form, and nothing after it: ANSWER: <value>"
)
#: stage-2/3 output format (all conditions, including A): ANSWER or CLARIFY
FORMAT_V2 = (
    "Keep your reasoning concise (at most 200 words). Your response MUST end with a final line in exactly "
    "one of these two forms, and nothing after it: 'ANSWER: <value>' if the question gives every value "
    "needed; or 'CLARIFY: <the missing value>' if a value needed to answer is not given in the question "
    "(do not guess it)."
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
KINDS = ("nth_prime", "pow2", "pi_digits", "arith_chain", "kth_largest", "remainder", "day_of_week")
REMOVABLE: dict[str, tuple[str, ...]] = {
    "nth_prime": ("n",),
    "pow2": ("n",),
    "pi_digits": ("decimals",),
    "arith_chain": ("a", "b", "c", "d", "e"),
    "kth_largest": ("k",),
    "remainder": ("a", "b", "m"),
    "day_of_week": ("year",),
}
GIVEN = "the {name} given earlier"


def _ord(k: int) -> str:
    return f"{k}th" if k > 3 else ["", "", "2nd", "3rd"][k]


def body_sufficient(kind: str, p: dict[str, Any]) -> str:
    if kind == "nth_prime":
        return f"What is the {p['n']}th prime number?"
    if kind == "pow2":
        return f"What is 2 raised to the power {p['n']}, written out exactly?"
    if kind == "pi_digits":
        return f"Write pi to exactly {p['decimals']} decimal places (truncated, not rounded)."
    if kind == "arith_chain":
        return f"Compute ({p['a']} x {p['b']}) - ({p['c']} x {p['d']}) + {p['e']}."
    if kind == "kth_largest":
        nums = ", ".join(map(str, p["numbers"]))
        return f"Here are {len(p['numbers'])} numbers: {nums}. What is the {_ord(p['k'])} largest of them?"
    if kind == "remainder":
        return f"What is the remainder when {p['a']}^{p['b']} is divided by {p['m']}?"
    if kind == "day_of_week":
        d = datetime.date.fromisoformat(p["date"])
        return f"Which day of the week was {d.day} {d.strftime('%B')} {d.year}?"
    raise ValueError(kind)


def body_deficient(kind: str, p: dict[str, Any], removed: str) -> str:
    g = lambda name: GIVEN.format(name=name)  # noqa: E731
    if kind == "nth_prime":
        return "What is the prime number at the position given earlier?"
    if kind == "pow2":
        return "What is 2 raised to the power given earlier, written out exactly?"
    if kind == "pi_digits":
        return "Write pi to exactly the number of decimal places given earlier (truncated, not rounded)."
    if kind == "arith_chain":
        q = {k: p[k] for k in "abcde"}
        q[removed] = g("number")  # type: ignore[assignment]
        return f"Compute ({q['a']} x {q['b']}) - ({q['c']} x {q['d']}) + {q['e']}."
    if kind == "kth_largest":
        nums = ", ".join(map(str, p["numbers"]))
        return (
            f"Here are {len(p['numbers'])} numbers: {nums}. "
            "What is the number at the rank given earlier when they are sorted from largest to smallest?"
        )
    if kind == "remainder":
        q = {k: p[k] for k in "abm"}
        q[removed] = g("value")  # type: ignore[assignment]
        return f"What is the remainder when {q['a']}^{q['b']} is divided by {q['m']}?"
    if kind == "day_of_week":
        d = datetime.date.fromisoformat(p["date"])
        return f"Which day of the week was {d.day} {d.strftime('%B')} of the year given earlier?"
    raise ValueError(kind)


# ---- ground truth ------------------------------------------------------------------------------
def _nth_prime(n: int) -> int:
    count, k = 0, 1
    while count < n:
        k += 1
        if all(k % d for d in range(2, int(k**0.5) + 1)):
            count += 1
    return k


def _pi_digits(n: int) -> str:
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239), integer arithmetic
    def arccot(x: int, unity: int) -> int:
        total = term = unity // x
        x2, k, sign = x * x, 3, -1
        while term:
            term //= x2
            total += sign * (term // k)
            sign, k = -sign, k + 2
        return total

    unity = 10 ** (n + 10)
    pi = 4 * (4 * arccot(5, unity) - arccot(239, unity))
    s = str(pi)[: n + 1]
    return s[0] + "." + s[1:] if n else s[0]


def answer(kind: str, p: dict[str, Any]) -> str:
    if kind == "nth_prime":
        return str(_nth_prime(p["n"]))
    if kind == "pow2":
        return str(2 ** p["n"])
    if kind == "pi_digits":
        return _pi_digits(p["decimals"])
    if kind == "arith_chain":
        return str(p["a"] * p["b"] - p["c"] * p["d"] + p["e"])
    if kind == "kth_largest":
        return str(sorted(p["numbers"], reverse=True)[p["k"] - 1])
    if kind == "remainder":
        return str(pow(p["a"], p["b"], p["m"]))
    if kind == "day_of_week":
        return WEEKDAYS[datetime.date.fromisoformat(p["date"]).weekday()]
    raise ValueError(kind)


# ---- domains of the removed value (make_task ranges, by tier) -----------------------------------
def domain(kind: str, key: str, difficulty: str, p: dict[str, Any]) -> list[Any]:
    hard = difficulty == "hard"
    if kind == "nth_prime" or kind == "pow2":
        lo, hi = {"nth_prime": ((8, 30), (200, 450)), "pow2": ((8, 20), (70, 110))}[kind][hard]
        return list(range(lo, hi + 1))
    if kind == "pi_digits":
        return list(range(35, 61)) if hard else list(range(4, 9))
    if kind == "arith_chain":
        lo, hi = (1000, 9999) if hard else (11, 49)
        return list(range(lo, hi + 1, 97 if hard else 1))
    if kind == "kth_largest":
        return list(range(5, 10)) if hard else [2, 3]
    if kind == "remainder":
        easy = {"a": range(2, 6), "b": range(3, 7), "m": range(5, 14)}
        hardd = {"a": range(3, 10), "b": range(30, 61), "m": range(11, 98)}
        return list((hardd if hard else easy)[key])
    if kind == "day_of_week":
        return list(range(1700, 1900)) if hard else list(range(2001, 2025))
    raise ValueError(kind)


def _with_value(kind: str, p: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    q = dict(p)
    if kind == "day_of_week":
        d = datetime.date.fromisoformat(p["date"])
        q["date"] = datetime.date(value, d.month, d.day).isoformat()
    else:
        q[key] = value
    return q


def _literal(kind: str, p: dict[str, Any], key: str) -> str:
    if kind == "day_of_week":
        return str(datetime.date.fromisoformat(p["date"]).year)
    return str(p[key])


def removed_literal_present(body: str, literal: str) -> bool:
    return re.search(rf"(?<![\d.]){re.escape(literal)}(?![\d])", body) is not None


def build_item(rec: dict[str, Any]) -> dict[str, Any]:
    """rec: kind, difficulty, params, prompt (stored), expected (stored), prompt_id. Returns the item with
    its checks. ``valid`` is true when (i), (ii), (iii) of v0.2-A 2.3 all hold."""
    kind, p = rec["kind"], rec["params"]
    suffix = " " + ANSWER_INSTRUCTION
    stored = rec["prompt"]
    body = body_sufficient(kind, p)
    check_i = stored == body + suffix and answer(kind, p) == str(rec["expected"])
    keys = list(REMOVABLE[kind])
    h = int(hashlib.sha256(("D71|removed|" + rec["prompt_id"]).encode()).hexdigest(), 16)
    order = keys[h % len(keys) :] + keys[: h % len(keys)]
    out: dict[str, Any] = {
        "prompt_id": rec["prompt_id"], "kind": kind, "difficulty": rec["difficulty"],
        "params": p, "sufficient_body": body, "expected": str(rec["expected"]),
        "check_i_rerender_and_answer": check_i, "deficient": None, "attempts": [],
    }
    for key in order:
        dom = domain(kind, key, rec["difficulty"], p)
        answers = {answer(kind, _with_value(kind, p, key, v)) for v in dom}
        lit = _literal(kind, p, key)
        deficient = body_deficient(kind, p, key)
        ok_ii = len(answers) >= 2
        ok_iii = not removed_literal_present(deficient, lit)
        out["attempts"].append({"removed": key, "ii_distinct_answers": len(answers), "ii": ok_ii, "iii": ok_iii})
        if ok_ii and ok_iii and check_i:
            out["deficient"] = {"removed": key, "removed_literal": lit, "body": deficient}
            break
    out["valid"] = out["deficient"] is not None
    return out
