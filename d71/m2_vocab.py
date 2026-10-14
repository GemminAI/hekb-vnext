# ruff: noqa: E501
"""M2 vocabulary (v0.2-A 3.5): per (kind, removed value), regular expressions (case-insensitive) that the
text after ``CLARIFY:`` must match for the response to count as naming the missing value. Fixed (sha256
recorded) BEFORE the stage-2 run; changing it needs a new version and a re-scoring record."""

import hashlib
import json

VERSION = "m2_vocab_v1.1"
def _one(x: str) -> list[str]:
    """Single-letter names only in the forms 'value of X', 'X =', 'X is' (not the article 'a')."""
    return [rf"\bvalue of {x}\b", rf"\b{x}\s*=", rf"\b{x}\s+is\b"]


_GEN = r"\b(missing|unknown|not (been )?(given|provided|specified|stated)|unspecified)\b"
VOCAB: dict[str, list[str]] = {
    "nth_prime:n": [r"\bposition\b", r"\bwhich prime\b", r"\b(value|number) of n\b", *_one("n"), r"\bindex\b", r"\bnth\b", r"\bhow many\b", r"\bwhich (one|number)\b"],
    "pow2:n": [r"\bexponent\b", r"\bpower\b", r"\b(value|number) of n\b", *_one("n"), r"\bwhich (power|exponent)\b"],
    "pi_digits:decimals": [r"\bdecimal(s| places)?\b", r"\bdigits?\b", r"\bplaces\b", r"\bhow many\b", r"\bprecision\b", r"\bnumber of\b"],
    "arith_chain:a": [r"\b(first|1st) (number|factor|operand)\b", r"\bnumber\b", r"\bvalue\b", r"\boperand\b", r"\bfactor\b", r"\bmultiplier\b", _GEN],
    "arith_chain:b": [r"\b(second|2nd) (number|factor|operand)\b", r"\bnumber\b", r"\bvalue\b", r"\boperand\b", r"\bfactor\b", r"\bmultiplier\b", _GEN],
    "arith_chain:c": [r"\b(third|3rd) (number|factor|operand)\b", r"\bnumber\b", r"\bvalue\b", r"\boperand\b", r"\bfactor\b", r"\bmultiplier\b", _GEN],
    "arith_chain:d": [r"\b(fourth|4th) (number|factor|operand)\b", r"\bnumber\b", r"\bvalue\b", r"\boperand\b", r"\bfactor\b", r"\bmultiplier\b", _GEN],
    "arith_chain:e": [r"\b(fifth|5th|last|final) (number|term|operand)\b", r"\bnumber\b", r"\bvalue\b", r"\bterm\b", r"\boperand\b", _GEN],
    "kth_largest:k": [r"\brank\b", r"\bposition\b", r"\bwhich (largest|one)\b", *_one("k"), r"\bnth\b", r"\bhow (many|far)\b", r"\border\b"],
    "remainder:a": [r"\bbase\b", *_one("a"), r"\bnumber being raised\b", r"\bnumber\b", _GEN],
    "remainder:b": [r"\bexponent\b", r"\bpower\b", *_one("b"), _GEN],
    "remainder:m": [r"\bmodulus\b", r"\bdivisor\b", r"\bdivided by\b", *_one("m"), _GEN],
    "day_of_week:year": [r"\byear\b"],
}


def vocab_sha256() -> str:
    payload = json.dumps({"version": VERSION, "vocab": VOCAB}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
