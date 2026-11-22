"""``summarize_v1``: deterministic observation summary of one stored session (v0.2-A 2.5).

Facts only (positions and numbers); no interpretation and no correctness (correctness is the machine
verdict, given in the evidence block). Inputs are the causal-log records and the per-token pieces.
"""

from __future__ import annotations

import hashlib
import inspect
import statistics
from collections.abc import Sequence
from typing import Any

from graph.trace import port_scalar

VERSION = "summarize_v1"
CAPACITY = 8
#: anchor-independent set of the HEKBv3 addendum A (the 10-port window feature)
PORTS10 = ("P01", "P03", "P09", "P13", "P18", "P19", "P25", "P28", "P31", "P35")

TEMPLATE = (
    "Run {session} (node {node}, {version}).\n"
    "- Generated tokens: {n}. Full measurement window (n={cap}) from token {first_full}.\n"
    "- mean_step (P03): median {step_med}, max {step_max} at token {step_at} (\"{step_piece}\").\n"
    "- P18 v_norm (full window only): median {v_med}, max {v_max} at token {v_at} (\"{v_piece}\"); "
    "first above the fit-reference threshold {theta}: {first_above}.\n"
    "- Port medians {ports}.\n"
    "- Time axis: event-time-v1 (1 s per generated token; synthetic, not wall-clock). "
    "No interpretation and no correctness is stated."
)


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _piece(pieces: Sequence[str], t: int) -> str:
    s = pieces[t] if 0 <= t < len(pieces) else ""
    return s.replace("\n", "\\n").replace('"', "'")[:12]


def summarize_v1(
    *,
    session: str,
    object_id: str,
    records: Sequence[dict[str, Any]],
    pieces: Sequence[str],
    theta: float,
) -> str:
    n = len(records)
    step = [port_scalar("P03", r["ports"]["P03"].get("result")) for r in records]
    p18 = [r["ports"]["P18"].get("result") or {} for r in records]
    full = [t for t, res in enumerate(p18) if res.get("n") == CAPACITY and res.get("v_norm") is not None]
    first_full = full[0] if full else None
    step_ok = [t for t in range(n) if step[t] == step[t]]
    st = max(step_ok, key=lambda t: step[t]) if step_ok else 0
    step_med = statistics.median(step[t] for t in step_ok) if step_ok else float("nan")
    if full:
        v = {t: float(p18[t]["v_norm"]) for t in full}
        v_at = max(v, key=lambda t: v[t])
        v_med, v_max = statistics.median(v.values()), v[v_at]
        above = [t for t in full if v[t] > theta]
        first_above = f"token {above[0]} (\"{_piece(pieces, above[0])}\")" if above else "none"
    else:
        v_at, v_med, v_max, first_above = 0, float("nan"), float("nan"), "none"
    port_parts = []
    for port in PORTS10:
        vals = [port_scalar(port, r["ports"][port].get("result")) for r in records]
        vals = [x for x in vals if x == x]
        port_parts.append(f"{port} {_fmt(statistics.median(vals))}" if vals else f"{port} n/a")
    return TEMPLATE.format(
        session=session, node=object_id[:12], version=VERSION, n=n, cap=CAPACITY,
        first_full="n/a" if first_full is None else first_full,
        step_med=_fmt(step_med), step_max=_fmt(step[st]) if step_ok else "n/a", step_at=st,
        step_piece=_piece(pieces, st), v_med=_fmt(v_med), v_max=_fmt(v_max), v_at=v_at,
        v_piece=_piece(pieces, v_at), theta=_fmt(theta), first_above=first_above,
        ports=", ".join(port_parts),
    )


def version_fingerprint() -> dict[str, str]:
    return {
        "version": VERSION,
        "source_sha256": hashlib.sha256(inspect.getsource(summarize_v1).encode()).hexdigest(),
        "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
    }
