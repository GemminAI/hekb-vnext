"""RFC 8785 (JSON Canonicalization Scheme) for the values CTP seals: no floating point.

``canonicalize(value)`` returns the canonical text, ``canonical_bytes(value)`` its UTF-8 bytes.

* objects: members in the order of their keys as UTF-16 code units, no whitespace;
* arrays, ``true``, ``false``, ``null``;
* strings as ECMAScript ``JSON.stringify``: ``"`` ``\\`` ``\\b`` ``\\f`` ``\\n`` ``\\r`` ``\\t`` are the short
  escapes, any other code point below U+0020 is ``\\u00xx`` (lower-case hex), everything else is written as
  it is (U+007F, U+2028, U+2029 and non-ASCII included). A lone surrogate is refused;
* integers as plain decimal digits, and only inside +-(2**53 - 1): a larger integer cannot be read back
  exactly by every JSON implementation, so it must be passed as a string (``IntegerOutOfRange``);
* a floating-point number is refused (``FloatNotAllowed``); probabilities are irreducible fractions
  ``{num, den}`` (CTP data model v0.3 section 2).

The refusals are this function's own. ``graph.chainlog.canonical`` (the hash chain's serialisation, D5) and
``GraphStore`` writes are not changed and do not call it.
"""

from __future__ import annotations

from typing import Any, Final

from graph.model import GraphError

MAX_SAFE_INTEGER: Final = 2**53 - 1

_SHORT: Final = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


class JcsError(GraphError):
    """A value that cannot be written as canonical JSON."""


class FloatNotAllowed(JcsError):
    """A floating-point number (CTP seals integers, strings and fractions only)."""


class IntegerOutOfRange(JcsError):
    """An integer beyond +-(2**53 - 1): pass it as a string."""


def _string(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            raise JcsError("lone surrogate in a string")
        out.append(_SHORT.get(ch, f"\\u{o:04x}" if o < 0x20 else ch))
    out.append('"')
    return "".join(out)


def canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        raise FloatNotAllowed(f"floating-point number {value!r} is not allowed; use an integer, a string or a fraction")
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise IntegerOutOfRange(f"integer {value} is outside +-(2**53 - 1); pass it as a string")
        return str(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonicalize(x) for x in value) + "]"
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise JcsError(f"object key {k!r} is not a string")
        keys = sorted(value, key=lambda k: k.encode("utf-16-be", "surrogatepass"))
        return "{" + ",".join(_string(k) + ":" + canonicalize(value[k]) for k in keys) + "}"
    raise JcsError(f"not a canonicalizable value: {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    return canonicalize(value).encode("utf-8")
