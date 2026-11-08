"""RFC 8785 for envelope values: integers only.

Numbers are plain decimal integers within +-(2^53-1); floats are refused.
Object members are ordered by their keys as UTF-16 code units. Strings use the
ECMAScript JSON.stringify escapes (RFC 8785 section 3.2.2.2); lone surrogates are
refused.
"""

from __future__ import annotations

from typing import Any

from .errors import (
    E_FLOAT_TYPE,
    E_INT_RANGE,
    E_NOT_CANONICALIZABLE,
    E_SURROGATE,
    ContractError,
)

MAX_SAFE = 2**53 - 1
_SHORT = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _string(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            raise ContractError(E_SURROGATE, "lone surrogate in a string")
        if ch in _SHORT:
            out.append(_SHORT[ch])
        elif code < 0x20:
            out.append(f"\\u{code:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _utf16_key(key: str) -> bytes:
    return key.encode("utf-16-be", "surrogatepass")


def canonicalize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        raise ContractError(E_FLOAT_TYPE, f"float {value!r} is not allowed in an envelope")
    if isinstance(value, int):
        if abs(value) > MAX_SAFE:
            raise ContractError(E_INT_RANGE, f"{value} outside +-(2^53-1)")
        return str(value)
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, list | tuple):
        return "[" + ",".join(canonicalize(item) for item in value) + "]"
    if isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise ContractError(E_NOT_CANONICALIZABLE, f"object key {key!r} is not a string")
        keys = sorted(value, key=_utf16_key)
        return "{" + ",".join(_string(k) + ":" + canonicalize(value[k]) for k in keys) + "}"
    raise ContractError(E_NOT_CANONICALIZABLE, type(value).__name__)


def jcs(value: Any) -> bytes:
    """Canonical UTF-8 bytes of ``value``."""
    return canonicalize(value).encode("utf-8")
