"""Error type shared by the reference implementation. ``code`` is language-neutral
(the same strings appear in docs/architecture/contracts/common/vectors/)."""

from __future__ import annotations


class ContractError(Exception):
    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


# strict_load / jcs / quantize codes
E_UTF8 = "E_UTF8"  # bytes are not valid UTF-8
E_SYNTAX = "E_SYNTAX"  # not JSON
E_SURROGATE = "E_SURROGATE"  # lone surrogate in a string or key
E_FLOAT_TOKEN = "E_FLOAT_TOKEN"  # number token with '.' or exponent
E_CONSTANT = "E_CONSTANT"  # NaN / Infinity / -Infinity
E_DUP_KEY = "E_DUP_KEY"  # duplicate object key (after unescaping)
E_INT_RANGE = "E_INT_RANGE"  # integer outside +-(2^53-1)
E_FLOAT_TYPE = "E_FLOAT_TYPE"  # a float reached jcs()
E_NOT_CANONICALIZABLE = "E_NOT_CANONICALIZABLE"
E_QUANT_NONFINITE = "E_QUANT_NONFINITE"
E_QUANT_RANGE = "E_QUANT_RANGE"
E_PARENTS = "E_PARENTS"  # malformed parents for id computation
E_LOOKUP = "E_LOOKUP"  # a parent needed for content_id is not in lookup
E_ID_INPUT = "E_ID_INPUT"  # bad input to session_id / observation_id
