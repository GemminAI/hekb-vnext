"""Source text of a session as a blob: text, token ids, per-token byte and character ranges.

Token ``t`` of the session is ``ids[t]`` = ``causal_log[t]`` (``token_index`` t). The text is the
byte-exact concatenation of the byte-level BPE pieces (Llama-3 tokenizer), decoded as UTF-8 with
replacement. A token that ends inside a multi-byte character gets an empty character range; the
character is attributed to the token that completes it.
"""

from __future__ import annotations

import codecs
import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "hekbv3.source_text/1"


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


class Decoder:
    """Byte-level BPE decoder from a ``tokenizer.json`` (ids >= ``n_base`` are special: no bytes)."""

    def __init__(self, tokenizer_json: Path, n_base: int = 128000) -> None:
        tk = json.loads(Path(tokenizer_json).read_text())
        self.inv = {i: t for t, i in tk["model"]["vocab"].items()}
        self.u2b = _byte_decoder()
        self.n_base = n_base

    def piece(self, token_id: int) -> bytes:
        if token_id >= self.n_base or token_id not in self.inv:
            return b""
        return bytes(self.u2b[c] for c in self.inv[token_id])


def build(decoder: Decoder, session_id: str, token_ids: list[int], generated_text: str) -> dict[str, Any]:
    inc = codecs.getincrementaldecoder("utf-8")("replace")
    chars: list[str] = []
    byte_ranges: list[list[int]] = []
    char_ranges: list[list[int]] = []
    offset = 0
    for tid in token_ids:
        piece = decoder.piece(int(tid))
        c0 = len(chars)
        text = "".join(inc.decode(bytes([b])) for b in piece)
        chars.extend(text)
        byte_ranges.append([offset, offset + len(piece)])
        char_ranges.append([c0, len(chars)])
        offset += len(piece)
    chars.extend(inc.decode(b"", final=True))
    text = "".join(chars)

    def squash(x: str) -> str:
        return re.sub(r"\s+", "", x)

    return {
        "schema": SCHEMA,
        "session_id": session_id,
        "text": text,
        "token_ids": [int(t) for t in token_ids],
        "token_byte_ranges": byte_ranges,
        "token_char_ranges": char_ranges,
        "decode_equal_generated_text_modulo_whitespace": squash(text) == squash(generated_text),
        "note": "generated_text in meta.json is tokenizer-cleaned; this text is the byte-exact decode",
    }
