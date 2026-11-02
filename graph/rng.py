"""``sha256_rejection_v1``: the one random source of CTP selections (CTP data model v0.3 section 6, decided 11-03).

* The i-th number of a stream is the 256-bit integer ``int(sha256(JCS({"seed": seed, "counter": i})), big-endian)``;
  ``counter`` starts at 0 for every selection (an assignment, an experience) and counts every number taken,
  rejected ones included.
* To choose among ``m`` possibilities, a number ``r >= floor(2**256 / m) * m`` is rejected (the next counter is used)
  and the accepted one gives ``r % m``: every possibility has exactly the same chance (no modulo bias).
* ``k`` items without replacement are drawn one after another from what is left, in the order of the candidates
  (the caller sorts them); the order drawn is the order of presentation. There is no shuffle.

Nothing here reads a clock, the process's hash seed or Python's ``random``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Final

from graph.jcs import canonical_bytes
from graph.model import GraphError

ALGORITHM: Final = "sha256_rejection_v1"
MAX_SEED_LENGTH: Final = 128
_SPAN: Final = 2**256


class RngError(GraphError):
    """A seed or a request the stream cannot serve."""


def check_seed(seed: object) -> str:
    if not isinstance(seed, str) or not 1 <= len(seed) <= MAX_SEED_LENGTH:
        raise RngError(f"rng seed must be a string of 1 to {MAX_SEED_LENGTH} characters")
    return seed


def word(seed: str, counter: int) -> int:
    """The ``counter``-th 256-bit number of the stream of ``seed``."""
    return int.from_bytes(hashlib.sha256(canonical_bytes({"seed": seed, "counter": counter})).digest(), "big")


class Stream:
    def __init__(self, seed: str) -> None:
        self.seed = check_seed(seed)
        self.counter = 0

    def below(self, m: int) -> int:
        """A number in ``range(m)``, chosen with the same chance for each."""
        if not isinstance(m, int) or isinstance(m, bool) or m < 1:
            raise RngError("a choice needs at least one possibility")
        limit = (_SPAN // m) * m
        while True:
            r = word(self.seed, self.counter)
            self.counter += 1
            if r < limit:
                return r % m


def choose_one(candidates: Sequence[str], seed: str) -> str:
    return candidates[Stream(seed).below(len(candidates))]


def choose_ordered(candidates: Sequence[str], k: int, seed: str) -> list[str]:
    """``k`` candidates without replacement, in the order they were drawn."""
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(candidates):
        raise RngError(f"k must be an integer from 1 to the number of candidates ({len(candidates)})")
    stream, left, drawn = Stream(seed), list(candidates), []
    for _ in range(k):
        drawn.append(left.pop(stream.below(len(left))))
    return drawn
