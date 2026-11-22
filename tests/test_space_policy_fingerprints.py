"""The space carries the NVS-Kernel wiring and time policy fingerprints: measurements made under
a different wiring or time rule are a different space and must not compose."""

from __future__ import annotations

import pytest

from index.space_id import REQUIRED_KEYS, SpaceIdError, compute_space_id

SPACE = {
    "model_ref": "m", "layer": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": "a" * 64, "wiring_policy_sha256": "b" * 64,
    "time_policy_sha256": "c" * 64, "board_format": "36cell/1", "quant_step_log2": -30,
}  # fmt: skip


def test_policy_fingerprints_are_required_keys() -> None:
    assert "wiring_policy_sha256" in REQUIRED_KEYS
    assert "time_policy_sha256" in REQUIRED_KEYS


@pytest.mark.parametrize("key", ["wiring_policy_sha256", "time_policy_sha256"])
def test_a_space_without_the_fingerprint_is_rejected(key: str) -> None:
    space = {k: v for k, v in SPACE.items() if k != key}
    with pytest.raises(SpaceIdError, match="missing"):
        compute_space_id(space)


@pytest.mark.parametrize("key", ["wiring_policy_sha256", "time_policy_sha256"])
def test_a_different_policy_is_a_different_space(key: str) -> None:
    assert compute_space_id({**SPACE, key: "d" * 64}) != compute_space_id(SPACE)
    assert compute_space_id(SPACE) == compute_space_id(dict(SPACE))
