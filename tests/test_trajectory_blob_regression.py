"""SPEC-HEKB-REFACTOR-2026-v2.0 Step 1 acceptance test.

Acceptance criterion (spec section 6, Step 1 row): "MAC005 Check B の 22
ペアを保存→復元後、`mlba-core` で算出した d_manifold=0 および 128ステップが
完全に一致すること (回帰テスト通過)".

Reuses, rather than reimplements:
  - `mac005_data_loader.load_mac001_trajectories()` (EXP-BABY-MAC005) for the
    real MAC001 trajectories.
  - The real ground-truth d_manifold=0 pair classification from
    `EXP-SensOS001/evidence/edsr_task2_null_pairwise_distances.csv` (the same
    source EXP-BABY-MAC005's own Check B used -- see that experiment's
    run_mac005_preflight.py docstring for why turn_id grouping alone is
    NOT a reliable way to find these 22 pairs).
  - `mlba_core_bridge.pairwise_distance()` (EXP-BABY-MAC002) -- the real
    Rust FFI d_manifold implementation -- for the post-round-trip distance
    computation, per the acceptance criterion's own wording ("mlba-core で
    算出した").

Writes real TrajectoryBlob v2 files to a temp directory (not committed
anywhere), reads them back, and checks:
  1. Every one of the 22 real zero pairs still has 128 steps after restore.
  2. d_manifold(restored_a, restored_b), computed via the real mlba-core FFI,
     is exactly 0.0 for all 22 pairs.
  3. The restored float32 payload for each trajectory bit-exact-matches the
     original data cast to float32 (proves the blob format introduces no
     corruption of its own, independent of the d_manifold=0 check above).
"""
from __future__ import annotations

import csv
import struct
import sys
import tempfile
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path("/Users/tomonam3/Projects/sensos/experiments/EXP-BABY-MAC005")))
sys.path.insert(0, str(Path("/Users/tomonam3/Projects/sensos/experiments/EXP-BABY-MAC002")))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mac005_data_loader import load_mac001_trajectories  # noqa: E402
from mlba_core_bridge import pairwise_distance  # noqa: E402
from storage.trajectory_blob import write_trajectory, read_trajectory  # noqa: E402

MAC001_GROUND_TRUTH_CSV = Path(
    "/Users/tomonam3/Projects/sensos/experiments/EXP-SensOS001/evidence/edsr_task2_null_pairwise_distances.csv"
)


def _load_zero_pair_keys() -> set[frozenset]:
    keys = set()
    with MAC001_GROUND_TRUTH_CSV.open() as f:
        for row in csv.DictReader(f):
            if float(row["d_manifold"]) == 0.0:
                keys.add(frozenset((row["input_a"], row["input_b"])))
    return keys


def _cast_f32(points: list[list[float]]) -> list[list[float]]:
    return [
        [struct.unpack("<f", struct.pack("<f", v))[0] for v in row]
        for row in points
    ]


def main() -> int:
    mac001 = load_mac001_trajectories()
    zero_pair_keys = _load_zero_pair_keys()
    pairs = [
        (a, b) for a, b in combinations(mac001, 2)
        if frozenset((a["input_id"], b["input_id"])) in zero_pair_keys
    ]
    print(f"Loaded {len(mac001)} real MAC001 trajectories; {len(pairs)} real zero-distance pairs (expected 22).")
    assert len(pairs) == 22, f"expected 22 zero pairs, found {len(pairs)}"

    failures = []
    with tempfile.TemporaryDirectory(prefix="hekb_trajectoryblob_regression_") as tmpdir:
        trajectories_root = Path(tmpdir)
        for a, b in pairs:
            header_a = write_trajectory(trajectories_root, a["points"])
            header_b = write_trajectory(trajectories_root, b["points"])
            restored_a = read_trajectory(trajectories_root, header_a.trajectory_id_hex)
            restored_b = read_trajectory(trajectories_root, header_b.trajectory_id_hex)

            if len(restored_a) != 128 or len(restored_b) != 128:
                failures.append((a["input_id"], b["input_id"], "step count != 128 after restore"))
                continue

            expected_a = _cast_f32(a["points"])
            expected_b = _cast_f32(b["points"])
            if restored_a != expected_a or restored_b != expected_b:
                failures.append((a["input_id"], b["input_id"], "restored payload != original data cast to float32"))
                continue

            d_m = pairwise_distance(restored_a, restored_b)
            if d_m != 0.0:
                failures.append((a["input_id"], b["input_id"], f"d_manifold(restored) = {d_m} != 0.0"))

    print(f"Pairs checked: {len(pairs)}. Failures: {len(failures)}.")
    for f in failures:
        print("  FAIL:", f)

    if failures:
        print("\nACCEPTANCE CRITERION: FAIL")
        return 1
    print("\nACCEPTANCE CRITERION: PASS -- all 22 pairs round-trip through TrajectoryBlob v2 with d_manifold=0.0 and 128 steps intact.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
