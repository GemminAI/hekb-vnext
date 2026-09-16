"""HEKB vNext: `/trajectories/recalibrate` job logic (SPEC-HEKB-REFACTOR-
2026-v2.0 section 4.2 / section 5 Step 5).

Reuses, not reimplements: `mac005_quantizer.UltrametricQuantizer`
(EXP-BABY-MAC005, `/Users/tomonam3/Projects/sensos/experiments/EXP-BABY-
MAC005/mac005_quantizer.py`) for PCA + hierarchical binary encoding and the
common-prefix-depth `k_match` metric -- the same disclosed quantizer design
already built and regression-tested against real MAC001/MAC002 data. This
module does not define a second quantizer.

Design choices disclosed here (the spec specifies the API shape and the
response fields, not the exact algorithm for turning a 128x256 trajectory
into ONE semantic address, or what "recalibrate" fits its PCA on):

1. **Trajectory -> single object vector**: a semantic address identifies one
   memory *object* (spec section 3), not one of its 128 per-step hidden
   states. This module represents each stored trajectory by the mean of its
   128 step-vectors (a 256-dim summary), and addresses that. Per-step
   addressing is what EXP-BABY-MAC005's `t_exit`/`k_match` measurements use
   instead, for a different purpose (temporal exit detection within one
   trajectory pair) -- not conflated with this per-object index.
2. **What PCA is fit on**: "recalibrate" (re-*calibrate*) is read literally
   here -- each call fits a fresh PCA transform on the full set of object
   vectors currently in the trajectory store at call time, rather than
   reusing one external frozen pool (e.g. MAC002 Control Phase, as
   EXP-BABY-MAC005's Arm 1a Freeze Protocol does for a *different*,
   leak-sensitive experiment). This generic index has no single designated
   "training-only" corpus, so self-referential fit-and-apply is the
   disclosed choice.
3. **effective_depth**: mean pairwise common-prefix depth (`k_match`) across
   all object pairs under the new codebook -- the exact same metric
   EXP-BABY-MAC005's Check A already validated as meaningful (not a new,
   separately-invented notion of "depth").
4. **max_bucket_cardinality**: objects are bucketed by their address prefix
   truncated to `k_target` digits; this is the largest such bucket's size.
5. **address_churn_rate**: of the objects that already had SOME address on
   file (under any prior codebook_id), the fraction whose new address
   differs from their most recent previous one. Objects with no prior
   address are excluded from both numerator and denominator (a first-ever
   addressing isn't "churn"). If no object has prior history, churn_rate is
   reported as 0.0 with `"churn_rate_defined": false` alongside it, rather
   than a silently misleading 0.0 with no caveat.
"""
from __future__ import annotations

import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, "/Users/tomonam3/Projects/sensos/experiments/EXP-BABY-MAC005")
from mac005_quantizer import UltrametricQuantizer, k_match  # noqa: E402

from storage.trajectory_blob import trajectory_path  # noqa: E402
from index.semantic_index import connect, get_address, latest_address_any_codebook, upsert_address  # noqa: E402


def iter_stored_trajectory_ids(trajectories_root: Path) -> list[str]:
    ids = []
    if not trajectories_root.exists():
        return ids
    for path in trajectories_root.glob("*/*/trj_*.bin.zst"):
        name = path.name
        assert name.startswith("trj_") and name.endswith(".bin.zst")
        ids.append(name[len("trj_"):-len(".bin.zst")])
    return ids


@dataclass
class RecalibrationJob:
    job_id: str
    status: str = "PROCESSING"  # PROCESSING | COMPLETED | FAILED
    result: dict[str, Any] | None = None
    error: str | None = None


_JOBS: dict[str, RecalibrationJob] = {}


def new_job_id() -> str:
    return f"job_recalib_{time.strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"


def get_job(job_id: str) -> RecalibrationJob | None:
    return _JOBS.get(job_id)


async def run_recalibration_job(
    job_id: str,
    trajectories_root: Path,
    index_db_path: Path,
    n_pca: int,
    gamma: float,
    k_target: int,
    codebook_id: str,
    dry_run: bool,
) -> None:
    job = RecalibrationJob(job_id=job_id)
    _JOBS[job_id] = job
    try:
        object_ids = iter_stored_trajectory_ids(trajectories_root)
        object_vectors: dict[str, np.ndarray] = {}
        from storage.trajectory_blob import read_trajectory
        for object_id in object_ids:
            points = read_trajectory(trajectories_root, object_id)
            object_vectors[object_id] = np.asarray(points, dtype=np.float64).mean(axis=0)

        if len(object_vectors) < 2:
            job.status = "COMPLETED"
            job.result = {
                "job_id": job_id, "status": "COMPLETED",
                "processed_trajectories": len(object_vectors),
                "address_churn_rate": 0.0, "churn_rate_defined": False,
                "max_bucket_cardinality": len(object_vectors),
                "effective_depth": 0.0,
                "note": "fewer than 2 trajectories in store; no pairwise depth or bucket statistics possible",
            }
            return

        pool = np.stack(list(object_vectors.values()))
        quantizer = UltrametricQuantizer.fit(pool, n_pca=n_pca, gamma=gamma)

        addresses: dict[str, tuple[int, ...]] = {
            object_id: quantizer.encode(vec) for object_id, vec in object_vectors.items()
        }

        conn = connect(index_db_path)
        try:
            churned, had_prior = 0, 0
            for object_id, address in addresses.items():
                prior = latest_address_any_codebook(conn, object_id)
                if prior is not None:
                    had_prior += 1
                    _prior_codebook, prior_address = prior
                    if prior_address != address:
                        churned += 1
                if not dry_run:
                    upsert_address(conn, object_id, codebook_id, address)
            if not dry_run:
                conn.commit()
        finally:
            conn.close()

        buckets: dict[tuple[int, ...], int] = {}
        for address in addresses.values():
            prefix = address[:k_target]
            buckets[prefix] = buckets.get(prefix, 0) + 1
        max_bucket_cardinality = max(buckets.values())

        # Per-level bucket histogram (this task's explicit request: "各階層
        # k におけるバケット数のヒストグラム", not just the single k_target
        # slice above). For every level k = 1..n_pca, group addresses by
        # their length-k prefix and record how many distinct buckets exist
        # at that level and the largest one -- typically n_buckets roughly
        # doubles per level until it saturates at population size, and
        # max_bucket_size shrinks correspondingly.
        bucket_histogram_per_level: dict[str, dict[str, int]] = {}
        addr_values = list(addresses.values())
        max_depth = len(addr_values[0]) if addr_values else 0
        for k in range(1, max_depth + 1):
            level_buckets: dict[tuple[int, ...], int] = {}
            for address in addr_values:
                prefix = address[:k]
                level_buckets[prefix] = level_buckets.get(prefix, 0) + 1
            bucket_histogram_per_level[str(k)] = {
                "n_buckets": len(level_buckets),
                "max_bucket_size": max(level_buckets.values()),
            }

        addr_list = list(addresses.values())
        depths = [
            k_match(addr_list[i], addr_list[j])
            for i in range(len(addr_list)) for j in range(i + 1, len(addr_list))
        ]
        effective_depth = float(np.mean(depths)) if depths else 0.0

        job.status = "COMPLETED"
        job.result = {
            "job_id": job_id,
            "status": "COMPLETED",
            "dry_run": dry_run,
            "codebook_id": codebook_id,
            "processed_trajectories": len(addresses),
            "address_churn_rate": (churned / had_prior) if had_prior else 0.0,
            "churn_rate_defined": had_prior > 0,
            "max_bucket_cardinality": max_bucket_cardinality,
            "effective_depth": round(effective_depth, 4),
            "n_buckets_at_k_target": len(buckets),
            "bucket_histogram_per_level": bucket_histogram_per_level,
        }
    except Exception as exc:  # noqa: BLE001 -- surfaced via job status, not a crashed background task
        job.status = "FAILED"
        job.error = str(exc)
