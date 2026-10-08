"""HEKBV3-BUILD002 続き §4 陽性対照 (positive control): for each query session, adds that
session's OWN bundle back into the sampled library (config.positive_control semantics,
scripts/eval_curve_boardrecords.py's session_logloss/run_n_curve already define this), then
measures log-loss. If the retrieval pipeline works at all, the query should be able to find
itself and log-loss should collapse close to 0 -- this is a sanity check that memory/retrieval
is functioning, not a comparison against real/shuffle. Uses the SAME fast prefiltered-library
path as the main n-curve run (one prefilter per (resample, query_group), since each query_group
adds a different group back in -- the shared-filter trick from run_n_curve_real_and_shuffle does
not apply here). Checkpoints after every resample. No interpretation -- numbers and settings
only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from index import experience_index
import eval_curve_boardrecords as ecb


def build_query_sessions(entries, keys, query_groups):
    by_group = {}
    for i, e in enumerate(entries):
        by_group.setdefault(e["group"], []).append(i)
    out = {}
    for g in query_groups:
        out[g] = [(keys[i], entries[i]["object_id"], entries[i]["token_index"], entries[i]["category"])
                   for i in by_group.get(g, [])]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-root", required=True, type=Path)
    parser.add_argument("--space-id", required=True)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--n-resamples", type=int, default=5)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    t_start = time.time()
    config = ecb.PreregisteredConfig(positive_control=True)
    print(f"[run_positive_control] config={config} n={args.n} n_resamples={args.n_resamples}", flush=True)

    keys, entries, meta = experience_index.load_index(args.index_root, args.space_id)
    print(f"[run_positive_control] n_entries={len(entries)} d_kept={keys.shape[1]} "
          f"elapsed_s={time.time() - t_start:.1f}", flush=True)

    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=args.holdout_frac, seed=config.seed)
    if args.n > len(library_pool):
        raise ValueError(f"n={args.n} exceeds library pool size {len(library_pool)}")
    print(f"[run_positive_control] n_groups={len(all_groups)} library_pool={len(library_pool)} "
          f"query_groups={len(query_groups)}", flush=True)

    query_sessions = build_query_sessions(entries, keys, query_groups)
    n_query_sessions = len({oid for rows in query_sessions.values() for (_, oid, _, _) in rows})
    n_query_windows = sum(len(rows) for rows in query_sessions.values())
    print(f"[run_positive_control] n_query_sessions={n_query_sessions} n_query_windows={n_query_windows}", flush=True)

    pool = sorted(library_pool)
    rows: list[dict] = []

    def checkpoint(tag: str) -> None:
        payload = {
            "n": args.n, "n_resamples_done": len(rows) // max(len(query_groups), 1),
            "n_resamples_target": args.n_resamples,
            "holdout_frac": args.holdout_frac, "seed": config.seed,
            "n_groups": len(all_groups), "library_pool_size": len(library_pool),
            "query_groups_size": len(query_groups), "n_query_sessions": n_query_sessions,
            "n_query_windows": n_query_windows,
            "rows": rows, "status": tag, "elapsed_s": time.time() - t_start,
        }
        args.out.write_text(json.dumps(payload))

    for r in range(args.n_resamples):
        t_r = time.time()
        resample_rng = np.random.default_rng((config.seed, args.n, r))
        sampled = set(resample_rng.choice(pool, size=args.n, replace=False).tolist())
        for qgroup, qrows in query_sessions.items():
            keep_groups = sampled | {qgroup}
            kept_keys, kept_entries, t0, library_categories = ecb._prefiltered_library(
                entries, keys, keep_groups, config)
            ll = ecb._session_logloss_prefiltered(kept_keys, kept_entries, t0, library_categories, qrows, config)
            rows.append({"group": qgroup, "resample": r, "logloss": ll})
        checkpoint(tag=f"resample={r + 1}/{args.n_resamples} done")
        print(f"[run_positive_control] resample={r + 1}/{args.n_resamples} "
              f"elapsed_r_s={time.time() - t_r:.1f} total_elapsed_s={time.time() - t_start:.1f}", flush=True)

    checkpoint(tag="all resamples complete")
    print(f"[run_positive_control] DONE total_elapsed_s={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
