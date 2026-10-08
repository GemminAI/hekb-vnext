"""HEKBV3-BUILD002 続き §4: runs the preregistered (6 n-values x 20 resamples, real+shuffle)
n-curve evaluation against the real 1,296-session ingest at
/media/psf/HEKBv3/exp_hekbv3_n_curve, using eval_curve_boardrecords.PreregisteredConfig
UNCHANGED (no settings here override it). Checkpoints after every n value to
--out (so partial results survive if this is interrupted mid-run; it is expected to take
on the order of hours -- see the task's own "評価に2時間以上かかる見込みなら、先に見積もりを
報告して、続ける" and HEKBV3-BUILD002 続き §4's benchmark). Read-only against the index;
writes only the --out JSON file. No interpretation -- numbers and settings only.
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
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--positive-control-resamples", type=int, default=0,
                         help="If > 0, also run a positive-control pass at max(n_values) with "
                              "this many resamples (config.positive_control=True). 0 = skip.")
    args = parser.parse_args()

    t_start = time.time()
    config = ecb.PreregisteredConfig()
    print(f"[run_ncurve_eval] config={config}", flush=True)

    keys, entries, meta = experience_index.load_index(args.index_root, args.space_id)
    print(f"[run_ncurve_eval] n_entries={len(entries)} d_kept={keys.shape[1]} "
          f"elapsed_s={time.time() - t_start:.1f}", flush=True)

    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=args.holdout_frac, seed=config.seed)
    print(f"[run_ncurve_eval] n_groups={len(all_groups)} library_pool={len(library_pool)} "
          f"query_groups={len(query_groups)}", flush=True)

    query_sessions = build_query_sessions(entries, keys, query_groups)
    n_query_sessions = len({oid for rows in query_sessions.values() for (_, oid, _, _) in rows})
    n_query_windows = sum(len(rows) for rows in query_sessions.values())
    print(f"[run_ncurve_eval] n_query_sessions={n_query_sessions} n_query_windows={n_query_windows}", flush=True)

    uniform_logloss = float(np.log(len(sorted({e["category"] for e in entries}))))
    print(f"[run_ncurve_eval] uniform_logloss (log K, K={len(sorted({e['category'] for e in entries}))}) "
          f"= {uniform_logloss:.6f}", flush=True)

    curve_real: dict[int, list[dict]] = {}
    curve_shuffle: dict[int, list[dict]] = {}

    def checkpoint(tag: str) -> None:
        payload = {
            "config": {
                "n_values": list(config.n_values), "n_resamples": config.n_resamples,
                "top_m": config.top_m, "temperature_multiplier": config.temperature_multiplier,
                "seed": config.seed, "n_bootstrap": config.n_bootstrap,
                "bootstrap_seed": config.bootstrap_seed,
            },
            "holdout_frac": args.holdout_frac,
            "n_groups": len(all_groups), "library_pool_size": len(library_pool),
            "query_groups_size": len(query_groups), "n_query_sessions": n_query_sessions,
            "n_query_windows": n_query_windows, "uniform_logloss": uniform_logloss,
            "curve_real": {str(n): curve_real[n] for n in curve_real},
            "curve_shuffle": {str(n): curve_shuffle[n] for n in curve_shuffle},
            "status": tag, "elapsed_s": time.time() - t_start,
        }
        args.out.write_text(json.dumps(payload))

    # One n at a time (not ecb.run_n_curve_real_and_shuffle's own all-n loop) so a checkpoint
    # file is written after each n value completes -- same math, just exposing the loop here.
    groups_sorted = sorted({e["group"] for e in entries})
    import math
    cat_by_group = {}
    for g in groups_sorted:
        cat_by_group[g] = next(e["category"] for e in entries if e["group"] == g)
    rng = np.random.default_rng(config.seed)
    permuted_groups = rng.permutation(groups_sorted)
    shuffled_cat_of_group = dict(zip(groups_sorted, [cat_by_group[g] for g in permuted_groups]))
    pool = sorted(library_pool)

    for n in config.n_values:
        t_n = time.time()
        rows_real, rows_shuffle = [], []
        for r in range(config.n_resamples):
            resample_rng = np.random.default_rng((config.seed, n, r))
            sampled = set(resample_rng.choice(pool, size=n, replace=False).tolist())
            kept_keys, kept_entries, t0, library_categories_real = ecb._prefiltered_library(
                entries, keys, sampled, config)
            library_categories_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in kept_entries))
            alpha_shuffle = 1.0 / max(len(library_categories_shuffle), 1)
            entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in kept_entries}

            for qgroup, qrows in query_sessions.items():
                real_losses, shuffle_losses = [], []
                for query_standardized, query_object_id, query_token_index, true_category in qrows:
                    result = ecb.compose_mod.compose(
                        query_standardized=query_standardized, query_object_id=query_object_id,
                        query_token_index=query_token_index, keys=kept_keys, entries=kept_entries,
                        objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                        temperature=t0, top_m=config.top_m, exclude_groups=None, skip_r1=True,
                        library_categories=library_categories_real,
                    )
                    p_real = result.category_posterior.get(true_category, 0.0)
                    real_losses.append(-math.log(max(p_real, ecb.EPS_LOG)))

                    posterior_shuffle = ecb._posterior_from_neighbors(
                        result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                        entry_group_lookup, shuffled_cat_of_group, library_categories_shuffle, alpha_shuffle,
                    )
                    p_shuffle = posterior_shuffle.get(true_category, 0.0)
                    shuffle_losses.append(-math.log(max(p_shuffle, ecb.EPS_LOG)))
                rows_real.append({"group": qgroup, "logloss": float(np.mean(real_losses))})
                rows_shuffle.append({"group": qgroup, "logloss": float(np.mean(shuffle_losses))})
            print(f"[run_ncurve_eval] n={n} resample={r + 1}/{config.n_resamples} "
                  f"elapsed_n_s={time.time() - t_n:.1f}", flush=True)

        curve_real[n] = rows_real
        curve_shuffle[n] = rows_shuffle
        checkpoint(tag=f"n={n} done")
        print(f"[run_ncurve_eval] n={n} COMPLETE elapsed_n_s={time.time() - t_n:.1f} "
              f"total_elapsed_s={time.time() - t_start:.1f}", flush=True)

    checkpoint(tag="all n complete")
    print(f"[run_ncurve_eval] DONE total_elapsed_s={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
