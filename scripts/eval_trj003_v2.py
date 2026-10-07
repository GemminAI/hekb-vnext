"""HEKBV3-BUILD001 差分: re-evaluate the TRJ003 48-session leave-one-group-out log-loss
with corrected R3 smoothing (alpha=1/K over ALL K library categories, not just top_m's).
No new writes to exp_hekbv3 -- reads the already-built index (local disk) and the already-
posted blobs/objects directly; `compose()` itself has no side effects (H7). The original
(invalid) E2E result file is left untouched (D-18); this writes a new, separate file.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from index import compose as compose_mod  # noqa: E402
from index import experience_index  # noqa: E402
from storage import trajectory_blob  # noqa: E402
from storage.blobs import get_blob  # noqa: E402

EXP_DATA_DIR = Path("/media/psf/HEKBv3/exp_hekbv3/experience")
OBJECTS_DIR = EXP_DATA_DIR / "objects"
BLOBS_DIR = EXP_DATA_DIR / "blobs"
INDEX_ROOT = Path.home() / ".hekb-v3-e2e-app-support" / "index"
SPACE_ID = "5bb9c6ba4febecaabc4b510cdd4ca47222579d678a835879d052109f75d3b4e7"
TOP_M = 64
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
SHUFFLE_SEED = 0
EPS_LOG = 1e-12  # numeric floor only for log(); smoothing itself has no separate epsilon


def _load_all_sessions() -> list[dict]:
    """Rebuild the per-session (object_id, category, group, n_steps) list by scanning the
    already-posted objects -- no re-ingestion, just reading what's already there."""
    sessions = []
    for path in sorted(OBJECTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        ext = record["payload"]["ext"]["hekbv3"]
        if ext.get("space", {}).get("space_id") != SPACE_ID:
            continue
        sessions.append({
            "object_id": record["object_id"],
            "category": ext["labels"]["category"],
            "group": ext["labels"]["group"],
            "n_steps": ext["trajectory"]["n_steps"],
        })
    return sessions


def _query_feature(object_id: str, token_index: int, meta: dict) -> np.ndarray:
    record = json.loads((OBJECTS_DIR / f"{object_id}.json").read_text())
    ports_ref = record["payload"]["ext"]["hekbv3"]["ports"]
    port_ids = ports_ref["port_ids"]
    points, _header = trajectory_blob.decode_trajectory_blob(get_blob(BLOBS_DIR, ports_ref["blob_sha256"]))
    ports_series = np.asarray(points, dtype=np.float32)
    l = meta["l"]
    col_idx = [port_ids.index(p) for p in experience_index.ANCHOR_INDEPENDENT_10]
    window = ports_series[token_index - l + 1: token_index + 1, col_idx]
    raw_feature = window.reshape(-1)
    return compose_mod.standardize_query(raw_feature, meta["mean"], meta["std"], meta["kept_dims"])


def laplace_posterior(category_counts_weighted: dict[str, float], all_categories: list[str], alpha: float) -> dict[str, float]:
    k = max(len(all_categories), 1)
    total_w = sum(category_counts_weighted.get(c, 0.0) for c in all_categories)
    return {c: (category_counts_weighted.get(c, 0.0) + alpha) / (total_w + k * alpha) for c in all_categories}


def run_fold(sessions: list[dict], keys: np.ndarray, entries: list[dict], meta: dict,
             query_idx: int, shuffled_category_by_group: dict[str, str] | None) -> dict:
    q = sessions[query_idx]
    own_group = q["group"]
    true_cat = shuffled_category_by_group[own_group] if shuffled_category_by_group is not None else q["category"]
    query_token_index = q["n_steps"] - 1

    use_entries = entries
    if shuffled_category_by_group is not None:
        use_entries = [dict(e, category=shuffled_category_by_group[e["group"]]) for e in entries]

    query_feature = _query_feature(q["object_id"], query_token_index, meta)
    temperature = _default_temperature_for(keys, use_entries, own_group)

    result = compose_mod.compose(
        query_standardized=query_feature, query_object_id=q["object_id"],
        query_token_index=query_token_index, keys=keys, entries=use_entries,
        objects_dir=OBJECTS_DIR, blobs_root=BLOBS_DIR, temperature=temperature,
        top_m=TOP_M, exclude_groups={own_group},
    )

    # library categories post-exclude, at the SESSION level (matches compose()'s own K
    # over kept_entries -- every token inherits its session's category, so the distinct
    # set is identical whichever granularity counts it).
    other_sessions = [s for s in sessions if s["group"] != own_group]
    other_cats = [shuffled_category_by_group[s["group"]] if shuffled_category_by_group is not None else s["category"]
                  for s in other_sessions]
    all_cats = sorted(set(other_cats))
    k_cat = max(len(all_cats), 1)
    alpha = 1.0 / k_cat

    # kernel posterior: already Laplace-smoothed the same way inside compose() (same
    # all_cats/K/alpha, since compose() derives K from kept_entries post-exclude).
    kernel_p = result.category_posterior
    kernel_logloss = -math.log(max(kernel_p.get(true_cat, 0.0), EPS_LOG))
    top_m_cat_weight = sum(w for w, oid in zip(result.weights, result.neighbor_object_ids)
                           if (shuffled_category_by_group[_group_of(entries, oid)] if shuffled_category_by_group is not None
                               else _category_of(entries, oid)) == true_cat)
    hit_in_top_m = top_m_cat_weight > 0.0

    # prior: session-level frequency over the post-exclude library, same K/alpha
    prior_counts = {c: float(other_cats.count(c)) for c in all_cats}
    prior_p = laplace_posterior(prior_counts, all_cats, alpha)
    prior_logloss = -math.log(max(prior_p.get(true_cat, 0.0), EPS_LOG))

    # uniform: 1/K over the same K categories (no smoothing needed, already proper)
    uniform_logloss = math.log(k_cat)

    true_cat_absent_from_library = true_cat not in all_cats

    return {
        "object_id": q["object_id"], "group": own_group, "true_category": true_cat,
        "k_categories": k_cat, "alpha": alpha,
        "kernel_logloss": kernel_logloss, "prior_logloss": prior_logloss, "uniform_logloss": uniform_logloss,
        "true_cat_absent_from_library": true_cat_absent_from_library,
        "top_m_has_true_category": hit_in_top_m,
        "n_eff": result.n_eff, "temperature_used": temperature,
    }


def _category_of(entries: list[dict], object_id: str) -> str:
    for e in entries:
        if e["object_id"] == object_id:
            return e["category"]
    raise KeyError(object_id)


def _group_of(entries: list[dict], object_id: str) -> str:
    for e in entries:
        if e["object_id"] == object_id:
            return e["group"]
    raise KeyError(object_id)


_temp_cache: dict[str, float] = {}


def _default_temperature_for(keys: np.ndarray, entries: list[dict], own_group: str) -> float:
    """Same kNN-bandwidth default as store_service.py's _default_temperature, computed
    directly here (no HTTP round trip) over the post-exclude library -- cached per exclude
    group since many queries can share an exclude set (not in this 1-query-per-session
    design, but kept for clarity/reuse)."""
    if own_group in _temp_cache:
        return _temp_cache[own_group]
    keep_mask = np.array([e["group"] != own_group for e in entries])
    kept = keys[keep_mask]
    n = kept.shape[0]
    max_n = 2000
    if n > max_n:
        rng = np.random.default_rng(0)
        kept = kept[rng.choice(n, size=max_n, replace=False)]
        n = max_n
    kk = min(16, n - 1)
    sq_norms = np.sum(kept ** 2, axis=1)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (kept @ kept.T)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, kk - 1, axis=1)[:, kk - 1]
    t0 = float(np.median(kth)) or 1.0
    _temp_cache[own_group] = t0
    return t0


def bootstrap_ci(per_query: list[dict], field: str, seed: int, n_rep: int) -> tuple[float, float, float]:
    by_group: dict[str, list[float]] = {}
    for row in per_query:
        by_group.setdefault(row["group"], []).append(row[field])
    distinct_groups = sorted(by_group.keys())
    rng = np.random.default_rng(seed)
    means = []
    all_vals = [v for row in per_query for v in [row[field]]]
    point_mean = float(np.mean(all_vals))
    for _ in range(n_rep):
        sampled_groups = rng.choice(distinct_groups, size=len(distinct_groups), replace=True)
        vals = [v for g in sampled_groups for v in by_group[g]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point_mean, float(lo), float(hi)


def main() -> None:
    sessions = _load_all_sessions()
    assert len(sessions) == 48, f"expected 48 sessions in exp_hekbv3, found {len(sessions)}"
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)

    print(f"n_sessions={len(sessions)} n_index_entries={len(entries)} d_kept={meta['d_kept']}")
    cat_counts = {}
    for s in sessions:
        cat_counts[s["category"]] = cat_counts.get(s["category"], 0) + 1
    print(f"category counts: {cat_counts}")

    # main run (true labels)
    per_query_main = [run_fold(sessions, keys, entries, meta, i, None) for i in range(48)]

    n_absent = sum(1 for r in per_query_main if r["true_cat_absent_from_library"])
    n_no_hit = sum(1 for r in per_query_main if not r["top_m_has_true_category"])
    print(f"queries where true category is entirely absent from the post-exclude library: {n_absent}")
    print(f"queries where top_m contained zero entries of the true category: {n_no_hit}/48")

    for label, field in [("kernel", "kernel_logloss"), ("prior", "prior_logloss"), ("uniform", "uniform_logloss")]:
        mean, lo, hi = bootstrap_ci(per_query_main, field, BOOTSTRAP_SEED, N_BOOTSTRAP)
        print(f"{label} log-loss: mean={mean:.6f} 95% group-bootstrap CI=[{lo:.6f}, {hi:.6f}] (n_rep={N_BOOTSTRAP})")

    # negative control: shuffle category labels at the GROUP level (fixed seed)
    distinct_groups = sorted(set(s["group"] for s in sessions))
    cats_by_group: dict[str, set[str]] = {}
    for s in sessions:
        cats_by_group.setdefault(s["group"], set()).add(s["category"])
    mixed_groups = {g: cs for g, cs in cats_by_group.items() if len(cs) > 1}
    assert not mixed_groups, f"group-level category shuffle assumes one category per group; violated by {mixed_groups}"
    group_to_category = {g: next(iter(cs)) for g, cs in cats_by_group.items()}
    cats_list = [group_to_category[g] for g in distinct_groups]
    rng = np.random.default_rng(SHUFFLE_SEED)
    shuffled_cats_list = rng.permutation(cats_list)
    shuffled_category_by_group = {g: c for g, c in zip(distinct_groups, shuffled_cats_list)}

    per_query_shuffled = [run_fold(sessions, keys, entries, meta, i, shuffled_category_by_group) for i in range(48)]
    print("--- negative control (category labels shuffled by group) ---")
    for label, field in [("kernel", "kernel_logloss"), ("prior", "prior_logloss"), ("uniform", "uniform_logloss")]:
        mean, lo, hi = bootstrap_ci(per_query_shuffled, field, BOOTSTRAP_SEED, N_BOOTSTRAP)
        print(f"{label} log-loss (shuffled): mean={mean:.6f} 95% group-bootstrap CI=[{lo:.6f}, {hi:.6f}]")

    out_path = EXP_DATA_DIR.parent.parent / "exp_hekbv3_e2e_result_v2.json"
    out_path.write_text(json.dumps({
        "space_id": SPACE_ID, "top_m": TOP_M, "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED, "shuffle_seed": SHUFFLE_SEED,
        "category_counts": cat_counts,
        "n_true_cat_absent_from_library": n_absent,
        "n_top_m_no_true_category_hit": n_no_hit,
        "main": per_query_main, "negative_control_shuffled": per_query_shuffled,
        "shuffled_category_by_group": shuffled_category_by_group,
    }, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
