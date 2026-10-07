"""HEKBV3-BUILD001 差分2: diagnostics only (read-only; no new exp_hekbv3 writes; compose()
has no side effects). Same 48 sessions, same already-built index as 差分1/差分2.

1. Positive controls: (i) no exclude, (ii) exclude own session only (siblings kept),
   (iii) exclude own group/item (current). Real labels + group-shuffled, each with
   group-level 95% bootstrap CI.
2. T x top_m sweep: T in {1-NN,4-NN,16-NN} sq-dist median x {1, 0.3, 0.1}, top_m in
   {16, 64, 256}. Real labels only (scope decision, 1 line: the shuffle control is
   item 1's question, not the sweep's).
3. Weight concentration: mean (over 48 queries) of the top-3-sessions' summed weight.
4. Diff table vs the Mac-side P0 REPORT.md is printed from literal values read there
   (not recomputed) -- see the dict near the bottom of main().
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
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
SHUFFLE_SEED = 0
EPS_LOG = 1e-12


def _load_all_sessions() -> list[dict]:
    sessions = []
    for path in sorted(OBJECTS_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        ext = record["payload"]["ext"]["hekbv3"]
        if ext.get("space", {}).get("space_id") != SPACE_ID:
            continue
        sessions.append({
            "object_id": record["object_id"], "category": ext["labels"]["category"],
            "group": ext["labels"]["group"], "n_steps": ext["trajectory"]["n_steps"],
        })
    return sessions


_feature_cache: dict[tuple[str, int], np.ndarray] = {}


def _query_feature(object_id: str, token_index: int, meta: dict) -> np.ndarray:
    key = (object_id, token_index)
    if key in _feature_cache:
        return _feature_cache[key]
    record = json.loads((OBJECTS_DIR / f"{object_id}.json").read_text())
    ports_ref = record["payload"]["ext"]["hekbv3"]["ports"]
    port_ids = ports_ref["port_ids"]
    points, _ = trajectory_blob.decode_trajectory_blob(get_blob(BLOBS_DIR, ports_ref["blob_sha256"]))
    ports_series = np.asarray(points, dtype=np.float32)
    l = meta["l"]
    col_idx = [port_ids.index(p) for p in experience_index.ANCHOR_INDEPENDENT_10]
    window = ports_series[token_index - l + 1: token_index + 1, col_idx]
    feat = compose_mod.standardize_query(window.reshape(-1), meta["mean"], meta["std"], meta["kept_dims"])
    _feature_cache[key] = feat
    return feat


def knn_sq_dist_median(keys: np.ndarray, k: int, exclude_mask: np.ndarray, max_n: int = 2000, seed: int = 0) -> float:
    """Median of the squared distance to the k-th nearest neighbour, over `keys[exclude_mask]`."""
    kept = keys[exclude_mask]
    n = kept.shape[0]
    if n > max_n:
        rng = np.random.default_rng(seed)
        kept = kept[rng.choice(n, size=max_n, replace=False)]
        n = max_n
    kk = min(k, n - 1)
    if kk < 1:
        return 1.0
    sq_norms = np.sum(kept ** 2, axis=1)
    d2 = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (kept @ kept.T)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, kk - 1, axis=1)[:, kk - 1]
    return float(np.median(kth)) or 1.0


def exclude_sets(mode: str, own_object_id: str, own_group: str) -> tuple[set[str], set[str]]:
    if mode == "none":
        return set(), set()
    if mode == "own_session":
        return set(), {own_object_id}
    if mode == "own_group":
        return {own_group}, set()
    raise ValueError(mode)


def run_one(sessions, keys, entries, meta, query_idx, exclude_mode, temperature, top_m,
            shuffled_category_by_group, laplace_alpha_override=None):
    q = sessions[query_idx]
    own_group, own_object_id = q["group"], q["object_id"]
    true_cat = shuffled_category_by_group[own_group] if shuffled_category_by_group is not None else q["category"]
    query_token_index = q["n_steps"] - 1
    use_entries = entries
    if shuffled_category_by_group is not None:
        use_entries = [dict(e, category=shuffled_category_by_group[e["group"]]) for e in entries]

    excl_groups, excl_object_ids = exclude_sets(exclude_mode, own_object_id, own_group)
    query_feature = _query_feature(own_object_id, query_token_index, meta)

    result = compose_mod.compose(
        query_standardized=query_feature, query_object_id=own_object_id,
        query_token_index=query_token_index, keys=keys, entries=use_entries,
        objects_dir=OBJECTS_DIR, blobs_root=BLOBS_DIR, temperature=temperature, top_m=top_m,
        exclude_groups=excl_groups, exclude_object_ids=excl_object_ids,
        laplace_alpha=laplace_alpha_override,
    )
    logloss = -math.log(max(result.category_posterior.get(true_cat, 0.0), EPS_LOG))

    # distinct sessions in top_m, and top-3-sessions weight concentration
    per_session_w: dict[str, float] = {}
    for w, oid in zip(result.weights, result.neighbor_object_ids):
        per_session_w[oid] = per_session_w.get(oid, 0.0) + w
    n_distinct_sessions = len(per_session_w)
    top3_weight = sum(sorted(per_session_w.values(), reverse=True)[:3])

    return {
        "logloss": logloss, "n_eff": result.n_eff, "group": own_group,
        "n_distinct_sessions_in_top_m": n_distinct_sessions, "top3_session_weight": top3_weight,
    }


def bootstrap_ci(rows: list[dict], field: str, seed: int, n_rep: int) -> tuple[float, float, float]:
    by_group: dict[str, list[float]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r[field])
    distinct = sorted(by_group.keys())
    point = float(np.mean([v for r in rows for v in [r[field]]]))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_rep):
        sampled = rng.choice(distinct, size=len(distinct), replace=True)
        vals = [v for g in sampled for v in by_group[g]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def shuffled_category_map(sessions: list[dict], seed: int) -> dict[str, str]:
    cats_by_group: dict[str, set[str]] = {}
    for s in sessions:
        cats_by_group.setdefault(s["group"], set()).add(s["category"])
    assert all(len(cs) == 1 for cs in cats_by_group.values())
    groups = sorted(cats_by_group.keys())
    cats = [next(iter(cats_by_group[g])) for g in groups]
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(cats)
    return {g: c for g, c in zip(groups, shuffled)}


def main() -> None:
    sessions = _load_all_sessions()
    assert len(sessions) == 48
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)
    shuf_map = shuffled_category_map(sessions, SHUFFLE_SEED)

    print("=== 1. positive controls (top_m=64, T=16-NN x1 default per-query) ===")
    for mode in ("none", "own_session", "own_group"):
        for label, shuf in (("real", None), ("shuffled", shuf_map)):
            rows = []
            for i in range(48):
                q = sessions[i]
                excl_groups, excl_object_ids = exclude_sets(mode, q["object_id"], q["group"])
                keep_mask = np.array([
                    (e["group"] not in excl_groups) and (e["object_id"] not in excl_object_ids)
                    for e in entries
                ])
                t0 = knn_sq_dist_median(keys, 16, keep_mask)
                rows.append(run_one(sessions, keys, entries, meta, i, mode, t0, 64, shuf))
            mean, lo, hi = bootstrap_ci(rows, "logloss", BOOTSTRAP_SEED, N_BOOTSTRAP)
            print(f"  exclude={mode:11s} labels={label:8s} logloss mean={mean:.6f} 95%CI=[{lo:.6f},{hi:.6f}] "
                  f"(uniform ref={math.log(12):.6f})")

    print("\n=== 2. T x top_m sweep (exclude=own_group, real labels) ===")
    for k in (1, 4, 16):
        for mult in (1.0, 0.3, 0.1):
            for top_m in (16, 64, 256):
                rows = []
                for i in range(48):
                    q = sessions[i]
                    keep_mask = np.array([e["group"] != q["group"] for e in entries])
                    t0 = knn_sq_dist_median(keys, k, keep_mask) * mult
                    rows.append(run_one(sessions, keys, entries, meta, i, "own_group", t0, top_m, None))
                mean, lo, hi = bootstrap_ci(rows, "logloss", BOOTSTRAP_SEED, N_BOOTSTRAP)
                mean_neff = float(np.mean([r["n_eff"] for r in rows]))
                mean_distinct = float(np.mean([r["n_distinct_sessions_in_top_m"] for r in rows]))
                print(f"  T={k}-NNx{mult:<4} top_m={top_m:<4} logloss={mean:.4f} CI=[{lo:.4f},{hi:.4f}] "
                      f"n_eff={mean_neff:.2f} distinct_sessions={mean_distinct:.2f}")

    print("\n=== 3. weight concentration (exclude=own_group, T=16-NNx1, top_m=64, real labels) ===")
    rows3 = []
    for i in range(48):
        q = sessions[i]
        keep_mask = np.array([e["group"] != q["group"] for e in entries])
        t0 = knn_sq_dist_median(keys, 16, keep_mask)
        rows3.append(run_one(sessions, keys, entries, meta, i, "own_group", t0, 64, None))
    mean_top3 = float(np.mean([r["top3_session_weight"] for r in rows3]))
    print(f"  mean top-3-sessions summed weight (of 1.0 total) over 48 queries: {mean_top3:.6f}")

    print("\n=== 4. diff vs P0 (Mac EXP-HEKBv3/results/REPORT.md, params.json) ===")
    diffs = [
        ("prior smoothing", "alpha=1 (add-one) over training-fold categories", "alpha=1/K over post-exclude library categories"),
        ("ports/features", "17 ports (PORTS17, incl. C-group), L=8, d=136", "10 ports (anchor-independent, PORTS17 minus C7), L=8, d=80 (d_kept=63 after dropping zero-std dims)"),
        ("T", "T0 from TRAIN fold only (inner leave-bundle-out CV picks beta too)", "T0 from the post-exclude library itself (no separate train/val split)"),
        ("aggregation", "per-session mean over ALL that session's windows, then mean across sessions", "ONE window per session (last valid token only), mean across sessions"),
        ("bundle/library definition", "k=6 leave-bundle-out folds; library = ~25/30 bundles (~40/48 sessions)", "leave-one-group-out; library = 29/30 or 28/30 groups (46-47/48 sessions)"),
        ("transport", "parallel transport across base points (transport=True)", "no parallel transport (tangent-plane-projected local average, disclosed approximation)"),
        ("library size (windows)", "12,693 windows (U1)", "12,741 index entries"),
    ]
    for name, p0, this in diffs:
        print(f"  {name}: P0=[{p0}] / this=[{this}]")


if __name__ == "__main__":
    main()
