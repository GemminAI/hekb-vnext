"""HEKBV3-BUILD001 差分3: P0 reproduction / step-by-step attribution (read-only; no new
exp_hekbv3 writes; compose() has no side effects). Same 48 sessions as 差分1/2.

Isolates five axes that differ between this build's category (R3) log-loss evaluation and
the Mac-side P0 reference (EXP-HEKBv3/code/{engine,run_exp,geom}.py, run seed 20261004):

  1. ports/features: 17 ports (P0's PORTS17, d=136) vs 10 ports (this build's
     anchor-independent set, d=80/63 kept) -- built in-memory here, standardized globally
     over all 48 sessions' windows (NOT P0's per-outer-fold train-only standardization;
     disclosed simplification, noted in the report, not fixed here).
  2. aggregation: per-session mean log-loss over ALL that session's valid windows (P0) vs
     a single last-window log-loss (this build's current default).
  3. posterior formula: P0's n_eff-scaled add-one Laplace ("p0_ne_add_one") vs this build's
     alpha=1/K-of-weights Laplace ("alpha_over_k", the shipped default, unchanged).
  4. library/bundle split: P0's seed=20261004 k=6 leave-bundle-out assignment (this script
     reproduces assign_folds() exactly) vs this build's leave-one-group-out.
  5. transport: NOT applicable to R3 -- P0's own engine.py R3 block (lines ~402-419) never
     calls blend_sphere/transport; transport only affects R1 (trajectory-direction cosine),
     not the category posterior. Reported as a fact, not run as a step.

Also runs: a positive-control T-sweep for exclude=none (does the kernel beat uniform,
and at what T does the self-match weight exceed 0.5), and a group-level category-label
shuffle at each step of the P0->current sequence.

No interpretation. Numbers and settings only.
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
P0_FOLD_SEED = 20261004  # EXP-HEKBv3 code/engine.py SEED, used verbatim by assign_folds()
P0_K_FOLDS = 6
PORTS17 = ["P01", "P03", "P06", "P09", "P10", "P13", "P16", "P18", "P19", "P25", "P26", "P28", "P30", "P31",
           "P35", "P36", "P38"]
L = 8
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


_ports_cache: dict[str, tuple[np.ndarray, list[str]]] = {}


def _raw_ports_series(object_id: str) -> tuple[np.ndarray, list[str]]:
    if object_id in _ports_cache:
        return _ports_cache[object_id]
    record = json.loads((OBJECTS_DIR / f"{object_id}.json").read_text())
    ports_ref = record["payload"]["ext"]["hekbv3"]["ports"]
    port_ids = ports_ref["port_ids"]
    points, _ = trajectory_blob.decode_trajectory_blob(get_blob(BLOBS_DIR, ports_ref["blob_sha256"]))
    series = np.asarray(points, dtype=np.float64)
    _ports_cache[object_id] = (series, port_ids)
    return series, port_ids


_windows_cache: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]] = {}


def _windows_for_session(object_id: str, ports: list[str], n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """All raw (unstandardized) L-token windows for t in [L-1, full_n_steps-1] (computed once
    per (object_id, ports) and cached; n_steps here only bounds the returned slice)."""
    cache_key = (object_id, tuple(ports))
    if cache_key not in _windows_cache:
        series, port_ids = _raw_ports_series(object_id)
        col_idx = [port_ids.index(p) for p in ports]
        sub = series[:, col_idx]
        full_n = series.shape[0]
        feats, idxs = [], []
        for t in range(L - 1, full_n):
            feats.append(sub[t - L + 1: t + 1].reshape(-1))
            idxs.append(t)
        _windows_cache[cache_key] = (np.stack(feats), np.array(idxs))
    feats_all, idxs_all = _windows_cache[cache_key]
    mask = idxs_all < n_steps
    return feats_all[mask], idxs_all[mask]


def build_library_17(sessions: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """In-memory PORTS17/d=136 library, standardized globally over all 48 sessions' windows
    (every valid window, matching the existing 10-port index's "all windows" library scope)."""
    all_raw, all_entries = [], []
    for s in sessions:
        feats, idxs = _windows_for_session(s["object_id"], PORTS17, s["n_steps"])
        all_raw.append(feats)
        for t in idxs:
            all_entries.append({"object_id": s["object_id"], "group": s["group"],
                                "category": s["category"], "token_index": int(t)})
    raw = np.concatenate(all_raw, axis=0)
    mean, std = raw.mean(0), raw.std(0)
    kept_dims = np.where(std > 1e-12)[0]
    keys = (raw[:, kept_dims] - mean[kept_dims]) / std[kept_dims]
    meta = {"mean": mean.tolist(), "std": std.tolist(), "kept_dims": kept_dims.tolist(), "l": L}
    return keys, all_entries, meta


def query_feature_17(object_id: str, token_index: int, meta: dict) -> np.ndarray:
    feats, idxs = _windows_for_session(object_id, PORTS17, token_index + 1)
    raw = feats[list(idxs).index(token_index)]
    mean, std, kept_dims = np.asarray(meta["mean"]), np.asarray(meta["std"]), meta["kept_dims"]
    return (raw[kept_dims] - mean[kept_dims]) / std[kept_dims]


def query_feature_10(object_id: str, token_index: int, meta: dict) -> np.ndarray:
    feats, idxs = _windows_for_session(object_id, list(experience_index.ANCHOR_INDEPENDENT_10), token_index + 1)
    raw = feats[list(idxs).index(token_index)]
    return compose_mod.standardize_query(raw, meta["mean"], meta["std"], meta["kept_dims"])


def p0_fold_of_group(groups_sorted: list[str]) -> dict[str, int]:
    """Reproduces EXP-HEKBv3/code/engine.py assign_folds() exactly (same seed/formula)."""
    nb = len(groups_sorted)
    rng = np.random.default_rng(P0_FOLD_SEED)
    order = rng.permutation(nb)
    fold_of_bundle = np.empty(nb, dtype=int)
    fold_of_bundle[order] = np.arange(nb) % P0_K_FOLDS
    return {g: int(f) for g, f in zip(groups_sorted, fold_of_bundle)}


def default_temperature(keys: np.ndarray, keep_mask: np.ndarray, k: int = 16, max_n: int = 2000, seed: int = 0) -> float:
    kept = keys[keep_mask]
    n = kept.shape[0]
    if n > max_n:
        rng = np.random.default_rng(seed)
        kept = kept[rng.choice(n, size=max_n, replace=False)]
        n = kept.shape[0]
    kk = min(k, n - 1)
    sq = np.sum(kept ** 2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (kept @ kept.T)
    np.fill_diagonal(d2, np.inf)
    kth = np.partition(d2, kk - 1, axis=1)[:, kk - 1]
    return float(np.median(kth)) or 1.0


def laplace_posterior_session_freq(other_sessions: list[dict], all_cats: list[str], alpha: float) -> dict[str, float]:
    k = max(len(all_cats), 1)
    counts = {c: 0.0 for c in all_cats}
    for s in other_sessions:
        counts[s["category"]] = counts.get(s["category"], 0.0) + 1.0
    total = float(len(other_sessions))
    return {c: (counts[c] + alpha) / (total + k * alpha) for c in all_cats}


def run_step(sessions, feature_dim, aggregation, posterior_mode, bundle_split,
             shuffled_category_by_group=None, fold_of_group=None):
    """One full 48-query pass under a given combination of settings. Returns per-query rows."""
    if feature_dim == 17:
        keys, entries, meta = build_library_17(sessions)
        qfeat = query_feature_17
    else:
        keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)
        qfeat = query_feature_10

    rows = []
    for q in sessions:
        own_group = q["group"]
        true_cat = shuffled_category_by_group[own_group] if shuffled_category_by_group else q["category"]
        if bundle_split == "kfold6":
            own_fold = fold_of_group[own_group]
            excl_groups = {g for g, f in fold_of_group.items() if f == own_fold}
        else:
            excl_groups = {own_group}

        use_entries = entries
        if shuffled_category_by_group:
            use_entries = [dict(e, category=shuffled_category_by_group[e["group"]]) for e in entries]

        keep_mask = np.array([e["group"] not in excl_groups for e in use_entries])
        other_sessions = [s for s in sessions if s["group"] not in excl_groups]
        other_cats = [shuffled_category_by_group[s["group"]] if shuffled_category_by_group else s["category"]
                      for s in other_sessions]
        all_cats = sorted(set(other_cats))
        k_cat = max(len(all_cats), 1)
        alpha = 1.0 / k_cat
        prior_p = laplace_posterior_session_freq(other_sessions, all_cats, alpha)
        prior_logloss = -math.log(max(prior_p.get(true_cat, 0.0), EPS_LOG))
        uniform_logloss = math.log(k_cat)

        t0 = default_temperature(keys, keep_mask)
        token_range = range(L - 1, q["n_steps"]) if aggregation == "session_mean" else [q["n_steps"] - 1]
        window_losses = []
        for tok in token_range:
            qf = qfeat(q["object_id"], tok, meta)
            result = compose_mod.compose(
                query_standardized=qf, query_object_id=q["object_id"], query_token_index=tok,
                keys=keys, entries=use_entries, objects_dir=OBJECTS_DIR, blobs_root=BLOBS_DIR,
                temperature=t0, top_m=TOP_M, exclude_groups=excl_groups, posterior_mode=posterior_mode,
                skip_r1=True,
            )
            p = result.category_posterior.get(true_cat, 0.0)
            window_losses.append(-math.log(max(p, EPS_LOG)))
        kernel_logloss = float(np.mean(window_losses))
        rows.append({"group": own_group, "kernel_logloss": kernel_logloss,
                     "prior_logloss": prior_logloss, "uniform_logloss": uniform_logloss})
    return rows


def bootstrap_ci(rows, field, seed=BOOTSTRAP_SEED, n_rep=N_BOOTSTRAP):
    by_group: dict[str, list[float]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r[field])
    distinct = sorted(by_group.keys())
    point = float(np.mean([r[field] for r in rows]))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_rep):
        sampled = rng.choice(distinct, size=len(distinct), replace=True)
        vals = [v for g in sampled for v in by_group[g]]
        means.append(np.mean(vals))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return point, float(lo), float(hi)


def summarize(rows, label):
    km, klo, khi = bootstrap_ci(rows, "kernel_logloss")
    pm, plo, phi = bootstrap_ci(rows, "prior_logloss")
    um, ulo, uhi = bootstrap_ci(rows, "uniform_logloss")
    red = 1.0 - km / pm
    print(f"  [{label}] kernel={km:.4f} [{klo:.4f},{khi:.4f}]  prior={pm:.4f} [{plo:.4f},{phi:.4f}]  "
          f"uniform={um:.4f}  reduction_vs_prior={red:.4f}")
    return {"kernel": km, "prior": pm, "uniform": um, "reduction_vs_prior": red}


def main() -> None:
    sessions = _load_all_sessions()
    assert len(sessions) == 48
    groups_sorted = sorted({s["group"] for s in sessions})
    fold_of_group = p0_fold_of_group(groups_sorted)
    fold_sizes = {f: sum(1 for g in fold_of_group.values() if g == f) for f in range(P0_K_FOLDS)}
    print(f"P0 fold_of_group (seed={P0_FOLD_SEED}): {len(groups_sorted)} groups, sizes={fold_sizes}")

    rng = np.random.default_rng(SHUFFLE_SEED)
    cats_list = [next(s["category"] for s in sessions if s["group"] == g) for g in groups_sorted]
    shuffled = rng.permutation(cats_list)
    shuf_map = {g: c for g, c in zip(groups_sorted, shuffled)}

    print("\n=== 3. P0 saved numbers (not recomputed; EXP-HEKBv3/results/primary.json U1.r3) ===")
    print("  prior_logloss=4.620351  M2_logloss=3.153694  M2_reduction_vs_prior=0.317434  n_windows_U1=12693 bundles=30")

    print("\n=== step sequence: P0-full -> current-full (reduction_vs_prior at each step) ===")
    steps = [
        ("P0-full (17port,session_mean,p0_ne,kfold6)", 17, "session_mean", "p0_ne_add_one", "kfold6"),
        ("->10port", 10, "session_mean", "p0_ne_add_one", "kfold6"),
        ("->last_window", 10, "last_window", "p0_ne_add_one", "kfold6"),
        ("->alpha_over_k", 10, "last_window", "alpha_over_k", "kfold6"),
        ("->loo (=current defaults)", 10, "last_window", "alpha_over_k", "loo"),
    ]
    results = {}
    for label, fd, agg, pm, bs in steps:
        rows = run_step(sessions, fd, agg, pm, bs, fold_of_group=fold_of_group)
        results[label] = summarize(rows, label)
        rows_shuf = run_step(sessions, fd, agg, pm, bs, shuffled_category_by_group=shuf_map, fold_of_group=fold_of_group)
        results[label + " [shuffled]"] = summarize(rows_shuf, label + " [shuffled]")

    print("\n=== note: transport ===")
    print("  P0 engine.py R3 block (lines ~402-419) does not call blend_sphere/transport;")
    print("  transport affects only R1 (trajectory-direction cosine), not the R3 category posterior. Not run as a step.")

    print("\n=== 5. positive control (i) exclude=none, T sweep ===")
    keys10, entries10, meta10 = experience_index.load_index(INDEX_ROOT, SPACE_ID)
    for mult in (1.0, 0.3, 0.1, 0.03):
        rows = []
        self_weights = []
        for q in sessions:
            tok = q["n_steps"] - 1
            qf = query_feature_10(q["object_id"], tok, meta10)
            keep_mask = np.ones(len(entries10), dtype=bool)
            t0 = default_temperature(keys10, keep_mask) * mult
            result = compose_mod.compose(
                query_standardized=qf, query_object_id=q["object_id"], query_token_index=tok,
                keys=keys10, entries=entries10, objects_dir=OBJECTS_DIR, blobs_root=BLOBS_DIR,
                temperature=t0, top_m=TOP_M, skip_r1=True,
            )
            p = result.category_posterior.get(q["category"], 0.0)
            rows.append({"group": q["group"], "kernel_logloss": -math.log(max(p, EPS_LOG)),
                        "prior_logloss": 0.0, "uniform_logloss": math.log(12)})
            self_w = sum(w for w, oid, ti in zip(result.weights, result.neighbor_object_ids,
                                                 result.neighbor_token_indices)
                        if oid == q["object_id"] and ti == tok)
            self_weights.append(self_w)
        km, klo, khi = bootstrap_ci(rows, "kernel_logloss")
        mean_self_w = float(np.mean(self_weights))
        print(f"  T x{mult}: kernel_logloss={km:.4f} [{klo:.4f},{khi:.4f}] (uniform=2.484907) "
              f"mean_self_match_weight={mean_self_w:.4f}")


if __name__ == "__main__":
    main()
