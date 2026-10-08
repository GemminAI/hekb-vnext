"""HEKBV3-BUILD001 差分4: feature separability + statistical power (read-only; no new
exp_hekbv3 writes; compose() has no side effects). Same 48 sessions, group-level
("項目単位") leave-one-out throughout. 10-port (L=8, d=63 kept) and 17-port (L=8, d=136)
feature sets, real labels and group-level category-label shuffle, group-level bootstrap.

1. Feature separability: per-port one-way ANOVA F / eta^2 over all valid windows (window's
   own-token raw scalar value, grouped by the window's session category). Session-level
   (session-mean feature, 1 point/session) top-1 classification via kNN(k=1,3) and a small
   numpy multinomial logistic regression, group-level LOO, chance=1/12. Group-level
   permutation test (1000 reps) for the two kNN statistics (cheap: the 48x48 distance
   matrix does not depend on labels, so it is computed once and reused for every
   permutation); logistic regression is reported as a point estimate only (real labels +
   the single fixed group-shuffle already used throughout this thread) -- a disclosed
   scope decision, not a full 1000-rep refit, to keep runtime bounded.
2. Power: inject a synthetic per-category offset (a fixed random unit direction per
   category, scaled by delta "standard deviations") into the standardized library+query
   features, delta in {0.1,0.2,0.5,1.0,2.0}, rerun the existing LOO compose() log-loss at
   T in {1,0.3,0.1} x T0, both port sets, real and group-shuffled labels, group-bootstrap
   CI. Reports, per (ports,T) combination, the smallest delta where (a) mean real logloss
   < uniform and (b) the real/shuffled 95% CIs no longer overlap.
3. Converts the real data's per-port eta^2 (step 1) to an equivalent delta via the
   standard f=sqrt(eta^2/(1-eta^2)), delta_equiv=2f convention, placed next to step 2's
   table for a direct numeric comparison (no claim about this being exact for k=12 groups).
4. A rough required-n estimate to detect step 3's delta at the same confidence as step
   2's minimal detectable delta at n=48, via n_required ~= 48 * (delta_min48/delta_real)^2
   (power scales with delta*sqrt(n)), compared against the TRJ003 pool size mentioned in
   the task (fit/dev ~1,296).

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
PERM_SEED = 1
POWER_SEED = 2
N_PERM = 1000
L = 8
EPS_LOG = 1e-12
PORTS17 = ["P01", "P03", "P06", "P09", "P10", "P13", "P16", "P18", "P19", "P25", "P26", "P28", "P30", "P31",
           "P35", "P36", "P38"]
ALL_PORT_IDS = [f"P{n:02d}" for n in range(1, 39)]


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


def build_library(sessions: list[dict], ports: list[str]) -> tuple[np.ndarray, list[dict], dict]:
    all_raw, all_entries = [], []
    for s in sessions:
        feats, idxs = _windows_for_session(s["object_id"], ports, s["n_steps"])
        all_raw.append(feats)
        for t in idxs:
            all_entries.append({"object_id": s["object_id"], "group": s["group"],
                                "category": s["category"], "token_index": int(t)})
    raw = np.concatenate(all_raw, axis=0)
    mean, std = raw.mean(0), raw.std(0)
    kept_dims = np.where(std > 1e-12)[0]
    keys = (raw[:, kept_dims] - mean[kept_dims]) / std[kept_dims]
    meta = {"mean": mean, "std": std, "kept_dims": kept_dims, "l": L}
    return keys, all_entries, meta


def query_feature(object_id: str, token_index: int, ports: list[str], meta: dict) -> np.ndarray:
    feats, idxs = _windows_for_session(object_id, ports, token_index + 1)
    raw = feats[list(idxs).index(token_index)]
    return (raw[meta["kept_dims"]] - meta["mean"][meta["kept_dims"]]) / meta["std"][meta["kept_dims"]]


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


# ---------------------------------------------------------------------------------------- 1

def anova_per_port(sessions: list[dict], ports: list[str]) -> dict[str, tuple[float, float]]:
    """Per-port one-way ANOVA F and eta^2 over the port's own-token raw scalar value at
    every valid window position, grouped by the window's session category."""
    cats_sorted = sorted({s["category"] for s in sessions})
    cat_idx = {c: i for i, c in enumerate(cats_sorted)}
    values_by_port: dict[str, list[np.ndarray]] = {p: [] for p in ports}
    labels_all = []
    for s in sessions:
        series, port_ids = _raw_ports_series(s["object_id"])
        col_idx = {p: port_ids.index(p) for p in ports}
        ts = np.arange(L - 1, s["n_steps"])
        for p in ports:
            values_by_port[p].append(series[ts, col_idx[p]])
        labels_all.append(np.full(len(ts), cat_idx[s["category"]]))
    labels = np.concatenate(labels_all)
    k = len(cats_sorted)
    n = len(labels)
    out = {}
    for p in ports:
        x = np.concatenate(values_by_port[p])
        grand_mean = x.mean()
        ss_total = float(((x - grand_mean) ** 2).sum())
        ss_between = 0.0
        for c in range(k):
            mask = labels == c
            ng = mask.sum()
            if ng == 0:
                continue
            ss_between += float(ng * (x[mask].mean() - grand_mean) ** 2)
        ss_within = ss_total - ss_between
        if ss_total <= 1e-12 or ss_within <= 1e-12:
            out[p] = (float("nan"), float("nan"))
            continue
        f_stat = (ss_between / (k - 1)) / (ss_within / (n - k))
        eta2 = ss_between / ss_total
        out[p] = (f_stat, eta2)
    return out


def session_mean_features(sessions: list[dict], keys: np.ndarray, entries: list[dict]) -> np.ndarray:
    by_object: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        by_object.setdefault(e["object_id"], []).append(i)
    return np.stack([keys[by_object[s["object_id"]]].mean(0) for s in sessions])


def knn_predict_all(dist2: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """dist2: (n,n) squared distance with the diagonal set to +inf. Returns top-1 predicted
    label for each row via unweighted majority vote among the k nearest (ties -> nearest)."""
    n = dist2.shape[0]
    preds = np.empty(n, dtype=labels.dtype)
    order = np.argsort(dist2, axis=1)
    for i in range(n):
        neigh = order[i, :k]
        vals, counts = np.unique(labels[neigh], return_counts=True)
        best = vals[np.argmax(counts)]
        preds[i] = best
    return preds


def multinomial_logreg_loo(features: np.ndarray, labels: np.ndarray, groups: np.ndarray,
                            n_classes: int, l2: float = 1.0, lr: float = 0.1, n_iter: int = 300) -> np.ndarray:
    """Tiny numpy multinomial logistic regression (gradient descent, L2), group-level LOO.
    Returns top-1 predictions for every row."""
    n, d = features.shape
    preds = np.empty(n, dtype=labels.dtype)
    for g in np.unique(groups):
        te = np.where(groups == g)[0]
        tr = np.where(groups != g)[0]
        X, y = features[tr], labels[tr]
        W = np.zeros((d, n_classes))
        b = np.zeros(n_classes)
        onehot = np.eye(n_classes)[y]
        for _ in range(n_iter):
            logits = X @ W + b
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(1, keepdims=True)
            grad_logits = (p - onehot) / len(tr)
            W -= lr * (X.T @ grad_logits + l2 * W / len(tr))
            b -= lr * grad_logits.sum(0)
        Xq = features[te]
        logits_q = Xq @ W + b
        preds[te] = logits_q.argmax(1)
    return preds


def run_step1(sessions, ports, label):
    print(f"\n--- 1. feature separability [{ports[0] and (len(ports))}ports] ({label}) ---")
    anova = anova_per_port(sessions, ports)
    for p in ports:
        f, e2 = anova[p]
        print(f"  {p}: F={f:.4f} eta2={e2:.4f}")
    vals = [e2 for f, e2 in anova.values() if not math.isnan(e2)]
    mean_eta2 = float(np.mean(vals)) if vals else float("nan")
    max_eta2 = float(np.max(vals)) if vals else float("nan")
    print(f"  mean_eta2={mean_eta2:.4f} max_eta2={max_eta2:.4f}")

    keys, entries, meta = build_library(sessions, ports)
    feats = session_mean_features(sessions, keys, entries)
    cats_sorted = sorted({s["category"] for s in sessions})
    cat_idx = {c: i for i, c in enumerate(cats_sorted)}
    labels = np.array([cat_idx[s["category"]] for s in sessions])
    groups = np.array([s["group"] for s in sessions])
    n = len(sessions)
    sq = np.sum(feats ** 2, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.T)
    group_mask = groups[:, None] == groups[None, :]
    dist2_loo = np.where(group_mask, np.inf, dist2)

    def acc(lab, k):
        pred = knn_predict_all(dist2_loo, lab, k)
        return float((pred == lab).mean())

    obs_k1 = acc(labels, 1)
    obs_k3 = acc(labels, 3)
    pred_lr = multinomial_logreg_loo(feats, labels, groups, len(cats_sorted))
    obs_lr = float((pred_lr == labels).mean())
    print(f"  session top-1 accuracy (chance={1/len(cats_sorted):.4f}): kNN1={obs_k1:.4f} kNN3={obs_k3:.4f} logreg={obs_lr:.4f}")

    distinct_groups = sorted(set(groups))
    rng = np.random.default_rng(PERM_SEED)
    null_k1, null_k3 = [], []
    group_to_label = {g: labels[np.where(groups == g)[0][0]] for g in distinct_groups}
    for _ in range(N_PERM):
        perm_labels_by_group = dict(zip(distinct_groups, rng.permutation([group_to_label[g] for g in distinct_groups])))
        lab_shuf = np.array([perm_labels_by_group[g] for g in groups])
        null_k1.append(acc(lab_shuf, 1))
        null_k3.append(acc(lab_shuf, 3))
    p_k1 = float((np.array(null_k1) >= obs_k1).mean())
    p_k3 = float((np.array(null_k3) >= obs_k3).mean())
    print(f"  permutation p-value (n={N_PERM}, group-level): kNN1 p={p_k1:.4f} kNN3 p={p_k3:.4f} (logreg: not permuted, scope decision)")

    fixed_shuf_rng = np.random.default_rng(SHUFFLE_SEED)
    shuf_map = dict(zip(distinct_groups, fixed_shuf_rng.permutation([group_to_label[g] for g in distinct_groups])))
    labels_shuf1 = np.array([shuf_map[g] for g in groups])
    pred_lr_shuf = multinomial_logreg_loo(feats, labels_shuf1, groups, len(cats_sorted))
    print(f"  single-shuffle (seed={SHUFFLE_SEED}) accuracy: kNN1={acc(labels_shuf1,1):.4f} kNN3={acc(labels_shuf1,3):.4f} "
          f"logreg={float((pred_lr_shuf==labels_shuf1).mean()):.4f}")
    return {"mean_eta2": mean_eta2, "max_eta2": max_eta2}


# ---------------------------------------------------------------------------------------- 2

def run_power_step(sessions, ports, delta, t_mult, shuffled_category_by_group=None, seed=POWER_SEED):
    keys, entries, meta = build_library(sessions, ports)
    cats_sorted = sorted({s["category"] for s in sessions})
    d = keys.shape[1]
    rng = np.random.default_rng(seed)
    direction = {c: rng.normal(size=d) for c in cats_sorted}
    for c in direction:
        direction[c] /= np.linalg.norm(direction[c])

    def offset_for(category):
        return delta * direction[category]

    keys_off = keys.copy()
    for i, e in enumerate(entries):
        keys_off[i] += offset_for(e["category"])

    rows = []
    for q in sessions:
        own_group = q["group"]
        true_cat = shuffled_category_by_group[own_group] if shuffled_category_by_group else q["category"]
        use_entries = entries
        if shuffled_category_by_group:
            use_entries = [dict(e, category=shuffled_category_by_group[e["group"]]) for e in entries]
        keep_mask = np.array([e["group"] != own_group for e in use_entries])
        other_cats = sorted(set(e["category"] for e in use_entries if e["group"] != own_group))
        t0 = default_temperature(keys_off, keep_mask) * t_mult
        tok = q["n_steps"] - 1
        qraw, idxs = _windows_for_session(q["object_id"], ports, q["n_steps"])
        raw = qraw[list(idxs).index(tok)]
        qf = (raw[meta["kept_dims"]] - meta["mean"][meta["kept_dims"]]) / meta["std"][meta["kept_dims"]]
        qf = qf + offset_for(true_cat)
        result = compose_mod.compose(
            query_standardized=qf, query_object_id=q["object_id"], query_token_index=tok,
            keys=keys_off, entries=use_entries, objects_dir=OBJECTS_DIR, blobs_root=BLOBS_DIR,
            temperature=t0, top_m=TOP_M, exclude_groups={own_group}, skip_r1=True,
        )
        p = result.category_posterior.get(true_cat, 0.0)
        rows.append({"group": own_group, "logloss": -math.log(max(p, EPS_LOG)),
                    "uniform": math.log(max(len(other_cats), 1))})
    return rows


def main() -> None:
    sessions = _load_all_sessions()
    assert len(sessions) == 48
    distinct_groups = sorted({s["group"] for s in sessions})
    rng = np.random.default_rng(SHUFFLE_SEED)
    group_to_cat = {g: next(s["category"] for s in sessions if s["group"] == g) for g in distinct_groups}
    shuffled = dict(zip(distinct_groups, rng.permutation([group_to_cat[g] for g in distinct_groups])))

    ports10 = list(experience_index.ANCHOR_INDEPENDENT_10)
    eta_summary = {}
    eta_summary["ports10"] = run_step1(sessions, ports10, "10ports")
    eta_summary["ports17"] = run_step1(sessions, PORTS17, "17ports")

    print("\n=== 2. power sweep (group-level LOO, real + group-shuffled) ===")
    deltas = [0.1, 0.2, 0.5, 1.0, 2.0]
    t_mults = [1.0, 0.3, 0.1]
    power_results = {}
    for ports, pname in ((ports10, "10ports"), (PORTS17, "17ports")):
        for t_mult in t_mults:
            for delta in deltas:
                rows_real = run_power_step(sessions, ports, delta, t_mult)
                rows_shuf = run_power_step(sessions, ports, delta, t_mult, shuffled_category_by_group=shuffled)
                rm, rlo, rhi = bootstrap_ci(rows_real, "logloss")
                sm, slo, shi = bootstrap_ci(rows_shuf, "logloss")
                uniform = float(np.mean([r["uniform"] for r in rows_real]))
                below_uniform = rm < uniform
                ci_separate = (rhi < slo) or (shi < rlo)
                key = (pname, t_mult)
                power_results.setdefault(key, []).append(
                    (delta, rm, rlo, rhi, sm, slo, shi, uniform, below_uniform, ci_separate))
                print(f"  ports={pname} Tx{t_mult} delta={delta}: real={rm:.4f}[{rlo:.4f},{rhi:.4f}] "
                      f"shuf={sm:.4f}[{slo:.4f},{shi:.4f}] uniform={uniform:.4f} "
                      f"below_uniform={below_uniform} ci_separate={ci_separate}")

    print("\n=== 2b. minimal delta meeting both criteria, per (ports,T) ===")
    for key, rows in power_results.items():
        hit = next((d for d, rm, *_r, uni, bu, cs in rows if bu and cs), None)
        print(f"  {key}: min_delta_both_criteria={hit}")

    print("\n=== 3. real-data eta2 -> equivalent delta (f=sqrt(eta2/(1-eta2)), delta=2f) ===")
    for pname, s in eta_summary.items():
        for which in ("mean_eta2", "max_eta2"):
            e2 = s[which]
            f = math.sqrt(e2 / (1 - e2)) if 0 < e2 < 1 else float("nan")
            d_equiv = 2 * f if not math.isnan(f) else float("nan")
            print(f"  {pname} {which}={e2:.4f} -> f={f:.4f} -> delta_equiv={d_equiv:.4f}")

    print("\n=== 4. rough required-n (n_required ~= 48*(delta_min48/delta_real)^2) ===")
    print("  (computed by hand in the report from sections 2b and 3's printed numbers)")


if __name__ == "__main__":
    main()
