"""HEKBV3-BUILD002 評価: 事前登録された library-size n カーブ (結果を見る前に固定;
PreregisteredConfig が設定そのもの)。sessions/ の取り込みが終わるまでは、このモジュールは
合成データ (tests/test_eval_curve_boardrecords.py) でのみテストする -- 実データでの実行は
まだ行っていない。Locked session はこのモジュールに一切渡さないこと (呼び出し側の責任)。

束 (bundle) = prompt_id = entries の "group" フィールド。R3 (category_posterior) のみを
使う -- compose() は skip_r1=True で呼ぶ (位置なし Trajectory, HEKBV3-BUILD002 §1 と同じ
理由: BoardRecord ソースのセッションに位置の列は無い)。objects_dir/blobs_root は
skip_r1=True のとき一切読まれないので、存在しないダミーの Path を渡す。

`index/compose.py` の compose()、temperature の既定 (post-exclude library の 16-NN 距離二乗
の中央値) と束単位のブートストラップは、scripts/p0_repro_trj003.py の
default_temperature()/bootstrap_ci() をそのまま再利用する (同じ式を二重に書かない)。

解釈はしない。数字と設定だけ。
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(REPO_ROOT))
from index import compose as compose_mod  # noqa: E402
import p0_repro_trj003 as p0r  # noqa: E402

DUMMY_PATH = Path("/nonexistent")  # never read: every compose() call here uses skip_r1=True
EPS_LOG = 1e-12


@dataclass(frozen=True)
class PreregisteredConfig:
    """Fixed BEFORE looking at any result (HEKBV3-BUILD002 評価セクション, 事前登録)."""
    n_values: tuple[int, ...] = (6, 12, 24, 48, 96, 192)
    n_resamples: int = 20
    top_m: int = 64
    temperature_multiplier: float = 1.0  # 主; x0.3 は別枠の探索run (別の Config インスタンス)
    seed: int = 20261008
    n_bootstrap: int = 2000
    bootstrap_seed: int = 0
    positive_control: bool = False  # True: クエリ自身の束を library に残す (記憶の陽性対照)


def split_groups(all_groups: list[str], holdout_frac: float, seed: int) -> tuple[list[str], list[str]]:
    """Deterministic, seed-fixed split into (library_pool_groups, query_groups). query_groups
    is the FIXED held-out set used at every n (never in the library pool, per spec)."""
    groups = sorted(all_groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    n_query = max(1, round(len(groups) * holdout_frac))
    query_idx = set(order[:n_query].tolist())
    query_groups = [g for i, g in enumerate(groups) if i in query_idx]
    library_pool = [g for i, g in enumerate(groups) if i not in query_idx]
    return library_pool, query_groups


def session_logloss(entries: list[dict], keys: np.ndarray, query_group: str, query_rows: list[tuple],
                     library_groups: set[str], config: PreregisteredConfig) -> float:
    """query_rows: [(query_standardized, query_object_id, query_token_index, true_category), ...]
    for every valid window of ONE query session. Mean kernel log-loss over those windows,
    library restricted to `library_groups` (plus always excluding the query's own group,
    unless config.positive_control)."""
    all_groups = {e["group"] for e in entries}
    keep_groups = set(library_groups)
    if not config.positive_control:
        keep_groups.discard(query_group)
    exclude_groups = all_groups - keep_groups
    keep_mask = np.array([e["group"] not in exclude_groups for e in entries])
    if not keep_mask.any():
        raise ValueError("every library entry excluded; nothing to compose against")
    t0 = p0r.default_temperature(keys, keep_mask) * config.temperature_multiplier

    losses = []
    for query_standardized, query_object_id, query_token_index, true_category in query_rows:
        result = compose_mod.compose(
            query_standardized=query_standardized, query_object_id=query_object_id,
            query_token_index=query_token_index, keys=keys, entries=entries,
            objects_dir=DUMMY_PATH, blobs_root=DUMMY_PATH,
            temperature=t0, top_m=config.top_m, exclude_groups=exclude_groups, skip_r1=True,
        )
        p = result.category_posterior.get(true_category, 0.0)
        losses.append(-math.log(max(p, EPS_LOG)))
    return float(np.mean(losses))


def _shuffle_categories_by_group(entries: list[dict], seed: int) -> list[dict]:
    """Group(=bundle)-level category-label shuffle: permutes which bundle's rows carry which
    category label (not a per-row/window shuffle)."""
    groups_sorted = sorted({e["group"] for e in entries})
    cat_by_group = {}
    for g in groups_sorted:
        cat_by_group[g] = next(e["category"] for e in entries if e["group"] == g)
    rng = np.random.default_rng(seed)
    permuted_groups = rng.permutation(groups_sorted)
    shuffled_cat_of_group = dict(zip(groups_sorted, [cat_by_group[g] for g in permuted_groups]))
    return [dict(e, category=shuffled_cat_of_group[e["group"]]) for e in entries]


def run_n_curve(
    entries: list[dict], keys: np.ndarray, library_pool_groups: list[str],
    query_sessions: dict[str, list[tuple]], config: PreregisteredConfig, shuffle: bool = False,
) -> dict[int, list[dict]]:
    """Returns {n: [{"group": query_group, "logloss": ...}, ...]} -- `config.n_resamples` rows
    per query group at each n (one row per library resample), so scripts/p0_repro_trj003.py's
    `bootstrap_ci` (which groups by "group" and resamples groups with replacement) performs a
    bundle-level bootstrap that also naturally pools across the n_resamples library draws."""
    query_groups = set(query_sessions.keys())
    overlap = query_groups & set(library_pool_groups)
    if overlap:
        raise ValueError(f"query groups must not overlap the library pool: {overlap}")

    use_entries = _shuffle_categories_by_group(entries, config.seed) if shuffle else entries
    pool = sorted(library_pool_groups)

    out: dict[int, list[dict]] = {}
    for n in config.n_values:
        if n > len(pool):
            raise ValueError(f"n={n} exceeds library pool size {len(pool)}")
        rows = []
        for r in range(config.n_resamples):
            resample_rng = np.random.default_rng((config.seed, n, r))
            sampled = set(resample_rng.choice(pool, size=n, replace=False).tolist())
            for qgroup, qrows in query_sessions.items():
                ll = session_logloss(use_entries, keys, qgroup, qrows, sampled, config)
                rows.append({"group": qgroup, "logloss": ll})
        out[n] = rows
    return out


def fit_slope(n_values: tuple[int, ...], means: list[float]) -> float:
    """Least-squares slope of mean log-loss against log2(n) (n is geometrically spaced by
    design, so log2(n) -- not n itself -- is the natural x-axis; disclosed choice)."""
    xs = np.array([math.log2(n) for n in n_values])
    ys = np.array(means)
    xs_c = xs - xs.mean()
    denom = np.sum(xs_c ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum(xs_c * (ys - ys.mean())) / denom)


def slope_bootstrap_ci(curve_rows: dict[int, list[dict]], n_values: tuple[int, ...],
                        n_bootstrap: int, seed: int) -> tuple[float, float]:
    """Bootstraps over query groups, resampling the SAME group set jointly across all n within
    one replicate (keeps each replicate's curve internally coherent)."""
    groups = sorted({r["group"] for rows in curve_rows.values() for r in rows})
    by_group_by_n = {n: {} for n in n_values}
    for n, rows in curve_rows.items():
        for r in rows:
            by_group_by_n[n].setdefault(r["group"], []).append(r["logloss"])
    rng = np.random.default_rng(seed)
    slopes = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        means = []
        for n in n_values:
            vals = [v for g in sampled for v in by_group_by_n[n].get(g, [])]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        slopes.append(fit_slope(n_values, means))
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(lo), float(hi)


def changepoint_scan(n_values: tuple[int, ...], means: list[float]) -> list[dict]:
    """Minimal, disclosed heuristic (not a formal changepoint test): for each interior n-point,
    compares the slope of the segment to its left against the segment to its right."""
    out = []
    for i in range(1, len(n_values) - 1):
        left = (means[i] - means[i - 1]) / (math.log2(n_values[i]) - math.log2(n_values[i - 1]))
        right = (means[i + 1] - means[i]) / (math.log2(n_values[i + 1]) - math.log2(n_values[i]))
        out.append({"at_n": n_values[i], "left_slope": left, "right_slope": right, "delta": right - left})
    return out


def check_acceptance(curve_real: dict[int, list[dict]], curve_shuffle: dict[int, list[dict]],
                      n_values: tuple[int, ...], uniform_logloss: float, config: PreregisteredConfig) -> dict:
    """(a) at max(n_values): real below uniform (95% CI) AND real/shuffle CIs separate.
    (b) 95% CI of the logloss-vs-log2(n) slope excludes 0 (and is negative).
    Returns `detected`=False (not "no effect") when either criterion fails -- report, don't
    interpret further than that, per the task's own instruction."""
    max_n = max(n_values)
    real_m, real_lo, real_hi = p0r.bootstrap_ci(curve_real[max_n], "logloss",
                                                 seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
    shuf_m, shuf_lo, shuf_hi = p0r.bootstrap_ci(curve_shuffle[max_n], "logloss",
                                                 seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
    below_uniform = real_hi < uniform_logloss
    ci_separate = (real_hi < shuf_lo) or (real_lo > shuf_hi)
    criterion_a = below_uniform and ci_separate

    means = [p0r.bootstrap_ci(curve_real[n], "logloss", seed=config.bootstrap_seed,
                               n_rep=config.n_bootstrap)[0] for n in n_values]
    slope = fit_slope(n_values, means)
    slope_lo, slope_hi = slope_bootstrap_ci(curve_real, n_values, config.n_bootstrap, config.bootstrap_seed)
    criterion_b = slope_hi < 0.0

    return {
        "criterion_a": criterion_a, "criterion_b": criterion_b, "detected": criterion_a and criterion_b,
        "real_at_max_n": {"mean": real_m, "lo": real_lo, "hi": real_hi},
        "shuffle_at_max_n": {"mean": shuf_m, "lo": shuf_lo, "hi": shuf_hi},
        "uniform_logloss": uniform_logloss,
        "means_by_n": dict(zip(n_values, means)),
        "slope": slope, "slope_ci": [slope_lo, slope_hi],
        "changepoints": changepoint_scan(n_values, means),
    }
