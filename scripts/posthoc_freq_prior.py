"""HEKBV3-BUILD002 事後分析 §1-2 (探索、判定ではない): compose() を一切使わない頻度ベースの
事前分布 (alpha=1/K ラプラス平滑化、library 内のカテゴリー頻度のみ) を、事前登録と同じ
seed・同じ (n, 再抽出) の library サンプルに対して計算し、real/shuffle と同じ表に並べる。

この頻度事前分布は compose() を全く呼ばない (束のカテゴリーラベルの頻度のみ) ので、この
スクリプト全体で評価に数分もかからない。

real/shuffle との対応付けは、scripts/run_ncurve_eval.py が書き出した curve_real/curve_shuffle
の各行に明示的な resample 番号が保存されていない (group とlogloss のみ) ため、同スクリプトの
ループ順序 (resample 外側 -> query_sessions.items() 内側、どちらも挿入順で安定) から、
69 行ずつの連続チャンク = 1 回の再抽出、として復元する。これは明示的なフィールドではなく
暗黙の順序保存に依存した復元であることを、ここに明記する (事後分析であり、判定には使わない)。

事前登録の結果ファイル (eval_results.json) は変更しない。出力は新しい JSON のみ。
解釈はしない。数字と設定だけ。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from index import experience_index
import eval_curve_boardrecords as ecb
import p0_repro_trj003 as p0r

INDEX_ROOT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/index")
SPACE_ID = "ec435e5ffc9f46cff1e0f5ef2a00a233367ff091884d4cbdc2b36cc56d913e27"
EVAL_RESULTS = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/eval_results.json")
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_freq_prior.json")

EPS_LOG = ecb.EPS_LOG


def main() -> None:
    config = ecb.PreregisteredConfig()
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)

    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=0.2, seed=config.seed)
    pool = sorted(library_pool)

    cat_by_group: dict[str, str] = {}
    for e in entries:
        cat_by_group.setdefault(e["group"], e["category"])

    overall_categories = sorted({cat_by_group[g] for g in all_groups})
    K = len(overall_categories)
    alpha = 1.0 / K

    # query_sessions dict insertion order, same construction as run_ncurve_eval.py's
    # build_query_sessions -- needed only for its KEY order (one logloss value per query
    # GROUP here, not per window, since every window of one bundle shares one category).
    query_group_order = [g for g in query_groups]  # split_groups already returns sorted-filtered order
    n_query_groups = len(query_group_order)

    # ---- step 1: base rates (no compose) ----
    overall_cat_counts = {c: 0 for c in overall_categories}
    for g in all_groups:
        overall_cat_counts[cat_by_group[g]] += 1
    fit_cat_counts = {c: 0 for c in overall_categories}
    for g in library_pool:
        fit_cat_counts[cat_by_group[g]] += 1
    dev_cat_counts = {c: 0 for c in overall_categories}
    for g in query_groups:
        dev_cat_counts[cat_by_group[g]] += 1

    dev_total = sum(dev_cat_counts.values())
    dev_entropy_nats = -sum(
        (c_n / dev_total) * math.log(c_n / dev_total) for c_n in dev_cat_counts.values() if c_n > 0
    )

    uniform_logloss = float(math.log(K))

    # ---- step 1b: frequency-prior curve, SAME (seed, n, r) library draws as the
    # preregistered run (run_ncurve_eval.py / eval_curve_boardrecords.run_n_curve) ----
    freq_prior_curve: dict[int, list[dict]] = {}
    for n in config.n_values:
        rows = []
        for r in range(config.n_resamples):
            resample_rng = np.random.default_rng((config.seed, n, r))
            sampled = resample_rng.choice(pool, size=n, replace=False).tolist()
            counts = {c: 0 for c in overall_categories}
            for g in sampled:
                counts[cat_by_group[g]] += 1
            posterior = {c: (counts[c] + alpha) / (n + K * alpha) for c in overall_categories}
            for qgroup in query_group_order:
                true_cat = cat_by_group[qgroup]
                p = posterior.get(true_cat, 0.0)
                rows.append({"group": qgroup, "logloss": -math.log(max(p, EPS_LOG))})
        freq_prior_curve[n] = rows

    # ---- load preregistered curve_real/curve_shuffle, reconstruct per-resample chunks ----
    data = json.loads(EVAL_RESULTS.read_text())
    curve_real = {int(k): v for k, v in data["curve_real"].items()}
    curve_shuffle = {int(k): v for k, v in data["curve_shuffle"].items()}
    n_resamples = config.n_resamples

    granularity_ok = True
    for n in config.n_values:
        if len(curve_real[n]) != n_resamples * n_query_groups:
            granularity_ok = False

    # ---- combined table: uniform / freq_prior / real / shuffle, per n, bundle-level
    # bootstrap CI (p0_repro_trj003.bootstrap_ci, same as the preregistered report) ----
    table = {}
    for n in config.n_values:
        fp_m, fp_lo, fp_hi = p0r.bootstrap_ci(freq_prior_curve[n], "logloss",
                                               seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
        r_m, r_lo, r_hi = p0r.bootstrap_ci(curve_real[n], "logloss",
                                            seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
        s_m, s_lo, s_hi = p0r.bootstrap_ci(curve_shuffle[n], "logloss",
                                            seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
        table[n] = {
            "uniform": uniform_logloss,
            "freq_prior": {"mean": fp_m, "lo": fp_lo, "hi": fp_hi},
            "real": {"mean": r_m, "lo": r_lo, "hi": r_hi},
            "shuffle": {"mean": s_m, "lo": s_lo, "hi": s_hi},
        }

    # ---- step 2: diff curves (real - shuffle, real - freq_prior), paired by (resample,
    # group) chunk position, pooled into {"group":..., "logloss": diff} rows so the existing
    # bundle-level bootstrap_ci / slope machinery applies unchanged. Relies on the SAME
    # implicit chunk-order assumption disclosed above. ----
    diff_real_minus_shuffle: dict[int, list[dict]] = {}
    diff_real_minus_freq: dict[int, list[dict]] = {}
    if granularity_ok:
        for n in config.n_values:
            rr, ss, ff = curve_real[n], curve_shuffle[n], freq_prior_curve[n]
            d_rs, d_rf = [], []
            for i in range(len(rr)):
                assert rr[i]["group"] == ss[i]["group"] == ff[i]["group"], \
                    "chunk-order assumption violated -- group mismatch at row %d" % i
                d_rs.append({"group": rr[i]["group"], "logloss": rr[i]["logloss"] - ss[i]["logloss"]})
                d_rf.append({"group": rr[i]["group"], "logloss": rr[i]["logloss"] - ff[i]["logloss"]})
            diff_real_minus_shuffle[n] = d_rs
            diff_real_minus_freq[n] = d_rf

        diff_table = {}
        for n in config.n_values:
            m1, lo1, hi1 = p0r.bootstrap_ci(diff_real_minus_shuffle[n], "logloss",
                                             seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
            m2, lo2, hi2 = p0r.bootstrap_ci(diff_real_minus_freq[n], "logloss",
                                             seed=config.bootstrap_seed, n_rep=config.n_bootstrap)
            diff_table[n] = {
                "real_minus_shuffle": {"mean": m1, "lo": lo1, "hi": hi1},
                "real_minus_freq_prior": {"mean": m2, "lo": lo2, "hi": hi2},
            }
        slope_rs = ecb.fit_slope(config.n_values, [diff_table[n]["real_minus_shuffle"]["mean"] for n in config.n_values])
        slope_rs_ci = ecb.slope_bootstrap_ci(diff_real_minus_shuffle, config.n_values, config.n_bootstrap, config.bootstrap_seed)
        slope_rf = ecb.fit_slope(config.n_values, [diff_table[n]["real_minus_freq_prior"]["mean"] for n in config.n_values])
        slope_rf_ci = ecb.slope_bootstrap_ci(diff_real_minus_freq, config.n_values, config.n_bootstrap, config.bootstrap_seed)
    else:
        diff_table = None
        slope_rs = slope_rs_ci = slope_rf = slope_rf_ci = None

    out = {
        "note": "post-hoc exploratory analysis, NOT the preregistered acceptance decision",
        "K_categories": K, "categories": overall_categories,
        "overall_group_counts_by_category": overall_cat_counts,
        "fit_group_counts_by_category": fit_cat_counts,
        "dev_group_counts_by_category": dev_cat_counts,
        "dev_category_entropy_nats": dev_entropy_nats,
        "uniform_logloss": uniform_logloss,
        "table_by_n": {str(n): table[n] for n in config.n_values},
        "granularity_ok_for_paired_diff": granularity_ok,
        "diff_table_by_n": {str(n): diff_table[n] for n in config.n_values} if diff_table else None,
        "slope_real_minus_shuffle": slope_rs,
        "slope_real_minus_shuffle_ci": list(slope_rs_ci) if slope_rs_ci else None,
        "slope_real_minus_freq_prior": slope_rf,
        "slope_real_minus_freq_prior_ci": list(slope_rf_ci) if slope_rf_ci else None,
        "chunk_order_assumption": ("curve_real/curve_shuffle rows have no explicit resample "
                                    "index; reconstructed by chunking into n_resamples blocks "
                                    "of n_query_groups rows each, relying on run_ncurve_eval.py's "
                                    "stable (resample outer, query_sessions.items() inner) "
                                    "iteration order. Verified group-for-group against this "
                                    "script's own freq_prior_curve chunks (same construction) "
                                    "before trusting the diff."),
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
