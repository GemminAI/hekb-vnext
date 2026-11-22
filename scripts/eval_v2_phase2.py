"""HEKBV3-BUILD003 Phase 2: 事前登録 v2 (結果を見る前に固定、変更しない)。

主な基準は library のカテゴリー頻度事前分布 (alpha=1/K、library内の頻度)。均一は参考のみ。
Phase 1 の較正規則 (診断A: (i) セッション単位のcompose argmax正答率 > 頻度事前argmax正答率、
かつ (ii) 最大確率の平均 - 正答率 >= 0.10) を満たしたときだけ、posthoc_diag_a.json に基づき
p' = (1-lambda)*normalize(p^tau) + lambda*p_freq の較正を使う (内側split で (tau,lambda) を
real用・shuffle用に別々に選ぶ、評価用のdevは使わない)。満たさなければ tau=1,lambda=0 (無変換)
で評価する。

n=48,96,192、再抽出10回 (時間見積もりが8時間を超える場合は5回に落として実行、要報告)。
real と 束単位ラベルshuffle を、同じ compose() 呼び出しから導出 (run_n_curve_real_and_shuffle
と同じ理屈)。再抽出ごとの値 (real_logloss, shuffle_logloss, freq_logloss, セッション単位
top-1正答)を全て保存する (チャンク順序には依存しない、明示的な resample フィールド)。

受け入れ: n=192 で (a) real-freq_prior の対数損失差の95%区間の上限 < 0 (束単位ブートストラップ、
再抽出ごとの値を束でグループ化してブートストラップ)、(b) real-shuffle の差の95%区間の上限 < 0。
両方満たせば「検出された」、どちらか欠ければ「この設計・このnでは検出できなかった」(「効か
ない」とは書かない)。副: real-shuffle 差の log2(n) に対する傾きの区間。変化点検出 (real_logloss
平均に対して)。

実行直前に、使用する compose/eval コードの commit hash と設定を PREREG_v2.json に書き、その
sha256 を記録する (内容確定後に計算、ファイル自身にも書く)。クラッシュ時は同じ設定・同じ
出力ファイルで再実行すれば、既に完了した n をスキップして再開する (--out のn別チェックポイント)。

v1 の結果ファイル (eval_results.json) は変更しない。本番 experience・既存 exp_hekbv3 には
書かない。Locked は使わない。解釈はしない。数字と設定だけ。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import sys
import time
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
DIAG_A_PATH = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_diag_a.json")
PREREG_PATH = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/PREREG_v2.json")
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/eval_v2_results.json")

N_VALUES_DEFAULT = (48, 96, 192)
N_RESAMPLES_DEFAULT = 10
N_RESAMPLES_FALLBACK = 5
TIME_BUDGET_S = 8 * 3600
TAUS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75)
EPS_LOG = ecb.EPS_LOG


def build_query_sessions(entries, keys, query_groups):
    by_group = {}
    for i, e in enumerate(entries):
        by_group.setdefault(e["group"], []).append(i)
    out = {}
    for g in query_groups:
        out[g] = [(keys[i], entries[i]["object_id"], entries[i]["token_index"], entries[i]["category"])
                   for i in by_group.get(g, [])]
    return out


def freq_posterior_over(cat_by_group, groups, categories, K):
    alpha = 1.0 / K
    counts = {c: 0 for c in categories}
    for g in groups:
        counts[cat_by_group[g]] += 1
    n = len(groups)
    return {c: (counts[c] + alpha) / (n + K * alpha) for c in categories}


def transform(p_vec: np.ndarray, freq_vec: np.ndarray, tau: float, lam: float) -> np.ndarray:
    p_tau = np.power(p_vec, tau)
    p_tau_norm = p_tau / p_tau.sum(axis=-1, keepdims=True)
    return (1 - lam) * p_tau_norm + lam * freq_vec


def git_commit_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()


def decide_calibration() -> tuple[bool, dict]:
    if not DIAG_A_PATH.exists():
        return False, {"reason": "posthoc_diag_a.json not found"}
    diag = json.loads(DIAG_A_PATH.read_text())
    rule_i = diag.get("decision_rule_i_session_acc_compose_gt_freq")
    rule_ii = diag.get("decision_rule_ii_overconfidence_ge_0.10")
    use_cal = bool(rule_i) and bool(rule_ii)
    return use_cal, {
        "rule_i_session_acc_compose_gt_freq": rule_i, "rule_ii_overconfidence_ge_0.10": rule_ii,
        "session_accuracy_compose_argmax": diag.get("session_accuracy_compose_argmax"),
        "session_accuracy_freq_argmax": diag.get("session_accuracy_freq_argmax"),
        "mean_confidence_maxp": diag.get("mean_confidence_maxp"),
        "window_accuracy_compose_argmax": diag.get("window_accuracy_compose_argmax"),
        "confidence_minus_window_accuracy": diag.get("confidence_minus_window_accuracy"),
    }


def main() -> None:
    t_start = time.time()
    v1_config = ecb.PreregisteredConfig()
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)

    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=0.2, seed=v1_config.seed)
    query_sessions = build_query_sessions(entries, keys, query_groups)
    pool = sorted(library_pool)

    cat_by_group = {}
    for e in entries:
        cat_by_group.setdefault(e["group"], e["category"])
    categories = sorted({cat_by_group[g] for g in all_groups})
    K = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    groups_sorted = sorted({e["group"] for e in entries})
    rng_shuf = np.random.default_rng(v1_config.seed)
    permuted_groups = rng_shuf.permutation(groups_sorted)
    shuffled_cat_of_group = dict(zip(groups_sorted, [cat_by_group[g] for g in permuted_groups]))

    use_calibration, cal_diag = decide_calibration()
    print(f"[eval_v2_phase2] use_calibration={use_calibration} diag={cal_diag}", flush=True)

    tau_real, lam_real, tau_shuffle, lam_shuffle = 1.0, 0.0, 1.0, 0.0
    grid_real = grid_shuffle = None
    inner_library_size = inner_query_size = inner_windows = None

    if use_calibration:
        inner_library_pool, inner_query_groups = ecb.split_groups(pool, holdout_frac=0.2, seed=v1_config.seed)
        inner_query_sessions = build_query_sessions(entries, keys, inner_query_groups)
        inner_kept_keys, inner_kept_entries, inner_t0, inner_lib_cats_real = ecb._prefiltered_library(
            entries, keys, set(inner_library_pool), v1_config)
        inner_lib_cats_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in inner_kept_entries))
        inner_alpha_shuffle = 1.0 / max(len(inner_lib_cats_shuffle), 1)
        inner_entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in inner_kept_entries}
        inner_freq_real_vec = np.array([freq_posterior_over(cat_by_group, inner_library_pool, categories, K)[c]
                                         for c in categories])
        inner_freq_shuffle_vec = np.array([freq_posterior_over(shuffled_cat_of_group, inner_library_pool, categories, K)[c]
                                            for c in categories])

        inner_p_real, inner_p_shuffle, inner_true = [], [], []
        for qgroup, qrows in inner_query_sessions.items():
            for qstd, qoid, qtok, true_cat in qrows:
                result = ecb.compose_mod.compose(
                    query_standardized=qstd, query_object_id=qoid, query_token_index=qtok,
                    keys=inner_kept_keys, entries=inner_kept_entries,
                    objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                    temperature=inner_t0, top_m=v1_config.top_m, exclude_groups=None, skip_r1=True,
                    library_categories=inner_lib_cats_real,
                )
                inner_p_real.append(np.array([result.category_posterior.get(c, 0.0) for c in categories]))
                posterior_shuffle = ecb._posterior_from_neighbors(
                    result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                    inner_entry_group_lookup, shuffled_cat_of_group, inner_lib_cats_shuffle, inner_alpha_shuffle,
                )
                inner_p_shuffle.append(np.array([posterior_shuffle.get(c, 0.0) for c in categories]))
                inner_true.append(cat_idx[true_cat])
        inner_P_real = np.stack(inner_p_real)
        inner_P_shuffle = np.stack(inner_p_shuffle)
        inner_TRUE = np.array(inner_true)
        idxr = np.arange(len(inner_TRUE))
        inner_library_size, inner_query_size, inner_windows = len(inner_library_pool), len(inner_query_groups), len(inner_TRUE)

        def grid_search(P, freq_vec):
            best, grid = None, []
            for tau in TAUS:
                for lam in LAMBDAS:
                    pm = transform(P, freq_vec[None, :], tau, lam)
                    loss = float(np.mean(-np.log(np.maximum(pm[idxr, inner_TRUE], EPS_LOG))))
                    grid.append({"tau": tau, "lambda": lam, "inner_logloss": loss})
                    if best is None or loss < best["inner_logloss"]:
                        best = {"tau": tau, "lambda": lam, "inner_logloss": loss}
            return best, grid

        best_real, grid_real = grid_search(inner_P_real, inner_freq_real_vec)
        best_shuffle, grid_shuffle = grid_search(inner_P_shuffle, inner_freq_shuffle_vec)
        tau_real, lam_real = best_real["tau"], best_real["lambda"]
        tau_shuffle, lam_shuffle = best_shuffle["tau"], best_shuffle["lambda"]
        print(f"[eval_v2_phase2] calibration selected real=({tau_real},{lam_real}) "
              f"shuffle=({tau_shuffle},{lam_shuffle})", flush=True)

    # ---- time estimate / resample-count fallback ----
    per_resample_s = {48: 242.76, 96: 445.35, 192: 987.8}  # measured, v1 run, combined real+shuffle
    n_values = N_VALUES_DEFAULT
    n_resamples = N_RESAMPLES_DEFAULT
    est_s = n_resamples * sum(per_resample_s[n] for n in n_values)
    fallback_used = False
    if est_s > TIME_BUDGET_S:
        n_resamples = N_RESAMPLES_FALLBACK
        est_s = n_resamples * sum(per_resample_s[n] for n in n_values)
        fallback_used = True
    print(f"[eval_v2_phase2] time_estimate_s={est_s:.0f} ({est_s/3600:.2f}h) "
          f"n_resamples={n_resamples} fallback_used={fallback_used}", flush=True)

    # ---- PREREG_v2.json: pin commit + config, compute sha256, write ----
    prereg_body = {
        "phase": "HEKBV3-BUILD003 Phase 2 (preregistered v2)",
        "commit_hash": git_commit_hash(),
        "data_dir": str(INDEX_ROOT.parent), "space_id": SPACE_ID,
        "n_values": list(n_values), "n_resamples": n_resamples, "fallback_used": fallback_used,
        "seed": v1_config.seed, "top_m": v1_config.top_m, "temperature_multiplier": v1_config.temperature_multiplier,
        "holdout_frac": 0.2, "n_bootstrap": v1_config.n_bootstrap, "bootstrap_seed": v1_config.bootstrap_seed,
        "use_calibration": use_calibration, "calibration_diagnosis": cal_diag,
        "tau_real": tau_real, "lambda_real": lam_real, "tau_shuffle": tau_shuffle, "lambda_shuffle": lam_shuffle,
        "inner_library_size": inner_library_size, "inner_query_size": inner_query_size, "inner_windows": inner_windows,
        "time_estimate_s": est_s,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    canonical = json.dumps(prereg_body, sort_keys=True).encode()
    sha256 = hashlib.sha256(canonical).hexdigest()
    prereg_body["sha256"] = sha256
    PREREG_PATH.write_text(json.dumps(prereg_body, indent=2))
    print(f"[eval_v2_phase2] PREREG_v2.json sha256={sha256}", flush=True)

    # ---- resumable checkpoint ----
    results: dict = {}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except Exception:
            results = {}
    rows_by_n: dict[int, list[dict]] = {int(k): v for k, v in results.get("rows_by_n", {}).items()}

    def checkpoint(tag: str) -> None:
        OUT.write_text(json.dumps({
            "note": "HEKBV3-BUILD003 Phase 2, preregistered v2 -- NOT v1's eval_results.json",
            "prereg_sha256": sha256, "n_values": list(n_values), "n_resamples": n_resamples,
            "rows_by_n": {str(n): rows_by_n[n] for n in rows_by_n},
            "status": tag, "elapsed_s": time.time() - t_start,
        }, indent=2))

    for n in n_values:
        if n in rows_by_n and len(rows_by_n[n]) == n_resamples * len(query_groups):
            print(f"[eval_v2_phase2] n={n} already checkpointed, skipping", flush=True)
            continue
        t_n = time.time()
        rows = []
        for r in range(n_resamples):
            resample_rng = np.random.default_rng((v1_config.seed, n, r))
            sampled = set(resample_rng.choice(pool, size=n, replace=False).tolist())
            kept_keys, kept_entries, t0, lib_cats_real = ecb._prefiltered_library(entries, keys, sampled, v1_config)
            lib_cats_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in kept_entries))
            alpha_shuffle = 1.0 / max(len(lib_cats_shuffle), 1)
            entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in kept_entries}
            freq_real_vec = np.array([freq_posterior_over(cat_by_group, sampled, categories, K)[c] for c in categories])
            freq_shuffle_vec = np.array([freq_posterior_over(shuffled_cat_of_group, sampled, categories, K)[c]
                                          for c in categories])

            for qgroup, qrows in query_sessions.items():
                real_losses, shuffle_losses, freq_losses = [], [], []
                p_real_t_windows = []
                for qstd, qoid, qtok, true_cat in qrows:
                    result = ecb.compose_mod.compose(
                        query_standardized=qstd, query_object_id=qoid, query_token_index=qtok,
                        keys=kept_keys, entries=kept_entries,
                        objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                        temperature=t0, top_m=v1_config.top_m, exclude_groups=None, skip_r1=True,
                        library_categories=lib_cats_real,
                    )
                    ti = cat_idx[true_cat]
                    p_real_vec = np.array([result.category_posterior.get(c, 0.0) for c in categories])
                    p_real_t = transform(p_real_vec, freq_real_vec, tau_real, lam_real)
                    real_losses.append(-math.log(max(p_real_t[ti], EPS_LOG)))
                    p_real_t_windows.append(p_real_t)

                    posterior_shuffle = ecb._posterior_from_neighbors(
                        result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                        entry_group_lookup, shuffled_cat_of_group, lib_cats_shuffle, alpha_shuffle,
                    )
                    p_shuffle_vec = np.array([posterior_shuffle.get(c, 0.0) for c in categories])
                    p_shuffle_t = transform(p_shuffle_vec, freq_shuffle_vec, tau_shuffle, lam_shuffle)
                    shuffle_losses.append(-math.log(max(p_shuffle_t[ti], EPS_LOG)))

                    freq_losses.append(-math.log(max(freq_real_vec[ti], EPS_LOG)))

                session_pred = int(np.argmax(np.mean(p_real_t_windows, axis=0)))
                session_true = cat_idx[qrows[0][3]]
                rows.append({
                    "resample": r, "group": qgroup,
                    "real_logloss": float(np.mean(real_losses)), "shuffle_logloss": float(np.mean(shuffle_losses)),
                    "freq_logloss": float(np.mean(freq_losses)),
                    "session_top1_correct": bool(session_pred == session_true),
                })
            print(f"[eval_v2_phase2] n={n} resample={r+1}/{n_resamples} elapsed_n_s={time.time()-t_n:.1f}", flush=True)

        rows_by_n[n] = rows
        checkpoint(tag=f"n={n} done")
        print(f"[eval_v2_phase2] n={n} COMPLETE elapsed_n_s={time.time()-t_n:.1f} "
              f"total_elapsed_s={time.time()-t_start:.1f}", flush=True)

    # ---- acceptance (paired per-row differences, bundle-level bootstrap) ----
    def diff_rows(rows, field_a, field_b):
        return [{"group": row["group"], "logloss": row[field_a] - row[field_b]} for row in rows]

    max_n = max(n_values)
    diff_a_max = diff_rows(rows_by_n[max_n], "real_logloss", "freq_logloss")
    diff_b_max = diff_rows(rows_by_n[max_n], "real_logloss", "shuffle_logloss")
    a_m, a_lo, a_hi = p0r.bootstrap_ci(diff_a_max, "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
    b_m, b_lo, b_hi = p0r.bootstrap_ci(diff_b_max, "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
    criterion_a = a_hi < 0.0
    criterion_b = b_hi < 0.0
    detected = criterion_a and criterion_b

    diff_b_by_n = {n: diff_rows(rows_by_n[n], "real_logloss", "shuffle_logloss") for n in n_values}
    diff_b_means = [p0r.bootstrap_ci(diff_b_by_n[n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)[0]
                    for n in n_values]
    slope_b = ecb.fit_slope(n_values, diff_b_means)
    slope_b_ci = ecb.slope_bootstrap_ci(diff_b_by_n, n_values, v1_config.n_bootstrap, v1_config.bootstrap_seed)

    real_means = [p0r.bootstrap_ci(rows_by_n[n], "real_logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)[0]
                  for n in n_values]
    changepoints = ecb.changepoint_scan(n_values, real_means)

    table = {}
    for n in n_values:
        um = float(math.log(K))
        fm, flo, fhi = p0r.bootstrap_ci(rows_by_n[n], "freq_logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        rm, rlo, rhi = p0r.bootstrap_ci(rows_by_n[n], "real_logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        sm, slo, shi = p0r.bootstrap_ci(rows_by_n[n], "shuffle_logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        n_sessions_total = len(rows_by_n[n])
        n_top1_correct = sum(1 for row in rows_by_n[n] if row["session_top1_correct"])
        table[n] = {
            "uniform": um, "freq_prior": {"mean": fm, "lo": flo, "hi": fhi},
            "real": {"mean": rm, "lo": rlo, "hi": rhi}, "shuffle": {"mean": sm, "lo": slo, "hi": shi},
            "session_top1_accuracy": n_top1_correct / n_sessions_total if n_sessions_total else None,
        }

    verdict = "detected" if detected else "この設計・この n では検出できなかった"

    final = {
        "prereg_sha256": sha256, "n_values": list(n_values), "n_resamples": n_resamples,
        "use_calibration": use_calibration, "tau_real": tau_real, "lambda_real": lam_real,
        "tau_shuffle": tau_shuffle, "lambda_shuffle": lam_shuffle,
        "table_by_n": {str(n): table[n] for n in n_values},
        "acceptance_at_max_n": {
            "max_n": max_n, "criterion_a_real_minus_freq_hi_lt_0": criterion_a,
            "criterion_b_real_minus_shuffle_hi_lt_0": criterion_b, "detected": detected, "verdict": verdict,
            "real_minus_freq_diff": {"mean": a_m, "lo": a_lo, "hi": a_hi},
            "real_minus_shuffle_diff": {"mean": b_m, "lo": b_lo, "hi": b_hi},
        },
        "slope_real_minus_shuffle_vs_log2n": slope_b, "slope_real_minus_shuffle_ci": list(slope_b_ci),
        "changepoints_real_logloss": changepoints,
        "elapsed_s": time.time() - t_start,
    }
    OUT.write_text(json.dumps({**json.loads(OUT.read_text()), **final}, indent=2))
    print(json.dumps(final, indent=2, default=str))
    print(f"[eval_v2_phase2] DONE total_elapsed_s={time.time()-t_start:.1f} verdict={verdict}", flush=True)


if __name__ == "__main__":
    main()
