"""HEKBV3-BUILD002 事後分析2 §B (事前登録 v2, A の結果を見ずに固定した設計): compose() の R3
事後確率を p'=(1-λ)*normalize(p^τ) + λ*p_freq で再キャリブレーションしたうえで、n=48,96,192
(事前登録 v1 と同じ評価用の束・seed・T x1.0・top_m=64・特徴) を再評価する。

(τ,λ) の決め方: 事前登録 v1 の library_pool (69件を外側のdevとして既に除いた278束) のうち
さらに20%を「内側のクエリ」として split_groups (seed=config.seed, 同じ関数) で切り出し、
残り80%を「内側のlibrary」として丸ごと使う (n でサンプリングし直さない、一回だけの固定split)。
内側のクエリ窓に対する compose() を1回だけ呼び、real-label 事後と shuffle-label 事後を
(run_n_curve_real_and_shuffle と同じ理屈で) 同じ呼び出しから導出し、それぞれ別に
tau in {0.1,0.2,0.3,0.5,0.75,1}, lambda in {0,0.1,0.25,0.5,0.75} の格子で内側の対数損失を
最小化する (τ,λ) を選ぶ -- real用とshuffle用は別々に選ぶ (「公平」、本文の指示通り)。
外側の dev (69束) はこの選択に一切使わない。

本評価: n in (48,96,192) (事前に時間見積もり: 48/96/192 x 10 再抽出 ≈ 4.9時間、8時間以下なので
3点とも実施)、再抽出10回、real/shuffle とも選んだ (τ,λ) で変換してから bootstrap CI を取る。
uniform/頻度事前(無変換)/real(変換後)/shuffle(変換後) を同じ表に。

事前登録 v1 の結果ファイルは変更しない。出力は新しい JSON のみ。解釈はしない。数字と設定だけ。
"""
from __future__ import annotations

import dataclasses
import json
import math
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
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_eval_v2.json")

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


def main() -> None:
    t_start = time.time()
    v1_config = ecb.PreregisteredConfig()
    v2_config = dataclasses.replace(v1_config, n_values=(48, 96, 192), n_resamples=10)
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

    groups_sorted = sorted({e["group"] for e in entries})
    rng_shuf = np.random.default_rng(v1_config.seed)
    permuted_groups = rng_shuf.permutation(groups_sorted)
    shuffled_cat_of_group = dict(zip(groups_sorted, [cat_by_group[g] for g in permuted_groups]))

    print(f"[eval_v2] n_entries={len(entries)} K={K} library_pool={len(library_pool)} "
          f"dev_query_groups={len(query_groups)} elapsed_s={time.time()-t_start:.1f}", flush=True)

    # ---- (tau, lambda) selection on an INNER split of library_pool only (dev untouched) ----
    inner_library_pool, inner_query_groups = ecb.split_groups(pool, holdout_frac=0.2, seed=v1_config.seed)
    inner_query_sessions = build_query_sessions(entries, keys, inner_query_groups)
    inner_kept_keys, inner_kept_entries, inner_t0, inner_lib_categories_real = ecb._prefiltered_library(
        entries, keys, set(inner_library_pool), v1_config)
    inner_lib_categories_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in inner_kept_entries))
    inner_alpha_shuffle = 1.0 / max(len(inner_lib_categories_shuffle), 1)
    inner_entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in inner_kept_entries}

    inner_freq_real = freq_posterior_over(cat_by_group, inner_library_pool, categories, K)
    inner_freq_real_vec = np.array([inner_freq_real.get(c, 0.0) for c in categories])
    inner_freq_shuffle = freq_posterior_over(shuffled_cat_of_group, inner_library_pool, categories, K)
    inner_freq_shuffle_vec = np.array([inner_freq_shuffle.get(c, 0.0) for c in categories])

    print(f"[eval_v2] inner_library={len(inner_library_pool)} inner_query={len(inner_query_groups)} "
          f"elapsed_s={time.time()-t_start:.1f}", flush=True)

    inner_p_real, inner_p_shuffle, inner_true = [], [], []
    cat_idx = {c: i for i, c in enumerate(categories)}
    for qgroup, qrows in inner_query_sessions.items():
        for query_standardized, query_object_id, query_token_index, true_category in qrows:
            result = ecb.compose_mod.compose(
                query_standardized=query_standardized, query_object_id=query_object_id,
                query_token_index=query_token_index, keys=inner_kept_keys, entries=inner_kept_entries,
                objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                temperature=inner_t0, top_m=v1_config.top_m, exclude_groups=None, skip_r1=True,
                library_categories=inner_lib_categories_real,
            )
            inner_p_real.append(np.array([result.category_posterior.get(c, 0.0) for c in categories]))
            posterior_shuffle = ecb._posterior_from_neighbors(
                result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                inner_entry_group_lookup, shuffled_cat_of_group, inner_lib_categories_shuffle, inner_alpha_shuffle,
            )
            inner_p_shuffle.append(np.array([posterior_shuffle.get(c, 0.0) for c in categories]))
            inner_true.append(cat_idx[true_category])
    inner_P_real = np.stack(inner_p_real)
    inner_P_shuffle = np.stack(inner_p_shuffle)
    inner_TRUE = np.array(inner_true)
    idx_inner = np.arange(len(inner_TRUE))
    print(f"[eval_v2] inner_windows={len(inner_TRUE)} compose_done elapsed_s={time.time()-t_start:.1f}", flush=True)

    def grid_search(P, freq_vec):
        best = None
        grid = []
        for tau in TAUS:
            for lam in LAMBDAS:
                p_mixed = transform(P, freq_vec[None, :], tau, lam)
                loss = float(np.mean(-np.log(np.maximum(p_mixed[idx_inner, inner_TRUE], EPS_LOG))))
                grid.append({"tau": tau, "lambda": lam, "inner_logloss": loss})
                if best is None or loss < best["inner_logloss"]:
                    best = {"tau": tau, "lambda": lam, "inner_logloss": loss}
        return best, grid

    best_real, grid_real = grid_search(inner_P_real, inner_freq_real_vec)
    best_shuffle, grid_shuffle = grid_search(inner_P_shuffle, inner_freq_shuffle_vec)
    tau_real, lam_real = best_real["tau"], best_real["lambda"]
    tau_shuffle, lam_shuffle = best_shuffle["tau"], best_shuffle["lambda"]
    print(f"[eval_v2] selected real: tau={tau_real} lambda={lam_real} inner_logloss={best_real['inner_logloss']:.4f}", flush=True)
    print(f"[eval_v2] selected shuffle: tau={tau_shuffle} lambda={lam_shuffle} inner_logloss={best_shuffle['inner_logloss']:.4f}", flush=True)

    uniform_logloss = float(math.log(K))

    curve_real: dict[int, list[dict]] = {}
    curve_shuffle: dict[int, list[dict]] = {}
    freq_prior_curve: dict[int, list[dict]] = {}

    def checkpoint(tag: str) -> None:
        payload = {
            "note": "post-hoc exploratory re-evaluation (B / preregistered v2), NOT the v1 acceptance decision",
            "v1_n_values": list(v1_config.n_values), "v2_n_values": list(v2_config.n_values),
            "v2_n_resamples": v2_config.n_resamples, "K": K, "uniform_logloss": uniform_logloss,
            "inner_library_size": len(inner_library_pool), "inner_query_size": len(inner_query_groups),
            "inner_windows": len(inner_TRUE),
            "selected_real": best_real, "selected_shuffle": best_shuffle,
            "tau_lambda_grid_real": grid_real, "tau_lambda_grid_shuffle": grid_shuffle,
            "curve_real": {str(n): curve_real[n] for n in curve_real},
            "curve_shuffle": {str(n): curve_shuffle[n] for n in curve_shuffle},
            "freq_prior_curve": {str(n): freq_prior_curve[n] for n in freq_prior_curve},
            "status": tag, "elapsed_s": time.time() - t_start,
        }
        OUT.write_text(json.dumps(payload))

    checkpoint(tag="calibration done")

    for n in v2_config.n_values:
        t_n = time.time()
        rows_real, rows_shuffle, rows_freq = [], [], []
        for r in range(v2_config.n_resamples):
            resample_rng = np.random.default_rng((v1_config.seed, n, r))
            sampled = set(resample_rng.choice(pool, size=n, replace=False).tolist())
            kept_keys, kept_entries, t0, library_categories_real = ecb._prefiltered_library(
                entries, keys, sampled, v1_config)
            library_categories_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in kept_entries))
            alpha_shuffle = 1.0 / max(len(library_categories_shuffle), 1)
            entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in kept_entries}

            freq_real_vec = np.array([freq_posterior_over(cat_by_group, sampled, categories, K).get(c, 0.0)
                                       for c in categories])
            freq_shuffle_vec = np.array([freq_posterior_over(shuffled_cat_of_group, sampled, categories, K).get(c, 0.0)
                                          for c in categories])

            for qgroup, qrows in query_sessions.items():
                real_losses, shuffle_losses, freq_losses = [], [], []
                for query_standardized, query_object_id, query_token_index, true_category in qrows:
                    result = ecb.compose_mod.compose(
                        query_standardized=query_standardized, query_object_id=query_object_id,
                        query_token_index=query_token_index, keys=kept_keys, entries=kept_entries,
                        objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                        temperature=t0, top_m=v1_config.top_m, exclude_groups=None, skip_r1=True,
                        library_categories=library_categories_real,
                    )
                    p_real_vec = np.array([result.category_posterior.get(c, 0.0) for c in categories])
                    p_real_t = transform(p_real_vec, freq_real_vec, tau_real, lam_real)
                    ti = cat_idx[true_category]
                    real_losses.append(-math.log(max(p_real_t[ti], EPS_LOG)))

                    posterior_shuffle = ecb._posterior_from_neighbors(
                        result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                        entry_group_lookup, shuffled_cat_of_group, library_categories_shuffle, alpha_shuffle,
                    )
                    p_shuffle_vec = np.array([posterior_shuffle.get(c, 0.0) for c in categories])
                    p_shuffle_t = transform(p_shuffle_vec, freq_shuffle_vec, tau_shuffle, lam_shuffle)
                    shuffle_losses.append(-math.log(max(p_shuffle_t[ti], EPS_LOG)))

                    freq_losses.append(-math.log(max(freq_real_vec[ti], EPS_LOG)))
                rows_real.append({"group": qgroup, "logloss": float(np.mean(real_losses))})
                rows_shuffle.append({"group": qgroup, "logloss": float(np.mean(shuffle_losses))})
                rows_freq.append({"group": qgroup, "logloss": float(np.mean(freq_losses))})
            print(f"[eval_v2] n={n} resample={r+1}/{v2_config.n_resamples} elapsed_n_s={time.time()-t_n:.1f}", flush=True)

        curve_real[n] = rows_real
        curve_shuffle[n] = rows_shuffle
        freq_prior_curve[n] = rows_freq
        checkpoint(tag=f"n={n} done")
        print(f"[eval_v2] n={n} COMPLETE elapsed_n_s={time.time()-t_n:.1f} total_elapsed_s={time.time()-t_start:.1f}", flush=True)

    # ---- acceptance + changepoints (B's own definition) ----
    max_n = max(v2_config.n_values)
    real_m, real_lo, real_hi = p0r.bootstrap_ci(curve_real[max_n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
    shuf_m, shuf_lo, shuf_hi = p0r.bootstrap_ci(curve_shuffle[max_n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
    freq_m, freq_lo, freq_hi = p0r.bootstrap_ci(freq_prior_curve[max_n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
    below_freq_prior = real_hi < freq_m
    ci_separate = (real_hi < shuf_lo) or (real_lo > shuf_hi)
    detected = below_freq_prior and ci_separate

    means_real = [p0r.bootstrap_ci(curve_real[n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)[0]
                  for n in v2_config.n_values]
    changepoints = ecb.changepoint_scan(v2_config.n_values, means_real)

    table = {}
    for n in v2_config.n_values:
        rm, rlo, rhi = p0r.bootstrap_ci(curve_real[n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        sm, slo, shi = p0r.bootstrap_ci(curve_shuffle[n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        fm, flo, fhi = p0r.bootstrap_ci(freq_prior_curve[n], "logloss", seed=v1_config.bootstrap_seed, n_rep=v1_config.n_bootstrap)
        table[n] = {
            "uniform": uniform_logloss,
            "freq_prior": {"mean": fm, "lo": flo, "hi": fhi},
            "real_transformed": {"mean": rm, "lo": rlo, "hi": rhi},
            "shuffle_transformed": {"mean": sm, "lo": slo, "hi": shi},
        }

    final = {
        "note": "post-hoc exploratory re-evaluation (B / preregistered v2), NOT the v1 acceptance decision",
        "v2_n_values": list(v2_config.n_values), "v2_n_resamples": v2_config.n_resamples,
        "selected_real": best_real, "selected_shuffle": best_shuffle,
        "table_by_n": {str(n): table[n] for n in v2_config.n_values},
        "acceptance_at_max_n": {
            "max_n": max_n, "below_freq_prior": below_freq_prior, "ci_separate_real_shuffle": ci_separate,
            "detected": detected,
            "real": {"mean": real_m, "lo": real_lo, "hi": real_hi},
            "shuffle": {"mean": shuf_m, "lo": shuf_lo, "hi": shuf_hi},
            "freq_prior": {"mean": freq_m, "lo": freq_lo, "hi": freq_hi},
        },
        "changepoints": changepoints,
        "elapsed_s": time.time() - t_start,
    }
    OUT.write_text(json.dumps({**json.loads(OUT.read_text()), **final}, indent=2))
    print(json.dumps(final, indent=2, default=str))
    print(f"[eval_v2] DONE total_elapsed_s={time.time()-t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
