"""HEKBV3-BUILD002 事後分析2 §A (探索、判定ではない): compose() の R3 事後確率が過信かどうかの
診断。n=192、事前登録 v1 と同じ seed の最初の3再抽出 (r=0,1,2) だけを対象に、各クエリ窓の
compose() 事後確率ベクトル (全K=12カテゴリーに展開、絶対にその場で捨てずメモリ上に保持) を
使って、argmax正答率・確信度キャリブレーション・損失分解・τ冪変換+頻度事前混合のグリッド
探索を計算する。shuffle はこの診断には使わない (A節の指示に無い)。

library の再抽出は run_ncurve_eval.py / eval_curve_boardrecords.run_n_curve と同じ
(config.seed, n, r) の rng。頻度事前は posthoc_freq_prior.py と同じ構成 (alpha=1/K、
サンプルした n 個の束のカテゴリー頻度)。

事前登録 v1 の結果ファイルは変更しない。出力は新しい JSON のみ。解釈はしない。数字と設定だけ。
"""
from __future__ import annotations

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

INDEX_ROOT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/index")
SPACE_ID = "ec435e5ffc9f46cff1e0f5ef2a00a233367ff091884d4cbdc2b36cc56d913e27"
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_diag_a.json")

N_DIAG_RESAMPLES = 3
N_VAL = 192
TAUS = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
LAMBDAS = (0.0, 0.1, 0.25, 0.5)
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


def main() -> None:
    t0 = time.time()
    config = ecb.PreregisteredConfig()
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)
    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=0.2, seed=config.seed)
    query_sessions = build_query_sessions(entries, keys, query_groups)
    pool = sorted(library_pool)

    cat_by_group = {}
    for e in entries:
        cat_by_group.setdefault(e["group"], e["category"])
    categories = sorted({cat_by_group[g] for g in all_groups})
    K = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}
    alpha = 1.0 / K

    print(f"[diag_a] n_entries={len(entries)} K={K} library_pool={len(library_pool)} "
          f"query_groups={len(query_groups)} elapsed_s={time.time()-t0:.1f}", flush=True)

    all_p, all_true, all_group, all_resample, all_n_lib_cats = [], [], [], [], []
    freq_by_resample = []

    for r in range(N_DIAG_RESAMPLES):
        t_r = time.time()
        resample_rng = np.random.default_rng((config.seed, N_VAL, r))
        sampled = resample_rng.choice(pool, size=N_VAL, replace=False).tolist()
        kept_keys, kept_entries, t_temp, library_categories_real = ecb._prefiltered_library(
            entries, keys, set(sampled), config)
        all_n_lib_cats.append(len(library_categories_real))

        counts = {c: 0 for c in categories}
        for g in sampled:
            counts[cat_by_group[g]] += 1
        freq_posterior = {c: (counts[c] + alpha) / (N_VAL + K * alpha) for c in categories}
        freq_vec = np.array([freq_posterior[c] for c in categories])
        freq_by_resample.append(freq_vec)

        for qgroup, qrows in query_sessions.items():
            for query_standardized, query_object_id, query_token_index, true_category in qrows:
                result = ecb.compose_mod.compose(
                    query_standardized=query_standardized, query_object_id=query_object_id,
                    query_token_index=query_token_index, keys=kept_keys, entries=kept_entries,
                    objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                    temperature=t_temp, top_m=config.top_m, exclude_groups=None, skip_r1=True,
                    library_categories=library_categories_real,
                )
                p_vec = np.array([result.category_posterior.get(c, 0.0) for c in categories])
                all_p.append(p_vec)
                all_true.append(cat_idx[true_category])
                all_group.append(qgroup)
                all_resample.append(r)
        print(f"[diag_a] resample={r} n_lib_categories={len(library_categories_real)} "
              f"elapsed_r_s={time.time()-t_r:.1f} total_elapsed_s={time.time()-t0:.1f}", flush=True)

    P = np.stack(all_p)  # (N, K)
    TRUE = np.array(all_true)
    RESAMPLE = np.array(all_resample)
    N = P.shape[0]
    print(f"[diag_a] N_windows={N} compose_done elapsed_s={time.time()-t0:.1f}", flush=True)

    # ---- A.1 argmax accuracy: window-level and session-level (mean posterior per
    # (resample, query_group), then argmax) ----
    pred_window = np.argmax(P, axis=1)
    window_acc_compose = float(np.mean(pred_window == TRUE))

    session_true, session_pred_compose, session_pred_freq = [], [], []
    freq_argmax_by_resample = [int(np.argmax(v)) for v in freq_by_resample]
    idx_all = np.arange(N)
    for r in range(N_DIAG_RESAMPLES):
        mask_r = RESAMPLE == r
        groups_r = sorted({all_group[i] for i in idx_all[mask_r]})
        for g in groups_r:
            mask_g = mask_r & (np.array(all_group) == g)
            session_true.append(int(TRUE[mask_g][0]))
            session_pred_compose.append(int(np.argmax(P[mask_g].mean(axis=0))))
            session_pred_freq.append(freq_argmax_by_resample[r])
    session_true = np.array(session_true)
    session_pred_compose = np.array(session_pred_compose)
    session_pred_freq = np.array(session_pred_freq)
    session_acc_compose = float(np.mean(session_pred_compose == session_true))
    session_acc_freq = float(np.mean(session_pred_freq == session_true))
    window_acc_freq = float(np.mean(
        np.array([freq_argmax_by_resample[r] for r in all_resample]) == TRUE))

    # ---- A.2 confidence calibration (window-level) ----
    max_p = P.max(axis=1)
    mean_confidence = float(max_p.mean())
    confidence_minus_accuracy = mean_confidence - window_acc_compose
    p_true = P[idx_all, TRUE]
    frac_p_true_below_001 = float(np.mean(p_true < 0.01))

    bin_edges = np.linspace(0.0, 1.0, 11)
    bins = np.digitize(max_p, bin_edges[1:-1], right=False)
    reliability = []
    ece = 0.0
    for b in range(10):
        mask_b = bins == b
        n_b = int(mask_b.sum())
        if n_b == 0:
            reliability.append({"bin": [float(bin_edges[b]), float(bin_edges[b+1])], "n": 0,
                                 "avg_confidence": None, "accuracy": None})
            continue
        conf_b = float(max_p[mask_b].mean())
        acc_b = float(np.mean(pred_window[mask_b] == TRUE[mask_b]))
        reliability.append({"bin": [float(bin_edges[b]), float(bin_edges[b+1])], "n": n_b,
                             "avg_confidence": conf_b, "accuracy": acc_b})
        ece += (n_b / N) * abs(acc_b - conf_b)

    # ---- A.3 loss decomposition ----
    logloss_i = -np.log(np.maximum(p_true, EPS_LOG))
    correct_mask = pred_window == TRUE
    frac_correct = float(correct_mask.mean())
    frac_incorrect = 1.0 - frac_correct
    mean_loss_correct = float(logloss_i[correct_mask].mean()) if correct_mask.any() else None
    mean_loss_incorrect = float(logloss_i[~correct_mask].mean()) if (~correct_mask).any() else None
    contribution_correct = (mean_loss_correct * frac_correct) if mean_loss_correct is not None else None
    contribution_incorrect = (mean_loss_incorrect * frac_incorrect) if mean_loss_incorrect is not None else None
    overall_mean_logloss = float(logloss_i.mean())

    # ---- A.4 tau power-scaling + frequency-prior mixing grid (dev windows) ----
    freq_vec_per_window = np.stack([freq_by_resample[r] for r in all_resample])  # (N, K)
    grid = []
    for tau in TAUS:
        p_tau = np.power(P, tau)
        p_tau_norm = p_tau / p_tau.sum(axis=1, keepdims=True)
        for lam in LAMBDAS:
            p_mixed = (1 - lam) * p_tau_norm + lam * freq_vec_per_window
            p_mixed_true = p_mixed[idx_all, TRUE]
            loss = float(np.mean(-np.log(np.maximum(p_mixed_true, EPS_LOG))))
            grid.append({"tau": tau, "lambda": lam, "mean_logloss": loss})

    out = {
        "note": "post-hoc exploratory diagnosis (A), NOT the preregistered acceptance decision",
        "n_value": N_VAL, "n_diag_resamples": N_DIAG_RESAMPLES, "N_windows": N, "K": K,
        "n_library_categories_by_resample": all_n_lib_cats,
        "window_accuracy_compose_argmax": window_acc_compose,
        "window_accuracy_freq_argmax": window_acc_freq,
        "session_accuracy_compose_argmax": session_acc_compose,
        "session_accuracy_freq_argmax": session_acc_freq,
        "n_session_instances": len(session_true),
        "mean_confidence_maxp": mean_confidence,
        "confidence_minus_window_accuracy": confidence_minus_accuracy,
        "frac_p_true_below_0.01": frac_p_true_below_001,
        "ece_10bin": ece,
        "reliability_10bin": reliability,
        "overall_mean_logloss": overall_mean_logloss,
        "frac_correct": frac_correct, "frac_incorrect": frac_incorrect,
        "mean_loss_correct": mean_loss_correct, "mean_loss_incorrect": mean_loss_incorrect,
        "contribution_correct": contribution_correct, "contribution_incorrect": contribution_incorrect,
        "tau_lambda_grid": grid,
        "elapsed_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2))

    # ---- decision rule for B ----
    rule_i = session_acc_compose > session_acc_freq
    rule_ii = confidence_minus_accuracy >= 0.10
    proceed_to_b = rule_i and rule_ii
    out["decision_rule_i_session_acc_compose_gt_freq"] = rule_i
    out["decision_rule_ii_overconfidence_ge_0.10"] = rule_ii
    out["proceed_to_B"] = proceed_to_b
    OUT.write_text(json.dumps(out, indent=2))

    print(json.dumps({k: v for k, v in out.items() if k != "tau_lambda_grid" and k != "reliability_10bin"},
                      indent=2, default=str))
    print(f"[diag_a] DONE elapsed_s={time.time()-t0:.1f} proceed_to_B={proceed_to_b}", flush=True)


if __name__ == "__main__":
    main()
