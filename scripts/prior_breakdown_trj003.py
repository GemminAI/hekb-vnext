"""HEKBV3-BUILD001 差分4 追加: breaks down why P0's own prior_logloss (4.620) is worse than
uniform (log 12 = 2.485). Read-only; no exp_hekbv3 writes; compose() has no side effects.
Reuses scripts/p0_repro_trj003.py's fold replication (seed=20261004, k=6) and run_step().

1. P0's literal prior formula (window-count add-one Laplace over a FIXED K=12, engine.py
   line 321): per query, prior_logloss, whether the true category has zero TRAIN windows,
   and its train window frequency. Splits mean prior_logloss (and M2/kernel, from
   p0_repro's "P0-full" step) by absent-vs-not.
2. Same kfold6 split, three alternative prior smoothings instead of P0's window-count
   add-one: (a) session-count add-one (fixed K=12), (b) session-count alpha=1/K (K shrunk
   to the post-exclude library, reusing p0_repro's laplace_posterior_session_freq), (c) a
   direct mixture with uniform: p(c) = 0.5*(1/K) + 0.5*(session_freq(c)), K=12 fixed, no
   Laplace term. Table split by absent-vs-not.
3. Non-absent queries only, kfold6 split: kernel vs prior vs uniform, all three under (b)
   (this is exactly p0_repro's feature_dim=10/last_window/alpha_over_k/kfold6 step,
   filtered to the non-absent subset).
4. LOO (current default split): prior redefined as the OVERALL (not-exclude-own-group)
   session frequency, vs the existing exclude-own-group baseline.

No interpretation. Numbers and settings only.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))
import p0_repro_trj003 as p0r  # noqa: E402
from index import experience_index  # noqa: E402

L = p0r.L
EPS_LOG = p0r.EPS_LOG


def valid_window_count(n_steps: int) -> int:
    return max(0, n_steps - (L - 1))


def p0_window_prior(sessions, fold_of_group, cats_sorted):
    """Item 1: P0's literal window-count add-one Laplace prior, K=12 fixed."""
    ncat = len(cats_sorted)
    rows = []
    for q in sessions:
        own_fold = fold_of_group[q["group"]]
        train = [s for s in sessions if fold_of_group[s["group"]] != own_fold]
        win_by_cat = {c: 0 for c in cats_sorted}
        for s in train:
            win_by_cat[s["category"]] += valid_window_count(s["n_steps"])
        total = sum(win_by_cat.values())
        true_count = win_by_cat[q["category"]]
        prior_p = (true_count + 1.0) / (total + ncat)
        rows.append({
            "group": q["group"], "object_id": q["object_id"],
            "p0_prior_logloss": -math.log(max(prior_p, EPS_LOG)),
            "absent": true_count == 0, "train_window_freq": true_count / total if total else 0.0,
        })
    return rows


def session_count_prior(train, true_cat, cats_sorted, mode):
    ncat_fixed = len(cats_sorted)
    counts = {c: 0 for c in cats_sorted}
    for s in train:
        counts[s["category"]] += 1
    total = len(train)
    if mode == "add_one_fixedK":
        return (counts[true_cat] + 1.0) / (total + ncat_fixed)
    if mode == "alpha_over_k_shrink":
        present = sorted({s["category"] for s in train})
        k = max(len(present), 1)
        alpha = 1.0 / k
        if true_cat not in present:
            return alpha / (total + k * alpha)
        return (counts[true_cat] + alpha) / (total + k * alpha)
    if mode == "mix_uniform_0.5":
        freq = counts[true_cat] / total if total else 0.0
        return 0.5 * (1.0 / ncat_fixed) + 0.5 * freq
    raise ValueError(mode)


def main() -> None:
    sessions = p0r._load_all_sessions()
    assert len(sessions) == 48
    groups_sorted = sorted({s["group"] for s in sessions})
    fold_of_group = p0r.p0_fold_of_group(groups_sorted)
    cats_sorted = sorted({s["category"] for s in sessions})
    print(f"groups={len(groups_sorted)} categories={len(cats_sorted)} (fixed K=12 for P0-style items 1-2)")

    print("\n=== 1. P0 literal prior (window-count add-one, K=12 fixed), split by absent-from-train ===")
    rows1 = p0_window_prior(sessions, fold_of_group, cats_sorted)
    absent_rows = [r for r in rows1 if r["absent"]]
    present_rows = [r for r in rows1 if not r["absent"]]
    print(f"  n_absent={len(absent_rows)}/48 n_present={len(present_rows)}/48")
    print(f"  absent subset: mean_prior_logloss={np.mean([r['p0_prior_logloss'] for r in absent_rows]):.6f} "
          f"(n={len(absent_rows)})")
    print(f"  present subset: mean_prior_logloss={np.mean([r['p0_prior_logloss'] for r in present_rows]):.6f} "
          f"mean_train_window_freq={np.mean([r['train_window_freq'] for r in present_rows]):.6f} (n={len(present_rows)})")
    print(f"  overall mean_prior_logloss={np.mean([r['p0_prior_logloss'] for r in rows1]):.6f} "
          f"(P0 saved primary.json value: 4.620351)")

    print("\n  M2/kernel under the SAME split (p0_repro P0-full: 17port,session_mean,p0_ne,kfold6), same absent flag:")
    kernel_rows = p0r.run_step(sessions, 17, "session_mean", "p0_ne_add_one", "kfold6", fold_of_group=fold_of_group)
    absent_groups = {r["group"] for r in absent_rows}
    k_absent = [r for r in kernel_rows if r["group"] in absent_groups]
    k_present = [r for r in kernel_rows if r["group"] not in absent_groups]
    print(f"  kernel absent subset mean_kernel_logloss={np.mean([r['kernel_logloss'] for r in k_absent]):.6f} (n={len(k_absent)})")
    print(f"  kernel present subset mean_kernel_logloss={np.mean([r['kernel_logloss'] for r in k_present]):.6f} (n={len(k_present)})")

    print("\n=== 2. alternative prior smoothings, same kfold6 split, split by absent-vs-not ===")
    for mode in ("add_one_fixedK", "alpha_over_k_shrink", "mix_uniform_0.5"):
        vals_absent, vals_present = [], []
        for q in sessions:
            own_fold = fold_of_group[q["group"]]
            train = [s for s in sessions if fold_of_group[s["group"]] != own_fold]
            p = session_count_prior(train, q["category"], cats_sorted, mode)
            ll = -math.log(max(p, EPS_LOG))
            is_absent = q["group"] in absent_groups
            (vals_absent if is_absent else vals_present).append(ll)
        print(f"  [{mode}] absent mean={np.mean(vals_absent):.6f} (n={len(vals_absent)})  "
              f"present mean={np.mean(vals_present):.6f} (n={len(vals_present)})")

    print("\n=== 3. non-absent queries only, kfold6 split: kernel vs prior vs uniform (alpha_over_k_shrink for all three) ===")
    full_rows = p0r.run_step(sessions, 10, "last_window", "alpha_over_k", "kfold6", fold_of_group=fold_of_group)
    non_absent = [r for r in full_rows if r["group"] not in absent_groups]
    for field in ("kernel_logloss", "prior_logloss", "uniform_logloss"):
        m, lo, hi = p0r.bootstrap_ci(non_absent, field)
        print(f"  {field}: mean={m:.6f} 95%CI=[{lo:.6f},{hi:.6f}] (n={len(non_absent)})")

    print("\n=== 4. LOO (current default), prior = overall (not-exclude-own-group) session frequency ===")
    rows_overall, rows_exclude = [], []
    for q in sessions:
        counts_overall = {c: 0 for c in cats_sorted}
        for s in sessions:
            counts_overall[s["category"]] += 1
        total_overall = len(sessions)
        k = len(cats_sorted)
        alpha = 1.0 / k
        p_overall = (counts_overall[q["category"]] + alpha) / (total_overall + k * alpha)
        rows_overall.append({"group": q["group"], "logloss": -math.log(max(p_overall, EPS_LOG))})

        other = [s for s in sessions if s["group"] != q["group"]]
        present = sorted({s["category"] for s in other})
        k2 = max(len(present), 1)
        alpha2 = 1.0 / k2
        counts_ex = {c: 0 for c in present}
        for s in other:
            counts_ex[s["category"]] += 1
        p_ex = (counts_ex.get(q["category"], 0) + alpha2) / (len(other) + k2 * alpha2) if q["category"] in present else alpha2 / (len(other) + k2 * alpha2)
        rows_exclude.append({"group": q["group"], "logloss": -math.log(max(p_ex, EPS_LOG))})
    m1, lo1, hi1 = p0r.bootstrap_ci(rows_overall, "logloss")
    m2, lo2, hi2 = p0r.bootstrap_ci(rows_exclude, "logloss")
    print(f"  prior(overall, not excluding own group): mean={m1:.6f} 95%CI=[{lo1:.6f},{hi1:.6f}]")
    print(f"  prior(exclude own group, current baseline): mean={m2:.6f} 95%CI=[{lo2:.6f},{hi2:.6f}]")


if __name__ == "__main__":
    main()
