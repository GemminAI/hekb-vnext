"""HEKBV3-BUILD002 評価: tests eval_curve_boardrecords.py against SYNTHETIC data only --
sessions/ ingestion has not happened yet, so this module has not been run against real
TRJ003/BoardRecord data (per the task's own "それまでは評価コードのテスト（合成データ）
まで"). Exercises run_n_curve / fit_slope / slope_bootstrap_ci / check_acceptance end to end
on a strongly-separable synthetic dataset (should detect) and a pure-noise one (should not
over-claim detection), using small n_values so the test runs fast."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import eval_curve_boardrecords as ecb  # noqa: E402


def _make_synthetic(n_groups: int, n_categories: int, d: int, signal: float, seed: int,
                     windows_per_group: int = 4):
    """n_groups bundles, one category each (round-robin), d-dim standardized-ish features.
    Each group's rows cluster around a per-category center (scaled by `signal`); `signal=0`
    is pure noise (no category structure)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(n_categories, d)) * signal
    entries = []
    rows = []
    query_rows_by_group: dict[str, list[tuple]] = {}
    for gi in range(n_groups):
        group = f"g{gi}"
        cat = f"cat{gi % n_categories}"
        center = centers[gi % n_categories]
        group_query_rows = []
        for w in range(windows_per_group):
            vec = center + rng.normal(size=d)
            rows.append(vec)
            entries.append({"object_id": f"{group}-obj", "token_index": w, "group": group, "category": cat})
            group_query_rows.append((vec, f"{group}-obj", w, cat))
        query_rows_by_group[group] = group_query_rows
    keys = np.stack(rows, axis=0)
    return entries, keys, query_rows_by_group


def test_split_groups_disjoint_and_deterministic():
    groups = [f"g{i}" for i in range(30)]
    pool1, query1 = ecb.split_groups(groups, holdout_frac=0.2, seed=42)
    pool2, query2 = ecb.split_groups(groups, holdout_frac=0.2, seed=42)
    assert pool1 == pool2 and query1 == query2
    assert set(pool1) & set(query1) == set()
    assert set(pool1) | set(query1) == set(groups)
    assert len(query1) == 6


def test_run_n_curve_separable_data_meets_acceptance():
    entries, keys, query_rows_by_group = _make_synthetic(
        n_groups=40, n_categories=4, d=6, signal=6.0, seed=1, windows_per_group=3,
    )
    groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(groups, holdout_frac=0.25, seed=7)
    query_sessions = {g: query_rows_by_group[g] for g in query_groups}

    config = ecb.PreregisteredConfig(n_values=(4, 8, 16, 24), n_resamples=5, top_m=16,
                                      seed=123, n_bootstrap=200, bootstrap_seed=0)
    curve_real = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=False)
    curve_shuffle = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=True)

    assert set(curve_real.keys()) == set(config.n_values)
    for n in config.n_values:
        assert len(curve_real[n]) == config.n_resamples * len(query_sessions)

    uniform_logloss = float(np.log(4))  # K=4 categories
    acceptance = ecb.check_acceptance(curve_real, curve_shuffle, config.n_values, uniform_logloss, config)
    assert acceptance["detected"] is True, acceptance
    assert acceptance["criterion_a"] is True
    assert acceptance["criterion_b"] is True
    assert acceptance["slope"] < 0


def test_run_n_curve_pure_noise_does_not_claim_detection():
    entries, keys, query_rows_by_group = _make_synthetic(
        n_groups=40, n_categories=4, d=6, signal=0.0, seed=2, windows_per_group=3,
    )
    groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(groups, holdout_frac=0.25, seed=7)
    query_sessions = {g: query_rows_by_group[g] for g in query_groups}

    config = ecb.PreregisteredConfig(n_values=(4, 8, 16, 24), n_resamples=5, top_m=16,
                                      seed=321, n_bootstrap=200, bootstrap_seed=0)
    curve_real = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=False)
    curve_shuffle = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=True)

    uniform_logloss = float(np.log(4))
    acceptance = ecb.check_acceptance(curve_real, curve_shuffle, config.n_values, uniform_logloss, config)
    assert acceptance["detected"] is False, acceptance


def test_run_n_curve_rejects_query_library_overlap():
    entries, keys, query_rows_by_group = _make_synthetic(
        n_groups=10, n_categories=2, d=3, signal=1.0, seed=3, windows_per_group=2,
    )
    config = ecb.PreregisteredConfig(n_values=(2, 4), n_resamples=2, top_m=8, seed=1, n_bootstrap=50)
    with pytest.raises(ValueError, match="overlap"):
        ecb.run_n_curve(entries, keys, ["g0", "g1", "g2"], {"g0": query_rows_by_group["g0"]}, config)


def test_fit_slope_sign():
    assert ecb.fit_slope((2, 4, 8), [3.0, 2.0, 1.0]) < 0
    assert ecb.fit_slope((2, 4, 8), [1.0, 2.0, 3.0]) > 0


def test_combined_matches_separate_passes():
    """HEKBV3-BUILD002 続き §4 [perf]: run_n_curve_real_and_shuffle's single-compose-call
    shortcut must produce byte-identical logloss values to two independent
    run_n_curve(shuffle=False) / run_n_curve(shuffle=True) calls -- this is the
    verification the combined function's own docstring promises before it's trusted on the
    real 1,296-session run."""
    entries, keys, query_rows_by_group = _make_synthetic(
        n_groups=30, n_categories=5, d=5, signal=3.0, seed=11, windows_per_group=4,
    )
    groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(groups, holdout_frac=0.3, seed=9)
    query_sessions = {g: query_rows_by_group[g] for g in query_groups}

    config = ecb.PreregisteredConfig(n_values=(3, 7, 15), n_resamples=4, top_m=10,
                                      seed=555, n_bootstrap=50, bootstrap_seed=0)

    combined_real, combined_shuffle = ecb.run_n_curve_real_and_shuffle(
        entries, keys, library_pool, query_sessions, config)
    separate_real = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=False)
    separate_shuffle = ecb.run_n_curve(entries, keys, library_pool, query_sessions, config, shuffle=True)

    for n in config.n_values:
        combined_r = sorted((r["group"], r["logloss"]) for r in combined_real[n])
        separate_r = sorted((r["group"], r["logloss"]) for r in separate_real[n])
        for (g1, l1), (g2, l2) in zip(combined_r, separate_r):
            assert g1 == g2
            assert l1 == pytest.approx(l2, abs=1e-12)

        combined_s = sorted((r["group"], r["logloss"]) for r in combined_shuffle[n])
        separate_s = sorted((r["group"], r["logloss"]) for r in separate_shuffle[n])
        for (g1, l1), (g2, l2) in zip(combined_s, separate_s):
            assert g1 == g2
            assert l1 == pytest.approx(l2, abs=1e-12)
