"""HEKBV3-BUILD002 事後分析 §3 (探索、判定ではない): compose() の入力特徴 (72次元の
標準化キー) に、カテゴリーごとの固定オフセット (delta * per_dim_std * 固定ランダム単位方向)
を加算した合成信号つきデータで、事前登録と同じ手順 (split_groups の同じ seed・同じ
query_groups、run_n_curve_real_and_shuffle と同じ library サンプリング) を n=24,48 だけで
再実行する。

合成信号の定義 (開示): カテゴリー c ごとに固定のランダム単位方向 direction_c (seed固定) を
1本引き、offset_c = delta * std_vec * direction_c (std_vec は全エントリに対する各次元の標準偏差、
要素積)。その offset_c を、その カテゴリーを持つ「すべての」エントリ (library 側・query 側の
両方) のキーに加える -- query 側にも加えないと、合成した分離がクエリ側の特徴と整合しなくなり、
「signal を強めたら検出力が上がるか」という設問そのものに答えられなくなるため。これは
disclosed な実装選択で、「library の特徴に足す」という元の記述を字面通りに library のみに
限定した場合との違いを明記する。

delta in (0.5, 1.0, 2.0) x n in (24, 48) の 6 ジョブをチェックポイントしながら直列実行する。
n=48 (事前登録 run の実測 4855.1s) が 1 時間を超えるため、元の (48, 96) を (24, 48) に
差し替える、というタスク指示のフォールバックを適用済み (このスクリプトの対象 n はその結果)。

事前登録の結果ファイルは変更しない。出力は新しい JSON のみ。解釈はしない。数字と設定だけ。
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
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_power_analysis.json")

N_VALUES = (24, 48)
DELTAS = (0.5, 1.0, 2.0)
DIRECTION_SEED_LABEL = "posthoc_power_analysis_directions_v1"


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
    t_start = time.time()
    config = ecb.PreregisteredConfig()
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)
    d = keys.shape[1]
    print(f"[posthoc_power] n_entries={len(entries)} d={d} elapsed_s={time.time()-t_start:.1f}", flush=True)

    all_groups = sorted({e["group"] for e in entries})
    library_pool, query_groups = ecb.split_groups(all_groups, holdout_frac=0.2, seed=config.seed)
    query_sessions = build_query_sessions(entries, keys, query_groups)
    pool = sorted(library_pool)

    categories = sorted({e["category"] for e in entries})
    std_vec = np.std(keys, axis=0)
    dir_seed = abs(hash(DIRECTION_SEED_LABEL)) % (2**32)
    rng_dir = np.random.default_rng((config.seed, dir_seed))
    direction = {}
    for c in categories:
        v = rng_dir.normal(size=d)
        v = v / np.linalg.norm(v)
        direction[c] = v

    category_of_entry = [e["category"] for e in entries]
    category_idx = {c: i for i, c in enumerate(categories)}
    offset_by_cat_idx = np.stack([std_vec * direction[c] for c in categories], axis=0)  # (K, d)
    cat_idx_of_entry = np.array([category_idx[c] for c in category_of_entry])

    results: dict[str, dict] = {}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text())
        except Exception:
            results = {}

    def checkpoint() -> None:
        OUT.write_text(json.dumps({
            "note": "post-hoc exploratory analysis, NOT the preregistered acceptance decision",
            "n_values": N_VALUES, "deltas": DELTAS, "n_resamples": config.n_resamples,
            "signal_construction": ("offset_c = delta * std_vec * fixed_unit_random_direction_c, "
                                     "added to EVERY entry (library and query) of category c -- "
                                     "see module docstring for the disclosed deviation from a "
                                     "library-only reading"),
            "results_by_delta_n": results,
            "elapsed_s": time.time() - t_start,
        }, indent=2))

    for delta in DELTAS:
        synthetic_keys = keys + delta * offset_by_cat_idx[cat_idx_of_entry]
        synthetic_query_sessions = build_query_sessions(entries, synthetic_keys, query_groups)
        for n in N_VALUES:
            key = f"delta={delta}_n={n}"
            if key in results:
                print(f"[posthoc_power] {key} already checkpointed, skipping", flush=True)
                continue
            t_job = time.time()
            rows_real, rows_shuffle = [], []
            groups_sorted = sorted({e["group"] for e in entries})
            rng = np.random.default_rng(config.seed)
            permuted_groups = rng.permutation(groups_sorted)
            cat_by_group = {}
            for g in groups_sorted:
                cat_by_group[g] = next(e["category"] for e in entries if e["group"] == g)
            shuffled_cat_of_group = dict(zip(groups_sorted, [cat_by_group[g] for g in permuted_groups]))

            for r in range(config.n_resamples):
                resample_rng = np.random.default_rng((config.seed, n, r))
                sampled = set(resample_rng.choice(pool, size=n, replace=False).tolist())
                kept_keys, kept_entries, t0, library_categories_real = ecb._prefiltered_library(
                    entries, synthetic_keys, sampled, config)
                library_categories_shuffle = sorted(set(shuffled_cat_of_group[e["group"]] for e in kept_entries))
                alpha_shuffle = 1.0 / max(len(library_categories_shuffle), 1)
                entry_group_lookup = {(e["object_id"], e["token_index"]): e["group"] for e in kept_entries}

                for qgroup, qrows in synthetic_query_sessions.items():
                    real_losses, shuffle_losses = [], []
                    for query_standardized, query_object_id, query_token_index, true_category in qrows:
                        result = ecb.compose_mod.compose(
                            query_standardized=query_standardized, query_object_id=query_object_id,
                            query_token_index=query_token_index, keys=kept_keys, entries=kept_entries,
                            objects_dir=ecb.DUMMY_PATH, blobs_root=ecb.DUMMY_PATH,
                            temperature=t0, top_m=config.top_m, exclude_groups=None, skip_r1=True,
                            library_categories=library_categories_real,
                        )
                        p_real = result.category_posterior.get(true_category, 0.0)
                        real_losses.append(-math.log(max(p_real, ecb.EPS_LOG)))

                        posterior_shuffle = ecb._posterior_from_neighbors(
                            result.weights, result.neighbor_object_ids, result.neighbor_token_indices,
                            entry_group_lookup, shuffled_cat_of_group, library_categories_shuffle, alpha_shuffle,
                        )
                        p_shuffle = posterior_shuffle.get(true_category, 0.0)
                        shuffle_losses.append(-math.log(max(p_shuffle, ecb.EPS_LOG)))
                    rows_real.append({"group": qgroup, "logloss": float(np.mean(real_losses))})
                    rows_shuffle.append({"group": qgroup, "logloss": float(np.mean(shuffle_losses))})
                print(f"[posthoc_power] {key} resample={r+1}/{config.n_resamples} "
                      f"elapsed_job_s={time.time()-t_job:.1f}", flush=True)

            results[key] = {"delta": delta, "n": n, "curve_real": rows_real, "curve_shuffle": rows_shuffle,
                             "elapsed_job_s": time.time() - t_job}
            checkpoint()
            print(f"[posthoc_power] {key} COMPLETE elapsed_job_s={time.time()-t_job:.1f} "
                  f"total_elapsed_s={time.time()-t_start:.1f}", flush=True)

    checkpoint()
    print(f"[posthoc_power] DONE total_elapsed_s={time.time()-t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
