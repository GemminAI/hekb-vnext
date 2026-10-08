"""HEKBV3-BUILD002 事後分析 §4 (探索、判定ではない): scripts/power_trj003.py の run_step1
(HEKBV3-BUILD001 差分4、48 件) と同じ手順 -- セッション平均特徴 (1点/セッション)、束 (group)
単位 leave-one-out の距離行列、kNN(k=1,3) top-1 分類、束単位ラベル permutation (N_PERM=1000)
-- を、同じ BoardRecord インデックスの全 1,296 セッションに対して再計算する。

[perf, disclosed]: 元のコードは acc() 呼び出し毎 (permutation 1000 回 x k=1,3 で2000回) に
argsort(dist2, axis=1) をラベル無関係に再計算していた (dist2 はラベルに依存しないため、
distinct_groups を permute してもソート順序そのものは変わらない)。ここでは1回だけソートし、
k 近傍のインデックス部分配列だけを使い回す -- 数値結果は変えない。

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

INDEX_ROOT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/index")
SPACE_ID = "ec435e5ffc9f46cff1e0f5ef2a00a233367ff091884d4cbdc2b36cc56d913e27"
OUT = Path("/media/psf/HEKBv3/exp_hekbv3_n_curve/posthoc_session_topk.json")

PERM_SEED = 1
N_PERM = 1000


def knn_predict_all_from_order(order: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    n = order.shape[0]
    preds = np.empty(n, dtype=labels.dtype)
    for i in range(n):
        neigh = order[i, :k]
        vals, counts = np.unique(labels[neigh], return_counts=True)
        preds[i] = vals[np.argmax(counts)]
    return preds


def main() -> None:
    t0 = time.time()
    keys, entries, meta = experience_index.load_index(INDEX_ROOT, SPACE_ID)

    by_object: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        by_object.setdefault(e["object_id"], []).append(i)
    object_ids = sorted(by_object.keys())
    n = len(object_ids)
    print(f"[posthoc_session_topk] n_sessions={n} elapsed_s={time.time()-t0:.1f}", flush=True)

    category_of_object = {}
    group_of_object = {}
    for oid in object_ids:
        idxs = by_object[oid]
        category_of_object[oid] = entries[idxs[0]]["category"]
        group_of_object[oid] = entries[idxs[0]]["group"]

    feats = np.stack([keys[by_object[oid]].mean(0) for oid in object_ids])
    cats_sorted = sorted({category_of_object[oid] for oid in object_ids})
    cat_idx = {c: i for i, c in enumerate(cats_sorted)}
    labels = np.array([cat_idx[category_of_object[oid]] for oid in object_ids])
    groups = np.array([group_of_object[oid] for oid in object_ids])

    sq = np.sum(feats ** 2, axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (feats @ feats.T)
    group_mask = groups[:, None] == groups[None, :]
    dist2_loo = np.where(group_mask, np.inf, dist2)
    order = np.argsort(dist2_loo, axis=1)  # computed once [perf, disclosed]; label-independent
    print(f"[posthoc_session_topk] distance+order built elapsed_s={time.time()-t0:.1f}", flush=True)

    def acc(lab, k):
        pred = knn_predict_all_from_order(order, lab, k)
        return pred, float((pred == lab).mean())

    pred_k1, obs_k1 = acc(labels, 1)
    pred_k3, obs_k3 = acc(labels, 3)
    chance = 1.0 / len(cats_sorted)
    print(f"[posthoc_session_topk] obs_k1={obs_k1:.4f} obs_k3={obs_k3:.4f} chance={chance:.4f} "
          f"elapsed_s={time.time()-t0:.1f}", flush=True)

    # per-category recall (top-1, k=1 and k=3)
    recall_k1 = {}
    recall_k3 = {}
    for c, ci in cat_idx.items():
        mask = labels == ci
        n_c = int(mask.sum())
        recall_k1[c] = {"n": n_c, "recall": float((pred_k1[mask] == ci).mean()) if n_c else None}
        recall_k3[c] = {"n": n_c, "recall": float((pred_k3[mask] == ci).mean()) if n_c else None}

    distinct_groups = sorted(set(groups))
    rng = np.random.default_rng(PERM_SEED)
    group_to_label = {g: labels[np.where(groups == g)[0][0]] for g in distinct_groups}
    null_k1, null_k3 = [], []
    for _ in range(N_PERM):
        perm_labels_by_group = dict(zip(distinct_groups,
                                         rng.permutation([group_to_label[g] for g in distinct_groups])))
        lab_shuf = np.array([perm_labels_by_group[g] for g in groups])
        null_k1.append(acc(lab_shuf, 1)[1])
        null_k3.append(acc(lab_shuf, 3)[1])
    p_k1 = float((np.array(null_k1) >= obs_k1).mean())
    p_k3 = float((np.array(null_k3) >= obs_k3).mean())
    print(f"[posthoc_session_topk] permutation p-values (n={N_PERM}, group-level): "
          f"kNN1 p={p_k1:.4f} kNN3 p={p_k3:.4f} elapsed_s={time.time()-t0:.1f}", flush=True)

    out = {
        "note": "post-hoc exploratory analysis, NOT the preregistered acceptance decision",
        "n_sessions": n, "n_groups": len(distinct_groups), "n_categories": len(cats_sorted),
        "categories": cats_sorted, "chance": chance,
        "obs_knn1_accuracy": obs_k1, "obs_knn3_accuracy": obs_k3,
        "permutation_n": N_PERM, "permutation_seed": PERM_SEED,
        "permutation_p_knn1": p_k1, "permutation_p_knn3": p_k3,
        "null_knn1_mean": float(np.mean(null_k1)), "null_knn1_std": float(np.std(null_k1)),
        "null_knn3_mean": float(np.mean(null_k3)), "null_knn3_std": float(np.std(null_k3)),
        "recall_by_category_knn1": recall_k1,
        "recall_by_category_knn3": recall_k3,
        "elapsed_s": time.time() - t0,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2, default=str))
    print(f"[posthoc_session_topk] DONE elapsed_s={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
