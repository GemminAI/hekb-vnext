"""POST /v3/compose: read-only (H7, K7) Softmax + Gaussian-kernel retrieval over the
experience index (HEKBv3 改修設計書 v0.1 §4.2-§4.5). No side effects: never touches
recall_count, objects, blobs, relations, or meta.

[HEKBV3-BUILD001, disclosed]: the design's §4.3 spatial blend averages each library
entry's OWN tangent-space displacement v_{h,i} = log_{x_t,i}(x_{t+h,i}) (computed at that
entry's own base point, not transported to the query's point) and then applies exp_map at
the QUERY's own current position -- this is the literal formula in §4.3/§4.4, read as
written (no parallel transport, which §4.3 itself does not mention); it is an
approximation that degrades as library points move further from the query, acceptable for
"窓の局所の合成のみ" (the design's own stated default scope).

[HEKBV3-BUILD001 差分]: R3's Laplace smoothing is applied over every category present in
the post-exclude library (K categories), including ones absent from top_m -- not just the
categories that happen to appear among the top_m neighbours (the original build's
behaviour, corrected here). alpha defaults to 1/K (`compose()`'s `laplace_alpha` param
overrides it); there is no separate epsilon floor for "entirely absent from top_m" --
such a category receives the same alpha as any other, per the corrected formula
p(c) = (sum_{i in top_m} w_i*1[c_i=c] + alpha) / (1 + K*alpha).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from storage import trajectory_blob
from storage.blobs import get_blob

HORIZONS = (1, 4, 8)


class ComposeError(ValueError):
    pass


def log_map(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Riemannian log map on the unit sphere: tangent vector at x pointing toward y."""
    cos_theta = float(np.clip(np.dot(x, y), -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros_like(x)
    return (theta / math.sin(theta)) * (y - cos_theta * x)


def exp_map(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Riemannian exp map on the unit sphere: walk from x along tangent vector v."""
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return x.copy()
    return math.cos(norm) * x + math.sin(norm) * (v / norm)


@dataclass
class ComposeResult:
    weights: list[float]
    neighbor_object_ids: list[str]
    neighbor_token_indices: list[int]
    support_s: float
    n_eff: float
    value_by_horizon: dict[int, list[float] | None]
    category_posterior: dict[str, float]


class _PositionCache:
    """Lazily decodes each object's position-trajectory blob at most once per compose call."""

    def __init__(self, objects_dir: Path, blobs_root: Path) -> None:
        self._objects_dir = objects_dir
        self._blobs_root = blobs_root
        self._cache: dict[str, np.ndarray] = {}

    def positions(self, object_id: str) -> np.ndarray:
        if object_id not in self._cache:
            import json
            record = json.loads((self._objects_dir / f"{object_id}.json").read_text())
            ref = record["payload"]["ext"]["hekbv3"]["trajectory"]
            blob_bytes = get_blob(self._blobs_root, ref["blob_sha256"])
            points, _header = trajectory_blob.decode_trajectory_blob(blob_bytes)
            self._cache[object_id] = np.asarray(points, dtype=np.float64)
        return self._cache[object_id]


def standardize_query(raw_feature: np.ndarray, mean: list[float], std: list[float], kept_dims: list[int]) -> np.ndarray:
    mean_a, std_a = np.asarray(mean), np.asarray(std)
    return (raw_feature[kept_dims] - mean_a[kept_dims]) / std_a[kept_dims]


def compose(
    *,
    query_standardized: np.ndarray,
    query_object_id: str,
    query_token_index: int,
    keys: np.ndarray,
    entries: list[dict],
    objects_dir: Path,
    blobs_root: Path,
    temperature: float,
    top_m: int = 64,
    exclude_groups: set[str] | None = None,
    exclude_object_ids: set[str] | None = None,
    laplace_alpha: float | None = None,
) -> ComposeResult:
    exclude_groups = exclude_groups or set()
    exclude_object_ids = exclude_object_ids or set()

    keep_mask = np.array([
        (e["group"] not in exclude_groups) and (e["object_id"] not in exclude_object_ids)
        for e in entries
    ])
    if not keep_mask.any():
        raise ComposeError("every library entry was excluded; nothing to compose against")

    kept_keys = keys[keep_mask]
    kept_entries = [e for e, m in zip(entries, keep_mask) if m]

    sq_dist = np.sum((kept_keys - query_standardized[None, :]) ** 2, axis=1)
    logits = -sq_dist / temperature
    support_s = float(np.logaddexp.reduce(logits)) if logits.size else float("-inf")

    if top_m < logits.size:
        top_idx = np.argpartition(logits, -top_m)[-top_m:]
    else:
        top_idx = np.arange(logits.size)
    truncated_logits = logits[top_idx]
    truncated_logits = truncated_logits - truncated_logits.max()
    exp_logits = np.exp(truncated_logits)
    weights_top = exp_logits / exp_logits.sum()

    n_eff = float(1.0 / np.sum(weights_top ** 2))

    chosen_entries = [kept_entries[i] for i in top_idx]
    weights_list = weights_top.tolist()

    cache = _PositionCache(objects_dir, blobs_root)
    query_positions = cache.positions(query_object_id)
    x_t = query_positions[query_token_index] / np.linalg.norm(query_positions[query_token_index])

    value_by_horizon: dict[int, list[float] | None] = {}
    for h in HORIZONS:
        v_sum = np.zeros_like(x_t)
        w_sum = 0.0
        for w, e in zip(weights_list, chosen_entries):
            positions = cache.positions(e["object_id"])
            t = e["token_index"]
            if t + h >= positions.shape[0]:
                continue
            xi = positions[t] / np.linalg.norm(positions[t])
            yi = positions[t + h] / np.linalg.norm(positions[t + h])
            v_sum = v_sum + w * log_map(xi, yi)
            w_sum += w
        if w_sum < 1e-9:
            value_by_horizon[h] = None
        else:
            v_hat = v_sum / w_sum
            # K2 ("ブレンドの位置が単位ベクトル、1e-9以内") holds for exp_map(x_t, v) only
            # when v is exactly tangent to x_t (v . x_t == 0): cos^2+sin^2=1 is then exact
            # regardless of |v|. Each v_{h,i} above is tangent to its OWN base point x_i,
            # not to the query's x_t (no parallel transport, §4.3 as literally written --
            # see module docstring); their weighted average is therefore not guaranteed
            # tangent to x_t. Projecting onto tangent(x_t) before exp_map is the minimal
            # fix that restores exact unit-norm output without changing the formula's
            # intent (same local-blend approximation, just evaluated in the correct
            # tangent plane at the point exp_map actually walks from).
            v_tangent = v_hat - np.dot(x_t, v_hat) * x_t
            x_pred = exp_map(x_t, v_tangent)
            value_by_horizon[h] = x_pred.tolist()

    # R3 (HEKBV3-BUILD001 差分, §4.5): Laplace smoothing is applied over every category
    # present in the (post-exclude) LIBRARY -- K distinct categories -- not only the ones
    # that happen to appear in top_m; a category entirely absent from top_m still gets the
    # same alpha, rather than falling through to an external epsilon floor (the previous
    # build's behaviour, now corrected). alpha defaults to 1/K, overridable by the caller.
    all_library_categories = sorted(set(e["category"] for e in kept_entries))
    k_cat = max(len(all_library_categories), 1)
    alpha = laplace_alpha if laplace_alpha is not None else 1.0 / k_cat

    category_counts: dict[str, float] = {c: 0.0 for c in all_library_categories}
    for w, e in zip(weights_list, chosen_entries):
        category_counts[e["category"]] = category_counts.get(e["category"], 0.0) + w
    total_w = sum(weights_list)  # == 1.0 by softmax construction, kept explicit for clarity
    posterior = {
        c: (category_counts[c] + alpha) / (total_w + k_cat * alpha)
        for c in all_library_categories
    }

    return ComposeResult(
        weights=weights_list,
        neighbor_object_ids=[e["object_id"] for e in chosen_entries],
        neighbor_token_indices=[e["token_index"] for e in chosen_entries],
        support_s=support_s,
        n_eff=n_eff,
        value_by_horizon=value_by_horizon,
        category_posterior=posterior,
    )
