"""Layer 5 (index): `index/<space_id>/experience/` -- rebuildable from objects + blobs only
(HEKBv3 改修設計書 v0.1 §3.3, §9 K-criteria). Experience channel only (追補A §3); the
knowledge channel does not exist yet (追補A §6 決定⑤).

Feature = the anchor-independent 10-port window (追補A §4: P01, P03, P09, P13, P18, P19,
P25, P28, P31, P35), the L most recent tokens flattened (d = L*10, default L=8 -> d=80),
standardized library-wide, zero-std dimensions dropped. The window for query position t
uses tokens [t-L+1, t] (inclusive) -- this is >=1 for all t>=L, so token 0 (9 of 38 ports
ERROR at the very first observation) is never in any window without a separate carve-out
(matches 追補A §4.1 "窓の最初は token 8 以降" exactly when L=8, t starts at 8).

[HEKBV3-BUILD001, disclosed schema extension]: §3.1's worked example has no field for the
*raw* per-token port scalars (only a precomputed "features" blob reference, spec_id
"w8p17z/1"). Without the raw per-token ports, the index could not be rebuilt from objects+
blobs alone if the port subset or window length ever changes (violating the Layer 5 "作り
直せる" guarantee at the PREREG's own standard). So `POST /v3/trajectories` additionally
writes a `ports` blob (all 38 ports, NaN where a port's status wasn't OK, dim=38, same
generalized `storage.trajectory_blob` format as the 64-D position) and records its
reference + `port_ids` order under `ext.hekbv3.ports`, alongside the existing `trajectory`
key for the 64-D position series.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from storage import trajectory_blob
from storage.blobs import get_blob

ANCHOR_INDEPENDENT_10 = ("P01", "P03", "P09", "P13", "P18", "P19", "P25", "P28", "P31", "P35")
ALL_38_PORTS = tuple(f"P{str(i).zfill(2)}" for i in range(1, 39))
DEFAULT_L = 8


class IndexError_(ValueError):
    pass


@dataclass
class IndexEntry:
    object_id: str
    token_index: int
    group: str
    category: str
    label: str | None


def _iter_experience_trajectories(objects_dir: Path, space_id: str):
    """Yields (object_id, payload_dict) for every stored object whose
    ext.hekbv3.kind == 'trajectory', origin == 'experience', and space.space_id matches."""
    for path in sorted(objects_dir.glob("*.json")):
        record = json.loads(path.read_text())
        payload = record.get("payload", {})
        ext = payload.get("ext", {}).get("hekbv3")
        if ext is None:
            continue
        if ext.get("kind") != "trajectory" or ext.get("origin") != "experience":
            continue
        if ext.get("space", {}).get("space_id") != space_id:
            continue
        yield record["object_id"], payload


def _window_features(ports_series: np.ndarray, port_ids: list[str], l: int) -> tuple[np.ndarray, np.ndarray]:
    """ports_series: (n_steps, 38) float32, NaN where not OK. Returns (features, token_indices)
    where features is (n_windows, l * len(ANCHOR_INDEPENDENT_10)), one row per valid t."""
    col_idx = [port_ids.index(p) for p in ANCHOR_INDEPENDENT_10]
    selected = ports_series[:, col_idx]  # (n_steps, 10)
    n_steps = selected.shape[0]
    rows, tokens = [], []
    for t in range(l - 1, n_steps):
        window = selected[t - l + 1: t + 1]  # (l, 10), inclusive of t
        rows.append(window.reshape(-1))
        tokens.append(t)
    if not rows:
        return np.zeros((0, l * len(ANCHOR_INDEPENDENT_10)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(rows, axis=0), np.array(tokens, dtype=np.int64)


def build_index(
    objects_dir: Path, blobs_root: Path, index_root: Path, space_id: str, l: int = DEFAULT_L,
) -> dict:
    """Builds (overwriting) index/<space_id>/experience/ from objects_dir + blobs_root.
    Returns the meta dict (including index_sha256)."""
    all_rows: list[np.ndarray] = []
    entries: list[IndexEntry] = []

    for object_id, payload in _iter_experience_trajectories(objects_dir, space_id):
        ext = payload["ext"]["hekbv3"]
        ports_ref = ext.get("ports")
        if ports_ref is None:
            raise IndexError_(f"object {object_id}: no ext.hekbv3.ports blob reference, cannot build features")
        port_ids = ports_ref["port_ids"]
        blob_bytes = get_blob(blobs_root, ports_ref["blob_sha256"])
        points, header = trajectory_blob.decode_trajectory_blob(blob_bytes)
        ports_series = np.asarray(points, dtype=np.float32)
        if ports_series.shape[1] != len(port_ids):
            raise IndexError_(f"object {object_id}: ports blob dim {ports_series.shape[1]} != len(port_ids) {len(port_ids)}")

        features, tokens = _window_features(ports_series, port_ids, l)
        labels = ext.get("labels", {})
        group = labels.get("group", object_id)
        category = labels.get("category", "")
        label = ext.get("outcome", {}).get("result") if isinstance(ext.get("outcome"), dict) else None
        for i in range(features.shape[0]):
            all_rows.append(features[i])
            entries.append(IndexEntry(object_id=object_id, token_index=int(tokens[i]), group=group,
                                       category=category, label=label))

    if not all_rows:
        raise IndexError_(f"space_id={space_id}: no experience trajectories found to index")

    raw = np.stack(all_rows, axis=0)  # (n_entries, l*10)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    kept_dims = np.where(std > 0)[0]
    if kept_dims.size == 0:
        raise IndexError_(f"space_id={space_id}: every feature dimension has zero std")
    standardized = (raw[:, kept_dims] - mean[kept_dims]) / std[kept_dims]

    exp_dir = index_root / space_id / "experience"
    exp_dir.mkdir(parents=True, exist_ok=True)
    keys_path = exp_dir / "keys.f32"
    ids_path = exp_dir / "ids.jsonl"
    meta_path = exp_dir / "meta.json"

    keys_bytes = standardized.astype(np.float32).tobytes()
    keys_path.write_bytes(keys_bytes)
    ids_lines = [json.dumps({
        "object_id": e.object_id, "token_index": e.token_index, "group": e.group,
        "category": e.category, "label": e.label,
    }, sort_keys=True) for e in entries]
    ids_path.write_text("\n".join(ids_lines) + "\n")

    import hashlib
    hasher = hashlib.sha256()
    hasher.update(keys_bytes)
    hasher.update("\n".join(ids_lines).encode("utf-8"))
    index_sha256 = hasher.hexdigest()

    meta = {
        "space_id": space_id, "l": l, "port_ids_used": list(ANCHOR_INDEPENDENT_10),
        "n_entries": standardized.shape[0], "d_raw": raw.shape[1], "d_kept": int(kept_dims.size),
        "kept_dims": kept_dims.tolist(), "mean": mean.tolist(), "std": std.tolist(),
        "index_sha256": index_sha256, "built_at_ms": int(time.time() * 1000),
    }
    meta_path.write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def load_index(index_root: Path, space_id: str) -> tuple[np.ndarray, list[dict], dict]:
    exp_dir = index_root / space_id / "experience"
    meta = json.loads((exp_dir / "meta.json").read_text())
    n, d = meta["n_entries"], meta["d_kept"]
    keys = np.frombuffer((exp_dir / "keys.f32").read_bytes(), dtype=np.float32).reshape(n, d)
    ids = [json.loads(line) for line in (exp_dir / "ids.jsonl").read_text().splitlines() if line]
    return keys, ids, meta
