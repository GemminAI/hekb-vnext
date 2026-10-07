"""HEKBV3-BUILD001: K1, K2 against `index.compose.compose()` directly (no HTTP layer --
these are pure-math acceptance criteria, HEKBv3 改修設計書 v0.1 §9). Each test builds its
own tiny on-disk objects/+blobs/ fixture (not going through POST /v3/trajectories) so they
stay fast and decoupled from the FastAPI route plumbing covered in test_v3_trajectory.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from index import compose as compose_mod
from storage import trajectory_blob
from storage.blobs import get_blob, put_blob


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _write_position_object(objects_dir: Path, blobs_root: Path, object_id: str, positions: np.ndarray) -> None:
    file_bytes, header = trajectory_blob.encode_trajectory_blob(positions.tolist(), dimensions=positions.shape[1])
    import hashlib
    blob_sha256 = hashlib.sha256(file_bytes).hexdigest()
    put_blob(blobs_root, blob_sha256, file_bytes)
    objects_dir.mkdir(parents=True, exist_ok=True)
    record = {"object_id": object_id, "payload": {"ext": {"hekbv3": {"trajectory": {"blob_sha256": blob_sha256}}}}}
    (objects_dir / f"{object_id}.json").write_text(json.dumps(record))


def _fixture(tmp_path: Path, n_entries: int = 6, dim64: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    objects_dir = tmp_path / "objects"
    blobs_root = tmp_path / "blobs"
    entries = []
    keys_rows = []
    query_key = rng.normal(size=8)  # small synthetic "standardized feature"
    for i in range(n_entries):
        object_id = f"obj{i}"
        # a short random-walk on the sphere, long enough for h in {1,4,8}
        n_steps = 12
        raw = rng.normal(size=(n_steps, dim64))
        positions = np.array([_unit(r) for r in raw])
        _write_position_object(objects_dir, blobs_root, object_id, positions)
        entries.append({"object_id": object_id, "token_index": 3, "group": f"g{i}", "category": "cat" + str(i % 2), "label": None})
        # key close to query for small i, far for large i (so T->0 nearest-neighbour is entry 0)
        keys_rows.append(query_key + rng.normal(scale=0.05 + i * 2.0, size=8))
    keys = np.stack(keys_rows, axis=0)

    query_object_id = "query_obj"
    query_positions = np.array([_unit(rng.normal(size=dim64)) for _ in range(12)])
    _write_position_object(objects_dir, blobs_root, query_object_id, query_positions)

    return dict(objects_dir=objects_dir, blobs_root=blobs_root, keys=keys, entries=entries,
                query_standardized=query_key, query_object_id=query_object_id, query_token_index=3)


def test_k1_weights_nonnegative_sum_to_one_and_deterministic(tmp_path):
    fx = _fixture(tmp_path)
    result = compose_mod.compose(
        query_standardized=fx["query_standardized"], query_object_id=fx["query_object_id"],
        query_token_index=fx["query_token_index"], keys=fx["keys"], entries=fx["entries"],
        objects_dir=fx["objects_dir"], blobs_root=fx["blobs_root"], temperature=1.0, top_m=64,
    )
    w = np.array(result.weights)
    assert np.all(w >= 0.0)
    assert abs(w.sum() - 1.0) < 1e-12

    result2 = compose_mod.compose(
        query_standardized=fx["query_standardized"], query_object_id=fx["query_object_id"],
        query_token_index=fx["query_token_index"], keys=fx["keys"], entries=fx["entries"],
        objects_dir=fx["objects_dir"], blobs_root=fx["blobs_root"], temperature=1.0, top_m=64,
    )
    assert result.weights == result2.weights
    assert result.support_s == result2.support_s
    assert result.n_eff == result2.n_eff


def test_k2_blend_is_unit_vector(tmp_path):
    fx = _fixture(tmp_path)
    result = compose_mod.compose(
        query_standardized=fx["query_standardized"], query_object_id=fx["query_object_id"],
        query_token_index=fx["query_token_index"], keys=fx["keys"], entries=fx["entries"],
        objects_dir=fx["objects_dir"], blobs_root=fx["blobs_root"], temperature=1.0, top_m=64,
    )
    for h, v in result.value_by_horizon.items():
        if v is None:
            continue
        norm = float(np.linalg.norm(v))
        assert abs(norm - 1.0) < 1e-9, f"h={h}: blended position norm {norm} not unit"


def test_k2_temperature_limits(tmp_path):
    fx = _fixture(tmp_path)

    # T -> 0: weight concentrates on the single nearest key (entry 0, by fixture construction)
    tiny = compose_mod.compose(
        query_standardized=fx["query_standardized"], query_object_id=fx["query_object_id"],
        query_token_index=fx["query_token_index"], keys=fx["keys"], entries=fx["entries"],
        objects_dir=fx["objects_dir"], blobs_root=fx["blobs_root"], temperature=1e-6, top_m=64,
    )
    w = np.array(tiny.weights)
    nearest_local_idx = int(np.argmax(w))
    assert w[nearest_local_idx] > 0.999
    assert tiny.neighbor_object_ids[nearest_local_idx] == "obj0"

    # Compare the h=1 blended position to the single nearest neighbour's own log/exp step --
    # at T->0 they must coincide (weight ~1 on that one entry).
    import json as _json

    def _positions_of(object_id: str) -> list[list[float]]:
        ref = _json.loads((fx["objects_dir"] / f"{object_id}.json").read_text())
        sha = ref["payload"]["ext"]["hekbv3"]["trajectory"]["blob_sha256"]
        points, _header = trajectory_blob.decode_trajectory_blob(get_blob(fx["blobs_root"], sha))
        return points

    obj0_positions = _positions_of("obj0")
    query_positions = _positions_of("query_obj")
    x_t = np.array(query_positions[3]) / np.linalg.norm(query_positions[3])
    xi = np.array(obj0_positions[3]) / np.linalg.norm(obj0_positions[3])
    yi = np.array(obj0_positions[4]) / np.linalg.norm(obj0_positions[4])
    v = compose_mod.log_map(xi, yi)
    v_tangent = v - np.dot(x_t, v) * x_t  # compose() projects onto tangent(x_t) before exp_map
    expected = compose_mod.exp_map(x_t, v_tangent)
    actual = np.array(tiny.value_by_horizon[1])
    assert np.linalg.norm(actual - expected) < 1e-6

    # T -> infinity: weights become uniform across all (top_m-capped) entries
    huge = compose_mod.compose(
        query_standardized=fx["query_standardized"], query_object_id=fx["query_object_id"],
        query_token_index=fx["query_token_index"], keys=fx["keys"], entries=fx["entries"],
        objects_dir=fx["objects_dir"], blobs_root=fx["blobs_root"], temperature=1e6, top_m=64,
    )
    w2 = np.array(huge.weights)
    assert np.allclose(w2, 1.0 / len(w2), atol=1e-3)
