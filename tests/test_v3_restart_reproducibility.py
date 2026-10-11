"""HEKBV3-BUILD001 K8 ("別プロセス・再起動後に object_id と index_sha256 が再現") and the
reuse loop's own "サービスを再起動→同じ compose が同じ結果を返す": a REAL subprocess is
started, used, killed, and restarted against the same (throwaway) on-disk directories --
not an in-process TestClient, since module-level state (e.g. the bearer token) would
trivially "survive" a reload in a way a real process restart would not.

Entirely under tmp_path; never touches the real 8300 service or /media/psf/HEKBv3.
"""
from __future__ import annotations

import base64
import hashlib
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHOR_10 = ("P01", "P03", "P09", "P13", "P18", "P19", "P25", "P28", "P31", "P35")
ALL_38 = tuple(f"P{str(i).zfill(2)}" for i in range(1, 39))
SPACE = {
    "model_ref": "toy-model", "layer": 1, "projection_sha256": "p" * 64, "w_sha256": "w" * 64,
    "anchor_policy_sha256": "a" * 64, "board_format": "36cell/1", "quant_step_log2": -30,
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, token: str, timeout_s: float = 10.0, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    last_exc = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"service process exited early (code={proc.returncode}):\n{out}")
        try:
            r = requests.get(f"{base_url}/health", headers={"Authorization": f"Bearer {token}"}, timeout=1.0)
            if r.status_code == 200:
                return
        except requests.RequestException as exc:
            last_exc = exc
        time.sleep(0.1)
    raise RuntimeError(f"service did not become healthy in time: {last_exc}")


def _start_service(data_dir: Path, app_support_dir: Path, index_dir: Path, pubkey_path: Path, port: int) -> subprocess.Popen:
    app_support_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "HEKB_DATA_DIR": str(data_dir), "HEKB_APP_SUPPORT_DIR": str(app_support_dir),
        "HEKB_INDEX_DIR": str(index_dir), "HEKB_AUDIT_PUBLIC_KEY_PATH": str(pubkey_path),
        "PATH": "/usr/bin:/bin",
    }
    proc = subprocess.Popen(
        [sys.executable, "store_service.py",
         "--host", "127.0.0.1", "--port", str(port), "--no-uds"],
        cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return proc


def _put_blob(base_url: str, token: str, points: list[list[float]], dim: int) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from storage.trajectory_blob import encode_trajectory_blob
    # created_at_ms=0 (fixed, not time.time()): this test re-encodes the identical logical
    # trajectory on both runs to check object_id/index_sha256 reproducibility (K8) -- the
    # default wall-clock timestamp would otherwise make blob_sha256 (and so object_id)
    # legitimately differ between the two runs even though the underlying data matches.
    file_bytes, _header = encode_trajectory_blob(points, dimensions=dim, created_at_ms=0)
    sha = hashlib.sha256(file_bytes).hexdigest()
    r = requests.put(f"{base_url}/v3/blobs/{sha}", data=file_bytes,
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
    assert r.status_code == 201, r.text
    return sha


def _make_ports_rows(rng: np.random.Generator, n_steps: int) -> list[list[float]]:
    rows = []
    for t in range(n_steps):
        row = rng.normal(size=38).tolist()
        if t == 0:
            for i, p in enumerate(ALL_38):
                if p not in ("P02", "P04"):
                    row[i] = float("nan")
        rows.append(row)
    return rows


def _make_causal_log_blob(base_url: str, token: str, rng: np.random.Generator, n_steps: int) -> tuple[str, int]:
    """feature/require-causal-log: see test_v3_trajectory.py's own helper of the same
    name for why this is independent of ports_rows's width. created_at carries no
    wall-clock component here either (encode_causal_log_blob has no timestamp field at
    all), so this stays reproducible across the test's two service runs (K8)."""
    sys.path.insert(0, str(REPO_ROOT))
    from storage.causal_log_blob import encode_causal_log_blob

    records = [
        {
            "token_index": t, "seq": t, "trajectory_hash": "t" * 64,
            "ports": dict(zip(ALL_38, rng.normal(size=38).tolist(), strict=True)),
            "cycle_raw": {"step": t},
            "source_file": {"path": "synthetic", "sha256": "s" * 64}, "line_number": t + 1,
        }
        for t in range(n_steps)
    ]
    data, sha256 = encode_causal_log_blob(records)
    r = requests.put(f"{base_url}/v3/blobs/{sha256}", data=data,
                      headers={"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
    assert r.status_code == 201, r.text
    return sha256, n_steps


def _post_trajectory(base_url: str, token: str, sign, *, seed: int, category: str, group: str, space_id: str) -> dict:
    rng = np.random.default_rng(seed)
    n_steps = 20
    positions = [list(v / np.linalg.norm(v)) for v in rng.normal(size=(n_steps, 64))]
    ports_rows = _make_ports_rows(rng, n_steps)
    pos_sha = _put_blob(base_url, token, positions, dim=64)
    ports_sha = _put_blob(base_url, token, ports_rows, dim=38)
    causal_sha, causal_n_steps = _make_causal_log_blob(base_url, token, rng, n_steps)
    payload = {
        "schema": "hekb.experience/2", "raw_input": f"trajectory seed={seed}", "resolved_meaning": None,
        "action": None, "title": f"traj-{seed}", "tags": ["trj"],
        "ext": {"hekbv3": {
            "kind": "trajectory", "origin": "experience",
            "space": {**SPACE, "space_id": space_id},
            "trajectory": {"blob_sha256": pos_sha, "n_steps": n_steps, "dim": 64},
            "ports": {"blob_sha256": ports_sha, "n_steps": n_steps, "dim": 38, "port_ids": list(ALL_38)},
            "causal_log": {"blob_sha256": causal_sha, "n_steps": causal_n_steps},
            "labels": {"category": category, "split": "dev", "group": group},
            "outcome": {"result": "n/a"},
        }},
    }
    sig = sign(payload)
    r = requests.post(f"{base_url}/v3/trajectories", json={"payload": payload},
                       headers={"Authorization": f"Bearer {token}", "X-Audit-Signature": sig})
    assert r.status_code == 201, r.text
    return r.json()


def test_restart_reproduces_object_id_index_sha256_and_compose(tmp_path):
    data_dir = tmp_path / "experience"
    app_support_dir = tmp_path / "app_support"
    index_dir = tmp_path / "index_local"
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()

    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    raw_point = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")
    pubkey_path = keys_dir / "dev_public_key.raw"
    pubkey_path.write_bytes(raw_point)

    def sign(payload: dict) -> str:
        sys.path.insert(0, str(REPO_ROOT / "vendor" / "hekb_vnext"))
        from core.object import canonical_bytes
        message = canonical_bytes(payload)
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(signature).decode("ascii")

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    proc = _start_service(data_dir, app_support_dir, index_dir, pubkey_path, port)
    try:
        token_path = app_support_dir / "hekb.token"
        deadline = time.time() + 10
        while not token_path.exists() and time.time() < deadline:
            time.sleep(0.1)
        token = token_path.read_text().strip()
        _wait_for_health(base_url, token)

        space_id = requests.post(f"{base_url}/v3/space-id", json=SPACE,
                                  headers={"Authorization": f"Bearer {token}"}).json()["space_id"]

        first_run_objects = []
        for i in range(5):
            out = _post_trajectory(base_url, token, sign, seed=100 + i, category=f"cat{i % 2}", group=f"g{i}", space_id=space_id)
            first_run_objects.append(out["object_id"])

        build1 = requests.post(f"{base_url}/v3/index/{space_id}/build", json={"L": 8},
                                headers={"Authorization": f"Bearer {token}"}).json()
        compose1 = requests.post(f"{base_url}/v3/compose", json={
            "space_id": space_id, "query_object_id": first_run_objects[0], "query_token_index": 10,
            "exclude": {"groups": ["g0"]}, "temperature": 1.0,
        }, headers={"Authorization": f"Bearer {token}"}).json()
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # restart: same directories, new process, new port
    port2 = _free_port()
    base_url2 = f"http://127.0.0.1:{port2}"
    proc2 = _start_service(data_dir, app_support_dir, index_dir, pubkey_path, port2)
    try:
        token_path = app_support_dir / "hekb.token"
        # the token file already exists (proc1's), so wait for its *content* to change to
        # proc2's freshly-generated value rather than just `.exists()` (which would pass
        # instantly on the stale file and read proc1's now-invalid token).
        deadline = time.time() + 10
        token2 = token_path.read_text().strip()
        while token2 == token and time.time() < deadline:
            time.sleep(0.1)
            token2 = token_path.read_text().strip()
        _wait_for_health(base_url2, token2, proc=proc2)

        # re-post the identical payloads: idempotent, must resolve to the same object_ids
        second_run_objects = []
        for i in range(5):
            out = _post_trajectory(base_url2, token2, sign, seed=100 + i, category=f"cat{i % 2}", group=f"g{i}", space_id=space_id)
            second_run_objects.append(out["object_id"])
        assert second_run_objects == first_run_objects, "object_id must reproduce across a restart (K8)"

        build2 = requests.post(f"{base_url2}/v3/index/{space_id}/build", json={"L": 8},
                                headers={"Authorization": f"Bearer {token2}"}).json()
        assert build2["index_sha256"] == build1["index_sha256"], "index_sha256 must reproduce across a restart (K8)"

        compose2 = requests.post(f"{base_url2}/v3/compose", json={
            "space_id": space_id, "query_object_id": first_run_objects[0], "query_token_index": 10,
            "exclude": {"groups": ["g0"]}, "temperature": 1.0,
        }, headers={"Authorization": f"Bearer {token2}"}).json()
        assert compose2["weights"] == compose1["weights"], "compose() must reproduce across a restart (reuse loop)"
        assert compose2["category_posterior"] == compose1["category_posterior"]
    finally:
        proc2.terminate()
        proc2.wait(timeout=10)
