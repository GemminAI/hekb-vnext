"""HEKBV3-BUILD001 E2E: ingest TRJ003's 48 sessions into the EXPERIMENTAL storage area
(/media/psf/HEKBv3/exp_hekbv3/, 改修設計書 v0.1 §10-3 -- never the production
/media/psf/HEKBv3/experience/), build the experience index, run leave-one-group-out
(item-level: prompt_id) compose 48 times, and report category log-loss vs the (also
leave-one-group-out) empirical prior. Numbers only -- no interpretation (per the task).

Run against a SEPARATE service instance on a throwaway port pointed at exp_hekbv3 (the
always-on production 8300 service, pointed at .../experience, is never touched)."""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import requests
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "vendor" / "hekb_vnext"))
from core.object import canonical_bytes  # noqa: E402
from storage.trajectory_blob import encode_trajectory_blob  # noqa: E402

TRJ_ROOT = Path("/media/psf/HEKBv3/trj003_48")
MANIFEST = TRJ_ROOT / "manifest.json"
SESSIONS_DIR = TRJ_ROOT / "results_vm" / "sessions"
INDEX_JSONL = TRJ_ROOT / "results_vm" / "INDEX.jsonl"

ALL_38 = tuple(f"P{str(i).zfill(2)}" for i in range(1, 39))
EPS_PRIOR = 1e-9


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text())
    missing = [s["session_id"] for s in manifest["sessions"] if "category" not in s or "prompt_id" not in s]
    if missing:
        raise SystemExit(f"STOP: manifest is missing category/prompt_id for: {missing}")
    return manifest


def index_jsonl_by_session_id() -> dict[str, dict]:
    out = {}
    for line in INDEX_JSONL.read_text().splitlines():
        row = json.loads(line)
        session_id = row["file"].split("_", 2)[-1].removesuffix(".jsonl.gz")
        out[session_id] = row
    return out


def read_session(file_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (positions (n_steps,64), ports (n_steps,38) with NaN where status != OK)."""
    path = SESSIONS_DIR / file_name
    positions, ports = [], []
    with gzip.open(path, "rt") as f:
        for line in f:
            rec = json.loads(line)
            positions.append(rec["position_64d"])
            row = [np.nan] * 38
            for i, p in enumerate(ALL_38):
                entry = rec["ports"].get(p)
                if entry is not None and entry.get("status") == "OK" and entry.get("scalar") is not None:
                    row[i] = float(entry["scalar"])
            ports.append(row)
    return np.asarray(positions, dtype=np.float64), np.asarray(ports, dtype=np.float32)


def compute_space(manifest: dict) -> dict:
    w_sha256 = _sha256_file(TRJ_ROOT / "runner" / "W_seed55_64x256_raw.npy")
    return {
        "model_ref": manifest["sessions"][0].get("source", "TRJ003") + ":Llama-3.1-8B-Instruct(MSD0001)",
        "layer": manifest["layer_number"],
        "projection_sha256": manifest["projection_P"]["sha256"],
        "w_sha256": w_sha256,
        # [HEKBV3-BUILD001, disclosed]: this TRJ003 run used NVS-Kernel v6.1.3 (health_before.json),
        # which predates v7's anchor_policy_sha256 scheme (docs/v7/ARCHITECTURE.md); every
        # session's own per-run summary (INDEX.jsonl) shows "anchor_info": null, confirming no
        # anchor policy was in effect -- "none" is used as the fixed sentinel for this space.
        "anchor_policy_sha256": "none",
        # same disclosure for the v7 rules: v6.1.3 predates both (fixed literals, fixed dt)
        "wiring_policy_sha256": "legacy-fixed-v1",
        "time_policy_sha256": "fixed-dt-v1",
        "board_format": "36cell/1",
        "quant_step_log2": -30,
    }


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8301"
    token = Path(sys.argv[2]).read_text().strip() if len(sys.argv) > 2 else None
    if token is None:
        raise SystemExit("usage: e2e_trj003.py <base_url> <token_path>")
    headers_auth = {"Authorization": f"Bearer {token}"}

    private_key_path = sys.argv[3] if len(sys.argv) > 3 else str(Path.home() / ".hekb-v3-keys" / "dev_private_key.pem")
    from cryptography.hazmat.primitives import serialization
    private_key = serialization.load_pem_private_key(Path(private_key_path).read_bytes(), password=None)

    def sign(payload: dict) -> str:
        message = canonical_bytes(payload)
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(signature).decode("ascii")

    manifest = load_manifest()
    index_rows = index_jsonl_by_session_id()
    space = compute_space(manifest)
    space_id = requests.post(f"{base_url}/v3/space-id", json=space, headers=headers_auth).json()["space_id"]
    print(f"space_id={space_id}")

    print(f"about to POST 48 sessions into exp_hekbv3 (categories: "
          f"{sorted(set(s['category'] for s in manifest['sessions']))}, "
          f"groups(prompt_id): {len(set(s['prompt_id'] for s in manifest['sessions']))} distinct)")

    object_ids, categories, groups, n_steps_list = [], [], [], []
    for s in manifest["sessions"]:
        row = index_rows[s["session_id"]]
        file_name = Path(row["file"]).name
        positions, ports = read_session(file_name)

        pos_bytes, _ = encode_trajectory_blob(positions.tolist(), dimensions=64, created_at_ms=0)
        pos_sha = hashlib.sha256(pos_bytes).hexdigest()
        r = requests.put(f"{base_url}/v3/blobs/{pos_sha}", data=pos_bytes,
                          headers={**headers_auth, "Content-Type": "application/octet-stream"})
        assert r.status_code == 201, r.text

        ports_bytes, _ = encode_trajectory_blob(ports.tolist(), dimensions=38, created_at_ms=0)
        ports_sha = hashlib.sha256(ports_bytes).hexdigest()
        r = requests.put(f"{base_url}/v3/blobs/{ports_sha}", data=ports_bytes,
                          headers={**headers_auth, "Content-Type": "application/octet-stream"})
        assert r.status_code == 201, r.text

        payload = {
            "schema": "hekb.experience/2", "raw_input": s["session_id"], "resolved_meaning": None,
            "action": None, "title": s["session_id"], "tags": ["TRJ003", s["category"]],
            "ext": {"hekbv3": {
                "kind": "trajectory", "origin": "experience",
                "space": {**space, "space_id": space_id},
                "trajectory": {"blob_sha256": pos_sha, "n_steps": int(positions.shape[0]), "dim": 64},
                "ports": {"blob_sha256": ports_sha, "n_steps": int(ports.shape[0]), "dim": 38, "port_ids": list(ALL_38)},
                # feature/require-causal-log: this TRJ003 run predates the causal_log
                # feature entirely (v6.1.3, see compute_space's own note above) -- there
                # is no raw per-token log to build one from, explicitly, not a bypass.
                "causal_log_omitted_reason": "pre-causal_log TRJ003 run (NVS-Kernel v6.1.3)",
                "labels": {"category": s["category"], "split": s["split"], "group": s["prompt_id"]},
                "outcome": {"result": s["label"], "intended_label": s["intended_label"]},
            }},
        }
        sig = sign(payload)
        r = requests.post(f"{base_url}/v3/trajectories", json={"payload": payload},
                           headers={**headers_auth, "X-Audit-Signature": sig})
        assert r.status_code == 201, r.text
        out = r.json()
        object_ids.append(out["object_id"])
        categories.append(s["category"])
        groups.append(s["prompt_id"])
        n_steps_list.append(int(positions.shape[0]))
        print(f"  posted {s['session_id']}: object_id={out['object_id'][:12]}... n_steps={positions.shape[0]}")

    build = requests.post(f"{base_url}/v3/index/{space_id}/build", json={"L": 8}, headers=headers_auth).json()
    print(f"index built: n_entries={build['n_entries']} d_kept={build['d_kept']} index_sha256={build['index_sha256']}")

    n = len(object_ids)
    all_categories = sorted(set(categories))
    kernel_logloss, prior_logloss, used_n_eff, used_top_m, used_temp = [], [], [], [], []
    TOP_M = 64
    for i in range(n):
        own_group = groups[i]
        true_cat = categories[i]
        query_token_index = n_steps_list[i] - 1  # the last valid window position (single
        # representative query per session -- decision, 1 line: no per-token aggregation
        # in this smoke E2E, see scripts/e2e_trj003.py module docstring / report).

        resp = requests.post(f"{base_url}/v3/compose", json={
            "space_id": space_id, "query_object_id": object_ids[i], "query_token_index": query_token_index,
            "top_m": TOP_M, "exclude": {"groups": [own_group]},
        }, headers=headers_auth)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        posterior = body["category_posterior"]
        kernel_logloss.append(-math.log(max(posterior.get(true_cat, 0.0), EPS_PRIOR)))
        used_n_eff.append(body["n_eff"])
        used_top_m.append(TOP_M)
        used_temp.append(body["temperature_used"])

        # leave-one-group-out empirical prior: category frequency among the same
        # (post-exclude) library the kernel saw, i.e. everyone except this session's group.
        other_cats = [c for c, g in zip(categories, groups) if g != own_group]
        freq = {c: other_cats.count(c) / len(other_cats) for c in all_categories}
        prior_logloss.append(-math.log(max(freq.get(true_cat, 0.0), EPS_PRIOR)))

    print(f"n_queries={n}")
    print(f"features=10-port L=8 window (d_kept={build['d_kept']}/{10*8}), top_m={TOP_M}, "
          f"temperature=per-query kNN-bandwidth default (median of 16th-NN sq-dist over the "
          f"post-exclude library); mean T_used={sum(used_temp)/n:.6g}, mean n_eff={sum(used_n_eff)/n:.3g}")
    print(f"kernel category log-loss (mean over 48, leave-one-group-out): {sum(kernel_logloss)/n:.6f}")
    print(f"prior category log-loss (leave-one-group-out empirical frequency, mean over 48): {sum(prior_logloss)/n:.6f}")
    print(f"uniform-prior log-loss reference: {math.log(len(all_categories)):.6f} (12 categories)")

    out_path = TRJ_ROOT.parent / "exp_hekbv3_e2e_result.json"
    out_path.write_text(json.dumps({
        "space_id": space_id, "n_queries": n, "top_m": TOP_M,
        "mean_temperature_used": sum(used_temp) / n, "mean_n_eff": sum(used_n_eff) / n,
        "kernel_logloss_mean": sum(kernel_logloss) / n, "prior_logloss_mean": sum(prior_logloss) / n,
        "uniform_logloss_reference": math.log(len(all_categories)),
        "per_query": [{"object_id": object_ids[i], "category": categories[i], "group": groups[i],
                       "kernel_logloss": kernel_logloss[i], "prior_logloss": prior_logloss[i],
                       "n_eff": used_n_eff[i], "temperature_used": used_temp[i]} for i in range(n)],
    }, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
