"""HEKBV3-BUILD002 §2: ingests Mac-side BoardRecord shards (nvs-runtime replay executor
output, /media/psf/HEKBv3/exp_hekbv3_boardrecords/) into a hekb-vnext-v3 data directory via
the real /v3/blobs + /v3/trajectories HTTP validation path (blob content-hash check, audit
signature check, idempotent-by-content object_id) -- never a bypass of that path.

Input layout (per the Mac-side task's own §2/§3, plus `model_ref`/`layer` added by the
2026-10-08 follow-up instruction):
  <root>/manifest.json: {"sessions": [{"session_id","category","group","split","n_records",
    "file"}, ...], "board_format", "quant_step_log2", "format_version", "projection_sha256",
    "w_sha256", "anchor_policy_sha256", "model_ref", "layer", ...commits/sha256s (read but
    not re-verified here beyond the per-session n_records/anchor_policy_sha256 checks
    below)}.
  <root>/<file> (e.g. validation48/<session_id>.jsonl.gz or sessions/<session_id>.jsonl.gz):
    one JSON object per line, each BoardRecord.to_wire() (nvs_edge/record.py), in token
    order, over the 32 non-excluded ports (CORE_IDS).

[HEKBV3-BUILD002, 2026-10-08 follow-up, disclosed]: `model_ref`/`layer` are read FROM
manifest.json when present (no longer CLI flags). `model_ref` is REQUIRED to be present; if
missing, this script substitutes the literal "不明" and reports it (does not stop) -- per the
explicit instruction, since the identifier lives in the *source* data (MSD0001 meta.json /
CDM-DATA001 selection_manifest.json), not something this script can derive on its own.

[HEKBV3-BUILD002 続き, 2026-10-08, disclosed]: the real Mac-produced manifest.json files
(sessions/, validation48/) have NEITHER `layer` NOR `model_ref`. Per the explicit instruction
("manifest を変更せず...ingest_context.json を新設"), manifest.json is never edited to add
them; instead this script also looks for an optional sibling file,
<manifest's own directory>/ingest_context.json, and uses its "layer"/"model_ref" fields when
the manifest itself lacks them. `layer` is still hard-required overall (manifest OR context;
raises if neither has it) -- its value is well-established (32, layer_index 7: confirmed from
/media/psf/HEKBv3/trj003_48/manifest.json's own layer_number/layer_index fields). `model_ref`
falls back further to "不明"+report only if BOTH manifest and ingest_context.json lack it.
Confirmed on this VM (see /media/psf/HEKBv3/exp_hekbv3_boardrecords/ingest_context.json for
the committed values and provenance):
  model_id = "meta-llama/Llama-3.1-8B-Instruct", revision = "0e9e39f249a16976918f6564b8830bc894c89659"
(source: /media/psf/EXP-MSD0001 v1.1/sessions/*/meta.json "model" field, cross-checked across
3 sessions, identical; corroborated by
/media/psf/sensos-experiments/CDM-DATA001/selection_manifest.json "model" field).

[HEKBV3-BUILD002, disclosed design]: runs the real store_service ASGI app through
starlette's TestClient (same mechanism as tests/test_v3_trajectory.py), pointed at
--data-dir via HEKB_DATA_DIR, with an ephemeral EC keypair generated for audit-signature
verification -- not a separately bound TCP port. This is a stronger form of "disposable
process / separate port": no port is bound at all, so there is no possibility of
colliding with the real 8300 service or any 8301+ instance. The 8300 service is never
imported, started, or contacted by this script.

Stops (raises, no retry) on: a shard's actual record count != manifest's n_records, a
session with inconsistent anchor_policy_sha256 across its own records, a session's cells
not forming a single consistent port_ids order, or any non-2xx HTTP response. Never writes
to HEKBv3/experience or exp_hekbv3 (only ever acts on the caller-supplied --data-dir).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


class IngestError(RuntimeError):
    pass


MODEL_REF_UNKNOWN = "不明"


def _load_ingest_context(manifest_path: Path) -> dict:
    """Optional context file supplying layer/model_ref without editing manifest.json (see
    module docstring's 2026-10-08 続き note). The task places this one level above the
    manifest (.../exp_hekbv3_boardrecords/ingest_context.json, shared by both sessions/ and
    validation48/ manifests), so this checks the manifest's own directory first, then its
    parent. Returns {} when neither has the file."""
    for candidate in (manifest_path.parent / "ingest_context.json",
                      manifest_path.parent.parent / "ingest_context.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text())
    return {}


def _load_manifest(manifest_path: Path) -> tuple[dict, bool]:
    """Returns (manifest, model_ref_was_missing). `layer` is required overall (from manifest
    itself, or failing that from the sibling ingest_context.json -- see _load_ingest_context);
    raises if neither has it. `model_ref` falls back the same way, then to MODEL_REF_UNKNOWN
    (reported, not a stop) if still absent."""
    manifest = json.loads(manifest_path.read_text())
    required = ("sessions", "board_format", "quant_step_log2", "anchor_policy_sha256")
    missing = [k for k in required if k not in manifest]
    if missing:
        raise IngestError(f"manifest.json missing required field(s): {missing}")
    if not isinstance(manifest["sessions"], list) or not manifest["sessions"]:
        raise IngestError("manifest.json 'sessions' must be a non-empty list")
    for i, s in enumerate(manifest["sessions"]):
        missing_s = [k for k in ("session_id", "category", "group", "split", "n_records", "file") if k not in s]
        if missing_s:
            raise IngestError(f"manifest.json sessions[{i}] missing field(s): {missing_s}")

    context = _load_ingest_context(manifest_path)
    manifest = dict(manifest)
    if "layer" not in manifest:
        if "layer" not in context:
            raise IngestError(
                "manifest.json missing required field(s): ['layer'] "
                "(no ingest_context.json sibling file supplying it either)"
            )
        manifest["layer"] = context["layer"]

    model_ref_was_missing = not manifest.get("model_ref")
    if model_ref_was_missing:
        if context.get("model_ref"):
            manifest["model_ref"] = context["model_ref"]
            model_ref_was_missing = False
        else:
            manifest["model_ref"] = MODEL_REF_UNKNOWN
    return manifest, model_ref_was_missing


def _read_shard(shard_path: Path) -> list[dict]:
    if not shard_path.exists():
        raise IngestError(f"shard file not found: {shard_path}")
    with gzip.open(shard_path, "rt", encoding="utf-8") as fh:
        lines = [line for line in fh.read().splitlines() if line]
    return [json.loads(line) for line in lines]


def _session_ports_matrix(records: list[dict], session_id: str, expected_anchor_policy: str) -> tuple[list[list[int]], list[str]]:
    """Validates internal consistency of one session's BoardRecord lines and returns
    (rows: n_steps x n_ports int64 cell values, in token order; port_ids: the fixed cell
    order, read from the first record)."""
    if not records:
        raise IngestError(f"session {session_id}: shard has zero records")

    port_ids = [cell[0] for cell in records[0]["cells"]]
    rows: list[list[int]] = []
    prev_token_index = -1
    for rec in records:
        if rec.get("session_id") != session_id:
            raise IngestError(f"session {session_id}: shard contains a record with session_id={rec.get('session_id')!r}")
        if rec.get("anchor_policy_sha256") != expected_anchor_policy:
            raise IngestError(
                f"session {session_id}: record token_index={rec.get('token_index')} has "
                f"anchor_policy_sha256={rec.get('anchor_policy_sha256')!r}, expected {expected_anchor_policy!r}"
            )
        these_ports = [cell[0] for cell in rec["cells"]]
        if these_ports != port_ids:
            raise IngestError(f"session {session_id}: cell port order is not consistent across records")
        token_index = rec["token_index"]
        if token_index <= prev_token_index:
            raise IngestError(
                f"session {session_id}: token_index did not strictly increase "
                f"({prev_token_index} -> {token_index})"
            )
        prev_token_index = token_index
        rows.append([cell[2] for cell in rec["cells"]])
    return rows, port_ids


def _generate_dev_signer():
    """Same throwaway-EC-keypair pattern as tests/test_v3_trajectory.py's `client` fixture:
    a fresh key per ingest run, never the real audit key, never written to a shared path."""
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    private_key = ec.generate_private_key(ec.SECP256R1())
    numbers = private_key.public_key().public_numbers()
    raw_point = b"\x04" + numbers.x.to_bytes(32, "big") + numbers.y.to_bytes(32, "big")

    def sign(payload: dict, canonical_bytes_fn) -> str:
        message = canonical_bytes_fn(payload)
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        return base64.b64encode(signature).decode("ascii")

    return raw_point, sign


def run_ingest(
    *, manifest_path: Path, shards_root: Path, data_dir: Path,
    only_session_ids: set[str] | None = None, progress_every: int = 50,
) -> dict:
    import os

    data_dir.mkdir(parents=True, exist_ok=True)
    public_key_path = data_dir / ".ingest_dev_public_key.raw"
    raw_point, sign = _generate_dev_signer()
    public_key_path.write_bytes(raw_point)

    os.environ["HEKB_DATA_DIR"] = str(data_dir)
    os.environ["HEKB_AUDIT_PUBLIC_KEY_PATH"] = str(public_key_path)

    import store_service  # noqa: E402  (imported after env vars are set; _data_dir() etc. read them lazily anyway)
    from starlette.testclient import TestClient

    from storage.trajectory_blob import DTYPE_INT64, encode_trajectory_blob

    manifest, model_ref_was_missing = _load_manifest(manifest_path)
    quant_step_log2 = manifest["quant_step_log2"]
    board_format = manifest["board_format"]
    manifest_anchor_policy = manifest["anchor_policy_sha256"]
    model_ref = manifest["model_ref"]
    layer = manifest["layer"]
    if model_ref_was_missing:
        print(f"[ingest_boardrecords] REPORT: manifest.json has no model_ref; using {MODEL_REF_UNKNOWN!r}", flush=True)

    space = {
        "model_ref": model_ref, "layer": layer,
        "projection_sha256": manifest.get("projection_sha256", "0" * 64),
        "w_sha256": manifest.get("w_sha256", "0" * 64),
        "anchor_policy_sha256": manifest_anchor_policy,
        "board_format": board_format, "quant_step_log2": quant_step_log2,
    }

    token = "ingest-dev-token"
    store_service._bearer_guard.token = token
    client = TestClient(store_service.app, headers={"Authorization": f"Bearer {token}"})

    space_resp = client.post("/v3/space-id", json=space)
    if space_resp.status_code != 200:
        raise IngestError(f"/v3/space-id failed: {space_resp.status_code} {space_resp.text}")
    space_id = space_resp.json()["space_id"]

    results: list[dict] = []
    sessions = manifest["sessions"]
    if only_session_ids is not None:
        sessions = [s for s in sessions if s["session_id"] in only_session_ids]

    for i, s in enumerate(sessions, start=1):
        session_id = s["session_id"]
        shard_path = shards_root / s["file"]
        records = _read_shard(shard_path)
        if len(records) != s["n_records"]:
            raise IngestError(
                f"session {session_id}: shard has {len(records)} records, "
                f"manifest declares n_records={s['n_records']}"
            )
        rows, port_ids = _session_ports_matrix(records, session_id, manifest_anchor_policy)

        # created_at_ms=0 (fixed, not wall-clock): the blob's bytes -- and therefore its
        # sha256 and the resulting object_id -- must be identical on every re-run for the
        # same shard content (task's own "冪等（再実行しても同じ object_id）"). The real
        # creation time already lives in the BoardRecord data itself, not in this wrapper.
        file_bytes, _header = encode_trajectory_blob(rows, dimensions=len(port_ids), dtype=DTYPE_INT64, created_at_ms=0)
        blob_sha256 = hashlib.sha256(file_bytes).hexdigest()
        blob_resp = client.put(f"/v3/blobs/{blob_sha256}", content=file_bytes,
                                headers={"Content-Type": "application/octet-stream"})
        if blob_resp.status_code != 201:
            raise IngestError(f"session {session_id}: PUT /v3/blobs failed: {blob_resp.status_code} {blob_resp.text}")

        payload = {
            "schema": "hekb.experience/2",
            "raw_input": f"boardrecord session={session_id}",
            "resolved_meaning": None, "action": None, "title": session_id, "tags": ["boardrecord"],
            "ext": {"hekbv3": {
                "kind": "trajectory", "origin": "experience",
                "space": {**space, "space_id": space_id},
                "ports": {
                    "blob_sha256": blob_sha256, "n_steps": len(rows), "dim": len(port_ids),
                    "port_ids": port_ids, "quant_step_log2": quant_step_log2,
                },
                "labels": {"category": s["category"], "split": s["split"], "group": s["group"]},
                "outcome": {"result": "n/a"},
            }},
        }
        sig = sign(payload, store_service.canonical_bytes)
        resp = client.post("/v3/trajectories", json={"payload": payload}, headers={"X-Audit-Signature": sig})
        if resp.status_code != 201:
            raise IngestError(f"session {session_id}: POST /v3/trajectories failed: {resp.status_code} {resp.text}")
        body = resp.json()
        results.append({
            "session_id": session_id, "object_id": body["object_id"],
            "n_records": len(rows), "category": s["category"], "group": s["group"], "split": s["split"],
        })
        if i % progress_every == 0 or i == len(sessions):
            print(f"[ingest_boardrecords] {i}/{len(sessions)} sessions ingested", flush=True)

    return {
        "space_id": space_id, "data_dir": str(data_dir), "n_sessions": len(results), "sessions": results,
        "model_ref": model_ref, "model_ref_was_missing": model_ref_was_missing, "layer": layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--shards-root", type=Path, default=None,
                         help="Directory the manifest's 'file' entries are relative to (default: manifest's own directory).")
    parser.add_argument("--data-dir", required=True, type=Path,
                         help="HEKB_DATA_DIR for this ingest (no default -- never points at production).")
    parser.add_argument("--only", nargs="*", default=None, help="Restrict to these session_ids (default: all in manifest).")
    args = parser.parse_args()

    shards_root = args.shards_root or args.manifest.resolve().parent
    only = set(args.only) if args.only else None
    try:
        result = run_ingest(
            manifest_path=args.manifest, shards_root=shards_root, data_dir=args.data_dir,
            only_session_ids=only,
        )
    except IngestError as exc:
        print(f"[ingest_boardrecords] STOP: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(json.dumps({k: v for k, v in result.items() if k != "sessions"}, indent=1))


if __name__ == "__main__":
    main()
