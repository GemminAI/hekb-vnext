"""Import one stored session (objects/meta/lineage/blobs of a source data dir) into a new data dir
and write its graph nodes and edges. The source is only read; the copied files are byte-identical
(checked), so ``object_id`` and every blob keep their values.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from graph.blobs import blob_path, read_blob, write_blob
from graph.chainlog import canonical
from graph.content import content_id
from graph.source_text import Decoder, build
from graph.store import GraphStore

CAPACITY = 8
HEXT_SCHEMA = "hext/1.0"


class LockedSessionError(RuntimeError):
    pass


def policy_set_id(policies: dict[str, str]) -> str:
    return "policy_set:" + hashlib.sha256(canonical(policies).encode()).hexdigest()


def _copy_checked(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.read_bytes() != src.read_bytes():
            raise RuntimeError(f"{dst} exists with different bytes")
        return
    shutil.copyfile(src, dst)
    if hashlib.sha256(src.read_bytes()).digest() != hashlib.sha256(dst.read_bytes()).digest():
        raise RuntimeError(f"copy of {src} differs")


def is_locked(assignment: dict[str, str], session_id: str) -> bool:
    return str(assignment.get(session_id, "")).endswith("_locked")


def ingest_session(
    *,
    src: Path,
    dst: Path,
    store: GraphStore,
    session_id: str,
    object_id: str,
    msd_dir: Path,
    decoder: Decoder,
    window_policy: dict[str, str],
    assignment: dict[str, str],
) -> dict[str, Any]:
    if is_locked(assignment, session_id):
        raise LockedSessionError(session_id)
    for sub in ("objects", "meta", "lineages"):
        source = src / sub / f"{object_id}.json"
        if source.exists():
            _copy_checked(source, dst / sub / f"{object_id}.json")
    obj = json.loads((dst / "objects" / f"{object_id}.json").read_text())
    ext = obj["payload"]["ext"]["hekbv3"]
    space = ext["space"]
    space_id = space["space_id"]
    cl_sha = ext["causal_log"]["blob_sha256"]
    ports_sha = ext["ports"]["blob_sha256"]
    for sha in (cl_sha, ports_sha):
        _copy_checked(blob_path(src, sha), blob_path(dst, sha))
        read_blob(dst, sha)  # re-verifies the content hash
    records: list[dict[str, Any]] = json.loads(read_blob(dst, cl_sha))
    n = len(records)

    meta = json.loads((msd_dir / session_id / "meta.json").read_text())
    ids = [int(x) for x in np.load(msd_dir / session_id / "tokens.npz")["ids"]]
    one_to_one = n == len(ids) == ext["causal_log"]["n_steps"] and all(
        r["token_index"] == i for i, r in enumerate(records)
    )
    st = build(decoder, session_id, ids, meta["generated_text"])
    st_sha = write_blob(dst, canonical(st).encode("utf-8"))
    st_verified = one_to_one and st["decode_equal_generated_text_modulo_whitespace"]

    n_ok = all(
        (r["ports"]["P18"].get("result") or {}).get("n") == min(t + 1, CAPACITY)
        for t, r in enumerate(records)
    )
    policies = {
        "window_policy_name": window_policy["name"],
        "window_policy_sha256": window_policy["sha256"],
        "time_policy_sha256": space["time_policy_sha256"],
        "wiring_policy_sha256": space["wiring_policy_sha256"],
        "anchor_policy_sha256": space["anchor_policy_sha256"],
    }
    ps_id = policy_set_id(policies)
    store.add_node(
        ps_id,
        "policy_set",
        kind="observation",
        verification="machine_verified" if n_ok else "unverified",
        attrs={
            **policies,
            "check": f"P18 result.n == min(token_index+1, {CAPACITY}) for every token: {n_ok}",
            "note": "window policy is not stored in the space; attached at ingest (D-65 follow-up)",
        },
    )

    cid = content_id(space_id, records)
    store.add_node(
        object_id,
        "trajectory",
        kind="observation",
        verification="machine_verified" if one_to_one else "unverified",
        attrs={
            "session_id": session_id,
            "space_id": space_id,
            "content_id": cid,
            "n_steps": n,
            "title": obj["payload"].get("title"),
        },
    )
    cl_node, st_node = f"blob:{cl_sha}", f"blob:{st_sha}"
    store.add_node(
        cl_node, "causal_log", verification="machine_verified", attrs={"n_steps": n, "schema": "hekbv3.causal_log/1"}
    )
    store.add_node(
        st_node,
        "source_text",
        verification="machine_verified" if st_verified else "unverified",
        attrs={"session_id": session_id, "n_tokens": len(ids), "schema": st["schema"]},
    )
    store.add_edge("has_causal_log", object_id, cl_node, {"token_start": 0, "token_end": n})
    store.add_edge(
        "has_source_text",
        object_id,
        st_node,
        {"token_start": 0, "token_end": len(ids), "char_start": 0, "char_end": len(st["text"])},
    )
    store.add_edge("measured_under", object_id, ps_id, {})
    for t, record in enumerate(records):
        hext = record["cycle_raw"]["hext"]
        hid = f"hext:{hext['object_id']}"
        store.add_node(
            hid,
            "hext_kernel",
            verification="unverified",
            attrs={k: hext.get(k) for k in ("schema_version", "risk", "tier", "risk_state", "basin")},
        )
        store.add_edge(
            "token_hext", cl_node, hid, {"token_index": t, "via": "cycle_raw.hext", "schema": HEXT_SCHEMA}
        )
    return {
        "session": session_id,
        "object_id": object_id,
        "content_id": cid,
        "space_id": space_id,
        "n_tokens": n,
        "one_to_one": one_to_one,
        "source_text_verified": st_verified,
        "policy_set": ps_id,
    }


def link_same_content(store: GraphStore) -> int:
    """Add ``same_content_as`` edges between trajectories with equal ``content_id`` (a < b)."""
    groups: dict[str, list[str]] = {}
    for node in store.nodes():
        if node["node_type"] == "trajectory":
            groups.setdefault(node["attrs"]["content_id"], []).append(node["node_id"])
    added = 0
    for cid, members in groups.items():
        members = sorted(members)
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                added += int(store.add_edge("same_content_as", a, b, {"content_id": cid}))
    return added
