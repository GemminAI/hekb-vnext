"""Formal Store (U6T-2a): T-T2-01 .. T-T2-13 of design U6T-2 v0.1, section 9, and the reads of section 5.

Every verdict is made in the test from its envelope (gv_support): the hashes are computed with the vendored JCS, nothing is copied
from a stored verdict. A refusal must leave the store as it was (``refused`` checks it).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from gv_support import (
    BUNDLE,
    BUNDLE_SHA,
    CATALOG_ROWS,
    OBS,
    SENSOS_COMMIT,
    clone,
    decl,
    lookup_of,
    make_env,
    make_verdict,
    seal_verdict,
    sha,
)

from graph import formal
from graph.blobs import blob_path, write_blob
from graph.chainlog import ChainLog
from graph.ic04 import IC04Error, IC04Store

REPO = Path(__file__).resolve().parents[1]
CHAIN = Path("graph") / "ic00_envelopes.jsonl"
WINDOW = {"id": "msr.trajectory-window", "version": "0.10.0"}


def new_store(tmp_path: Path, **changes: Any) -> IC04Store:
    return IC04Store.create_formal(tmp_path / "s", decl(**changes))


def reopen(store: IC04Store, **kw: Any) -> IC04Store:
    return IC04Store(store.data_dir, require_formal=True, **kw)


def put(
    store: IC04Store,
    env: dict[str, Any],
    *,
    lookup: Any = None,
    blobs: Any = None,
    bundle: Any = BUNDLE,
    verdict: Any = None,
    **kw: Any,
) -> dict[str, Any]:
    v = verdict if verdict is not None else make_verdict(env, lookup=lookup, **kw)
    return store.put(env, blobs, gate_verdict=v, checker_bundle=bundle)


def state(store: IC04Store) -> tuple[int, int, list[str]]:
    blobs = sorted(
        str(p.relative_to(store.data_dir))
        for p in (store.data_dir / "blobs").rglob("*")
        if p.is_file()
    )
    return len(store), len(store.log), blobs


def refused(store: IC04Store, code: str, env: dict[str, Any], **kw: Any) -> None:
    before = state(store)
    with pytest.raises(IC04Error) as raised:
        put(store, env, **kw)
    assert raised.value.code == code, raised.value
    assert state(store) == before, "a refused envelope must leave nothing behind"
    assert not store.has(env.get("artifact_id", ""))


def with_parent(
    store: IC04Store,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = make_env(seq=1)
    put(store, root)
    child = make_env(
        parents=[("generated_text", [root["artifact_id"]])],
        lookup=lookup_of(root),
        seq=2,
        payload={"b": 2},
    )
    return root, child, lookup_of(root)


# ---- T-T2-01 .. 05: the declaration -------------------------------------------------------


def test_t_t2_01_create_formal_only_in_an_empty_folder(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    assert store.is_formal and store.declaration == decl()
    assert not store.quarantined and len(store) == 0
    with pytest.raises(IC04Error) as raised:  # the same folder again
        IC04Store.create_formal(tmp_path / "s", decl())
    assert raised.value.code == "formal_store_not_empty"
    legacy = IC04Store(tmp_path / "old")
    legacy.put(_old_env(), {})
    with pytest.raises(IC04Error) as raised:
        IC04Store.create_formal(tmp_path / "old", decl())
    assert raised.value.code == "formal_store_not_empty"
    blob_dir = tmp_path / "blobs-only" / "blobs" / "aa" / "bb"
    blob_dir.mkdir(parents=True)
    (blob_dir / "x").write_bytes(b"x")
    with pytest.raises(IC04Error) as raised:
        IC04Store.create_formal(tmp_path / "blobs-only", decl())
    assert raised.value.code == "formal_store_not_empty"


def _old_env(**kw: Any) -> dict[str, Any]:
    return make_env(version="1.0.0", schema={"id": "x.test", "version": "0"}, **kw)


@pytest.mark.parametrize(
    "change",
    [
        {"declaration": "formal-store/2"},
        {"accepted_versions": []},
        {"accepted_payload_schemas": []},
        {"gate_verdict_versions": ["gate-verdict/2"]},
        {"allow_waived": True},
        {"missing_parents": "hold"},
        {"created_by": ""},
        {"extra": 1},
        {"allow_unverified": [{"check": "not_in_catalogue", "reason": "x"}]},
        {
            "unverifiable_catalog": {
                "version": "v",
                "sha256": "0" * 64,
                "rows": CATALOG_ROWS,
            }
        },
    ],
)
def test_create_formal_refuses_a_declaration_not_of_the_form(
    tmp_path: Path, change: dict[str, Any]
) -> None:
    with pytest.raises(IC04Error) as raised:
        IC04Store.create_formal(tmp_path / "s", decl(**change))
    assert raised.value.code == "formal_declaration_invalid"
    assert not (tmp_path / "s" / CHAIN).exists()


def test_the_declaration_is_seq_0_without_artifact_id_and_no_api_changes_it(
    tmp_path: Path,
) -> None:
    store = new_store(tmp_path)
    first = next(iter(store.log.verify()))
    assert first["seq"] == 0 and set(first["record"]) == {"ic04_store"}
    assert first["record"]["ic04_store"]["declaration"] == "formal-store/1"
    words = ("unquarantine", "release", "set_declaration", "amend", "redeclare")
    assert not [n for n in dir(store) if any(w in n.lower() for w in words)]
    assert reopen(store).declaration == store.declaration
    d = store.declaration
    assert d is not None
    assert d["accepted_versions"] == ["1.1.0"] and d["allow_waived"] is False
    assert d["gate_verdict_versions"] == ["gate-verdict/1"]
    accepted = {(p["id"], p["version"]) for p in d["accepted_payload_schemas"]}
    assert ("nvs.hidden-trajectory", "0.1.0") in accepted
    assert d["unverifiable_catalog"]["version"] == SENSOS_COMMIT
    assert d["created_by"] == "tests"


def test_t_t2_02_an_older_store_cannot_open_a_formal_one(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    # the code of the earlier IC04Store (origin/main 1c4c810): it reads record["artifact_id"] of every line
    done = subprocess.run(
        ["git", "show", "1c4c810:graph/ic04.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode:
        pytest.skip("git show of the earlier ic04.py is not available here")
    spec = importlib.util.spec_from_loader("ic04_before", loader=None)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ic04_before"] = module
    try:
        exec(  # noqa: S102
            compile(done.stdout, "ic04_before.py", "exec"), module.__dict__
        )  # our own earlier source
        with pytest.raises(KeyError):
            module.IC04Store(store.data_dir)
    finally:
        del sys.modules["ic04_before"]
    with pytest.raises(IC04Error) as raised:  # and today's class says so by name
        IC04Store(store.data_dir)
    assert raised.value.code == "formal_store_needs_require_formal"


def test_t_t2_03_require_formal_on_a_store_with_no_declaration(tmp_path: Path) -> None:
    legacy = IC04Store(tmp_path / "old")
    legacy.put(_old_env(), {})
    with pytest.raises(IC04Error) as raised:
        IC04Store(tmp_path / "old", require_formal=True)
    assert raised.value.code == "formal_declaration_missing"
    with pytest.raises(
        IC04Error
    ) as raised:  # an empty folder is not a Formal Store either
        IC04Store(tmp_path / "empty", require_formal=True)
    assert raised.value.code == "formal_declaration_missing"


def test_t_t2_04_a_broken_declaration_and_a_declaration_out_of_place(
    tmp_path: Path,
) -> None:
    broken = decl()
    del broken["allow_waived"]
    ChainLog(tmp_path / "a" / CHAIN).append({"ic04_store": broken})
    with pytest.raises(IC04Error) as raised:
        IC04Store(tmp_path / "a", require_formal=True)
    assert raised.value.code == "formal_declaration_invalid"
    extra_key = {"ic04_store": decl(), "artifact_id": "art:" + "0" * 64}
    ChainLog(tmp_path / "b" / CHAIN).append(extra_key)
    with pytest.raises(IC04Error) as raised:
        IC04Store(tmp_path / "b", require_formal=True)
    assert raised.value.code == "formal_declaration_invalid"
    store = new_store(tmp_path)  # a declaration after seq 0
    ChainLog(store.data_dir / CHAIN).append({"ic04_store": decl()})
    with pytest.raises(IC04Error) as raised:
        IC04Store(store.data_dir, require_formal=True)
    assert raised.value.code == "formal_declaration_misplaced"
    legacy = IC04Store(tmp_path / "old")  # a declaration put behind an older store
    legacy.put(_old_env(), {})
    ChainLog(tmp_path / "old" / CHAIN).append({"ic04_store": decl()})
    with pytest.raises(IC04Error) as raised:
        IC04Store(tmp_path / "old", require_formal=True)
    assert raised.value.code == "formal_declaration_missing"  # seq 0 is an envelope


def test_t_t2_05_open_arguments_that_differ_from_the_declaration(
    tmp_path: Path,
) -> None:
    store = new_store(tmp_path)
    for kw in (
        {"accepted_versions": ("1.0.0",)},
        {"accepted_versions": ("1.1.0", "1.0.0")},
        {"missing_parents": "hold"},
    ):
        with pytest.raises(IC04Error) as raised:
            reopen(store, **kw)
        assert raised.value.code == "formal_open_args_conflict"
    ok = reopen(store, accepted_versions=("1.1.0",), missing_parents="reject")
    assert ok.accepted_versions == ("1.1.0",) and ok.missing_parents == "reject"


# ---- T-T2-06 .. 11: put ---------------------------------------------------------------------


def test_a_good_envelope_is_stored_and_read_back_with_its_verdict(
    tmp_path: Path,
) -> None:
    store = new_store(tmp_path)
    env = make_env()
    verdict = make_verdict(env)
    receipt = put(store, env, verdict=verdict)
    assert receipt["hekb_seq"] == 1 and not receipt["already_stored"]
    got = store.get(env["artifact_id"])
    assert got is not None and got["envelope"] == env
    assert got["gate_verdict"] == verdict and "gate_ref" not in got
    last = list(store.log.verify())[-1]["record"]
    assert last["ic04"] == {
        "gate_verdict": verdict,
        "checker_bundle_sha256": BUNDLE_SHA,
    }
    assert store.checker_bundle(env["artifact_id"]) == BUNDLE
    assert blob_path(store.data_dir, BUNDLE_SHA).is_file()
    assert (
        reopen(store).get(env["artifact_id"]) == got
    )  # every record passes the open check
    again = put(store, env, verdict=verdict)  # the same envelope: the stored receipt
    assert again["already_stored"] and again["hekb_seq"] == 1 and len(store) == 1
    other = clone(env)
    other["seq"] = 9
    with pytest.raises(IC04Error) as raised:
        put(store, other)
    assert raised.value.code == "conflict"


def test_t_t2_06_c0_a_quarantined_store_refuses_put(tmp_path: Path) -> None:
    store = _quarantined(tmp_path)[0]
    refused(store, "formal_store_quarantined", make_env(payload={"z": 1}))


def test_t_t2_06_c1_the_old_acceptance(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    refused(store, "unsupported_contract_version", make_env(version="1.0.0"))
    ghost = "art:" + "7" * 64
    refused(
        store,
        "parent_unknown",
        make_env(
            parents=[("generated_text", [ghost])],
            lookup={ghost: {"content_id": "cnt:" + "7" * 64}},
        ),
    )
    env = make_env()
    bad = clone(env)
    bad["payload"] = {"a": 2}  # the ids no longer recompute
    refused(store, "id_mismatch", bad, verdict=make_verdict(env))
    refused(store, "blob_mismatch", env, blobs={"blob:sha256:" + "0" * 64: b"x"})
    refused(store, "malformed", {"artifact_id": "x"}, verdict=make_verdict(env))


def test_t_t2_11_a_payload_the_declaration_does_not_accept(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    refused(
        store,
        "payload_schema_not_accepted",
        make_env(schema={"id": "nvs.hidden-trajectory", "version": "9.9.9"}),
    )
    recall = {"id": "hekb.recall-event", "version": "0.1.0-provisional"}
    refused(store, "payload_schema_not_accepted", make_env(schema=recall))
    refused(
        store,
        "payload_schema_not_accepted",
        make_env(schema={"id": "x.test", "version": "0"}),
    )
    only = [{"id": "msr.calibration", "version": "0.8.0"}]
    narrow = new_store(tmp_path / "n", accepted_payload_schemas=only)
    refused(narrow, "payload_schema_not_accepted", make_env())


def test_t_t2_06_c2_a_verdict_is_required_and_a_gate_ref_is_not_accepted(
    tmp_path: Path,
) -> None:
    store = new_store(tmp_path)
    env = make_env()
    with pytest.raises(IC04Error) as raised:
        store.put(env, {})
    assert raised.value.code == "gate_required"
    with pytest.raises(IC04Error) as raised:
        store.put(env, {}, gate_ref={"checker": "x"})
    assert raised.value.code == "gate_ref_not_accepted"
    with pytest.raises(IC04Error) as raised:
        store.put(
            env,
            {},
            gate_ref={"checker": "x"},
            gate_verdict=make_verdict(env),
            checker_bundle=BUNDLE,
        )
    assert raised.value.code == "gate_ref_not_accepted"
    assert len(store) == 0


def _drop_gate(v: dict[str, Any]) -> None:
    v.pop("gate")


@pytest.mark.parametrize(
    "edit",
    [
        lambda v: v.update(verdict_version="gate-verdict/2"),
        lambda v: v.update(extra=1),
        _drop_gate,
        lambda v: v["gate"].pop("checker"),
        lambda v: v["gate"].pop("runtime"),
        lambda v: v["inputs"].pop("blobs_checked"),
        lambda v: v["subject"].update(content_id="cnt:short"),
        lambda v: v["layers"]["L1"].update(status="PASS_UNVERIFIED"),
        lambda v: v.update(result="FAIL"),
    ],
)
def test_t_t2_06_c3_a_verdict_that_is_not_of_the_form(
    tmp_path: Path, edit: Any
) -> None:
    store = new_store(tmp_path)
    env = make_env()
    v = make_verdict(env)
    edit(v)
    seal_verdict(v)  # sealed again: the form is wrong, the hash is right
    refused(store, "gate_verdict_malformed", env, verdict=v)


def test_t_t2_06_c3_damage_after_sealing_and_a_foreign_catalogue(
    tmp_path: Path,
) -> None:
    store = new_store(tmp_path)
    env = make_env()
    v = make_verdict(env)
    v["codes"] = ["x"]  # not resealed: verdict_sha256 no longer matches
    refused(store, "gate_verdict_malformed", env, verdict=v)
    v = make_verdict(env)
    v["inputs"]["lookup_sha256"] = (
        "1" * 64
    )  # resealed, but lookup_sha256 is not the lookup's
    body = {k: x for k, x in v.items() if k != "verdict_sha256"}
    from vendor.sensos_ic00.jcs import jcs

    v["verdict_sha256"] = sha(jcs(body))
    refused(store, "gate_verdict_malformed", env, verdict=v)
    v = make_verdict(
        env,
        edit=lambda d: d["gate"]["unverifiable_catalog"].update(sha256="2" * 64),
    )
    refused(store, "gate_catalog_mismatch", env, verdict=v)
    refused(store, "gate_verdict_malformed", env, verdict="not a verdict")


def test_t_t2_06_c4_the_verdict_must_be_about_this_envelope(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env, other = make_env(), make_env(payload={"a": 2})
    refused(store, "gate_subject_mismatch", env, verdict=make_verdict(other))
    wrong_hash = make_verdict(
        env, edit=lambda d: d["subject"].update(envelope_sha256="3" * 64)
    )
    refused(store, "gate_subject_mismatch", env, verdict=wrong_hash)
    wrong_content = make_verdict(
        env, edit=lambda d: d["subject"].update(content_id=other["content_id"])
    )
    refused(store, "gate_subject_mismatch", env, verdict=wrong_content)
    # HEKB computes the hash itself: an envelope that differs only in seq (outside the ids) is refused
    moved = clone(env)
    moved["seq"] = 5
    refused(store, "gate_subject_mismatch", moved, verdict=make_verdict(env))


def test_t_t2_06_c5_the_rules_must_be_the_envelopes(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env = make_env()
    edits = (
        lambda d: d["rules"].update(ic00_contract_version="1.0.0"),
        lambda d: d["rules"]["payload_schema"].update(version="0.2.0"),
        lambda d: d["rules"]["payload_schema"].update(id="msr.calibration"),
    )
    for edit in edits:
        refused(store, "gate_rule_mismatch", env, verdict=make_verdict(env, edit=edit))


def test_t_t2_09_the_lookup_must_repeat_the_stores_values(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    root, child, good = with_parent(store)
    pid = root["artifact_id"]
    changes = (
        ("seq", 7),
        ("content_id", "cnt:" + "9" * 64),
        ("observation_id", "obs:" + "8" * 64),
    )
    for key, value in changes:
        wrong = clone(good)
        wrong[pid][key] = value
        refused(store, "gate_lookup_mismatch", child, lookup=wrong)
    refused(
        store, "gate_lookup_mismatch", child, lookup={}
    )  # a parent the verdict does not cover
    extra = clone(good)
    extra["art:" + "5" * 64] = {
        "content_id": "cnt:" + "5" * 64,
        "seq": 1,
        "observation_id": OBS,
    }
    refused(
        store, "gate_lookup_mismatch", child, lookup=extra
    )  # an id the store does not have
    text = clone(good)
    text[pid]["seq"] = str(
        root["seq"]
    )  # "1" is not 1: the schema already refuses the form
    refused(store, "gate_verdict_malformed", child, lookup=text)
    assert put(store, child, lookup=good)["hekb_seq"] == 2  # the right one passes


def test_t_t2_06_c6_the_calibration_must_be_in_the_store(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    unknown = {"artifact_id": "art:" + "6" * 64, "content_id": "cnt:" + "6" * 64}
    refused(store, "calibration_unknown", make_env(calibration_ref=unknown))
    cal = make_env(
        schema={"id": "msr.calibration", "version": "0.8.0"},
        payload={"exponent": -30},
        seq=1,
        obs="obs:" + "4" * 64,
    )
    put(store, cal)
    known = {"artifact_id": cal["artifact_id"], "content_id": cal["content_id"]}
    env = make_env(calibration_ref=known, seq=2)
    refused(
        store, "gate_lookup_mismatch", env
    )  # a known calibration must be in the lookup
    assert put(store, env, lookup=lookup_of(cal))["hekb_seq"] == 2


def test_t_t2_06_c7_the_kind_of_pass(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env = make_env()
    points = (("points_bytes", "blob_not_supplied"),)
    refused(store, "gate_rejected", env, result="FAIL")
    refused(
        store,
        "gate_unverified_not_allowed",
        env,
        result="PASS_UNVERIFIED",
        unverifiable=points,
    )
    both = (("window_rows", "raw_not_in_artifact"), *points)
    refused(
        store,
        "gate_unverified_not_allowed",
        env,
        result="PASS_UNVERIFIED",
        unverifiable=both,
    )
    waived = make_verdict(env, edit=lambda d: d.update(result="WAIVED"))
    refused(store, "gate_waived_not_allowed", env, verdict=waived)
    waived = make_verdict(env, edit=lambda d: d.update(waived=[{"code": "x"}]))
    refused(store, "gate_waived_not_allowed", env, verdict=waived)
    win = make_env(
        schema=WINDOW, payload={"w": 1}
    )  # the two window rows are allowed by default
    window_rows = (
        ("window_rows", "raw_not_in_artifact"),
        ("window_space_derivation", "rule_not_derivable"),
    )
    assert (
        put(store, win, result="PASS_UNVERIFIED", unverifiable=window_rows)["hekb_seq"]
        == 1
    )
    wide = new_store(
        tmp_path / "w",
        allow_unverified=[
            {"check": "points_bytes", "reason": "blob_not_supplied"},
            {"check": "contract_binding.<field>", "reason": "contract_field_unknown"},
        ],
    )
    first = make_env()
    assert (
        put(wide, first, result="PASS_UNVERIFIED", unverifiable=points)["hekb_seq"] == 1
    )
    typed = (("contract_binding.origin.model", "contract_field_unknown"),)
    second = make_env(payload={"c": 3}, seq=2)
    assert (
        put(wide, second, result="PASS_UNVERIFIED", unverifiable=typed)["hekb_seq"] == 2
    )


def test_t_t2_08_a_fail_envelope_is_not_stored(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env = make_env()
    refused(store, "gate_rejected", env, result="FAIL")
    assert store.get(env["artifact_id"]) is None and len(store) == 0
    assert len(store.log) == 1  # the declaration only
    assert state(store)[2] == []  # not even the bundle was written
    assert store.select({}, 10) == ([], False)


def test_t_t2_10_the_checker_bundle(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env = make_env()
    refused(store, "checker_bundle_missing", env, bundle=None)
    refused(store, "checker_bundle_mismatch", env, bundle=b"another bundle")
    put(store, env)  # the first one brings the bundle ...
    second = make_env(payload={"n": 2}, seq=2)
    assert (
        put(store, second, bundle=None)["hekb_seq"] == 2
    )  # ... the next ones need not
    third = make_env(payload={"n": 3}, seq=3)
    refused(store, "checker_bundle_mismatch", third, bundle=b"another bundle")
    other = make_env(payload={"n": 4}, seq=4)
    named = make_verdict(
        other,
        edit=lambda d: d["gate"]["checker"].update(
            bundle_sha256=sha("a different bundle")
        ),
    )
    refused(store, "checker_bundle_missing", other, verdict=named, bundle=None)
    assert sum(1 for f in state(store)[2] if f.endswith(BUNDLE_SHA)) == 1


def test_t_t2_07_envelopes_shaped_like_the_u6_real_machine_run(tmp_path: Path) -> None:
    """A made-up capture with the form of U6-5e's envelopes: Llama one point, Qwen five; each with a points Blob and a token-index Blob."""
    qwen = ["mlp_out:15", "mlp_out:19", "block_out:15", "block_out:19", "final_norm"]
    for model, points in (("llama", ["final_norm"]), ("qwen", qwen)):
        store = new_store(tmp_path / model)
        stored = {}
        for seq, point in enumerate(points, start=1):
            names = {n: f"{model}-{point}-{n}" for n in ("points", "index")}
            blobs = {f"blob:sha256:{sha(v)}": v.encode() for v in names.values()}
            uris = list(blobs)
            payload = {
                "point": point,
                "length": 3,
                "points": {"uri": uris[0]},
                "token_index": {"uri": uris[1]},
            }
            env = make_env(
                payload=payload,
                seq=seq,
                space_id="hid:" + sha(point),
                obs="obs:" + sha(model),
            )
            verdict = make_verdict(env)
            put(
                store,
                env,
                blobs=blobs,
                verdict=verdict,
                bundle=BUNDLE if seq == 1 else None,
            )
            stored[point] = (env, blobs, verdict)
        assert len(store) == len(points)
        for env, blobs, verdict in stored.values():
            got = store.get(env["artifact_id"])
            assert got is not None and got["envelope"] == env and got["blobs"] == blobs
            assert got["gate_verdict"] == verdict
            assert store.gate_status(env["artifact_id"]) == {
                "self": "PASS",
                "verdict": verdict,
            }
        again = reopen(store)
        assert not again.quarantined and len(
            again.select({"kind": "observation"}, 10)[0]
        ) == len(points)


# ---- T-T2-12: quarantine and reads ----------------------------------------------------------


def _append(
    store: IC04Store,
    env: dict[str, Any],
    verdict: Any,
    *,
    bundle_sha: str | None = BUNDLE_SHA,
    gate_ref: Any = None,
) -> None:
    """Chain a record by hand, as a writer that skipped ``put`` would (the chain stays valid)."""
    write_blob(store.data_dir, BUNDLE)
    ic04: dict[str, Any] = {
        "gate_verdict": verdict,
        "checker_bundle_sha256": bundle_sha,
    }
    if gate_ref is not None:
        ic04 = {"gate_ref": gate_ref}
    record = {
        "artifact_id": env["artifact_id"],
        "content_id": env["content_id"],
        "envelope": env,
        "blob_shas": [],
        "ic04": ic04,
    }
    ChainLog(store.data_dir / CHAIN).append(record, sync=True)


def _quarantined(tmp_path: Path) -> tuple[IC04Store, dict[str, Any], dict[str, Any]]:
    store = new_store(tmp_path)
    good = make_env()
    put(store, good)
    bad = make_env(payload={"bad": 1}, seq=2)
    _append(
        store, bad, make_verdict(bad, result="FAIL")
    )  # chained correctly, behind put's back
    return reopen(store), good, bad


def test_t_t2_12_a_fail_line_added_to_the_chain_quarantines_the_store(
    tmp_path: Path,
) -> None:
    store, good, bad = _quarantined(tmp_path)
    assert store.quarantined
    found = [(v["artifact_id"], v["code"], v["hekb_seq"]) for v in store.violations]
    assert found == [(bad["artifact_id"], "gate_rejected", 2)]
    got = store.get(good["artifact_id"])  # reads still work
    assert got is not None and got["envelope"] == good
    assert store.get(bad["artifact_id"]) is not None
    bad_status = store.gate_status(bad["artifact_id"])
    good_status = store.gate_status(good["artifact_id"])
    assert bad_status is not None and bad_status["self"] == "FAIL_RECORDED"
    assert good_status is not None and good_status["self"] == "PASS"
    entries = store.lineage(bad["artifact_id"])["entries"]
    assert {e["artifact_id"] for e in entries} == {bad["artifact_id"]}
    assert set(store.select({}, 10)[0]) == {good["artifact_id"], bad["artifact_id"]}
    for env in (make_env(payload={"n": 5}, seq=3), good):
        before = len(store.log)
        with pytest.raises(IC04Error) as raised:
            put(store, env)
        assert raised.value.code == "formal_store_quarantined"
        assert len(store.log) == before
    context = {
        "session_id": "s",
        "observation_id": OBS,
        "seq": 9,
        "produced_at": "2026-11-17T00:00:00Z",
    }
    with pytest.raises(IC04Error) as raised:
        store.recall({}, requester="r", context=context)
    assert raised.value.code == "recall_not_supported_in_formal_store"
    assert reopen(store).quarantined  # and it stays so: nothing clears it
    words = ("unquarantine", "release", "clear")
    assert not [n for n in dir(store) if any(w in n.lower() for w in words)]


def test_the_other_ways_a_record_can_break_the_declaration(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env())
    cases: dict[str, tuple[dict[str, Any], Any, dict[str, Any]]] = {}
    e1 = make_env(payload={"c": 1}, seq=2)
    cases["gate_required"] = (e1, None, {})
    e2 = make_env(payload={"c": 2}, seq=3)
    tampered = make_verdict(e2)
    tampered["codes"] = ["x"]
    cases["gate_verdict_malformed"] = (e2, tampered, {})
    e3 = make_env(payload={"c": 3}, seq=4)
    cases["gate_subject_mismatch"] = (
        e3,
        make_verdict(make_env(payload={"other": 1})),
        {},
    )
    e4 = make_env(payload={"c": 4}, seq=5)
    unverified = make_verdict(
        e4,
        result="PASS_UNVERIFIED",
        unverifiable=(("points_bytes", "blob_not_supplied"),),
    )
    cases["gate_unverified_not_allowed"] = (e4, unverified, {})
    e5 = make_env(payload={"c": 5}, seq=6)
    cases["gate_ref_not_accepted"] = (e5, None, {"gate_ref": {"checker": "legacy"}})
    e6 = make_env(payload={"c": 6}, seq=7, schema={"id": "x.test", "version": "0"})
    cases["payload_schema_not_accepted"] = (e6, make_verdict(e6), {})
    e7 = make_env(payload={"c": 7}, seq=8)
    gone = make_verdict(
        e7, edit=lambda d: d["gate"]["checker"].update(bundle_sha256=sha("gone"))
    )
    cases["checker_bundle_missing"] = (e7, gone, {"bundle_sha": sha("gone")})
    for env, verdict, extra in cases.values():
        _append(
            store,
            env,
            verdict,
            bundle_sha=extra.get("bundle_sha", BUNDLE_SHA),
            gate_ref=extra.get("gate_ref"),
        )
    reopened = reopen(store)
    assert reopened.quarantined
    found = {v["artifact_id"]: v["code"] for v in reopened.violations}
    assert found == {env["artifact_id"]: code for code, (env, _v, _x) in cases.items()}
    ChainLog(reopened.data_dir / CHAIN).append(
        {"nothing": 1}, sync=True
    )  # not even a record
    assert reopen(store).violations[-1]["code"] == "record_malformed"
    status = reopen(store).gate_status(e2["artifact_id"])
    assert status is not None and status["self"] == "INCONSISTENT"


def test_a_chain_that_is_broken_does_not_open_at_all(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    put(store, make_env())
    path = store.data_dir / CHAIN
    path.write_text(
        path.read_text(encoding="utf-8").replace('"a":1', '"a":2'), encoding="utf-8"
    )
    with pytest.raises(Exception, match="chain broken"):
        reopen(store)


def test_gate_status_of_a_stored_record(tmp_path: Path) -> None:
    store = new_store(tmp_path)
    env = make_env(schema=WINDOW, payload={"w": 1})
    v = make_verdict(
        env,
        result="PASS_UNVERIFIED",
        unverifiable=(("window_rows", "raw_not_in_artifact"),),
    )
    put(store, env, verdict=v)
    assert store.gate_status(env["artifact_id"]) == {
        "self": "PASS_UNVERIFIED",
        "verdict": v,
    }
    assert store.gate_status("art:" + "0" * 64) is None


# ---- T-T2-13: a store not opened with require_formal is as it was ---------------------------


def test_t_t2_13_a_store_not_opened_with_require_formal_behaves_as_before(
    tmp_path: Path,
) -> None:
    store = IC04Store(tmp_path / "d")
    assert store.accepted_versions == ("1.0.0",) and store.missing_parents == "reject"
    assert not store.is_formal and not store.quarantined
    env = _old_env()
    receipt = store.put(env, {}, gate_ref={"checker": "x"})
    assert receipt["hekb_seq"] == 0
    got = store.get(env["artifact_id"])
    assert (
        got is not None
        and got["gate_ref"] == {"checker": "x"}
        and "gate_verdict" not in got
    )
    assert store.gate_status(env["artifact_id"]) == {
        "self": "LEGACY",
        "verdict": {"checker": "x"},
    }
    bare = _old_env(payload={"b": 1}, seq=2)
    store.put(bare, {})
    assert store.gate_status(bare["artifact_id"]) == {"self": "NONE", "verdict": None}
    with pytest.raises(IC04Error) as raised:  # a verdict is for a Formal Store
        store.put(_old_env(payload={"c": 1}, seq=3), {}, gate_verdict=make_verdict(env))
    assert raised.value.code == "formal_store_required" and len(store) == 2
    assert IC04Store(tmp_path / "d").has(env["artifact_id"])
    old_style = IC04Store(
        tmp_path / "e", accepted_versions=("1.1.0",), missing_parents="hold"
    )
    assert (
        old_style.accepted_versions == ("1.1.0",)
        and old_style.missing_parents == "hold"
    )
    assert (
        formal.DEFAULT_ALLOW_UNVERIFIED
    )  # the formal defaults exist and do not touch the old path
