"""The vendored GateVerdict schema and its standard-library validator (U6T-2a).

Always run: the LOCK, the validator on the verdicts of ``gv_support``, a schema with a keyword the validator lacks.
Opt-in (skipped, with the reason, otherwise): ``GATE_PY`` (an interpreter with ``jsonschema``) to check the validator against
``jsonschema`` on many damaged verdicts; ``SENSOS_TOOLS`` to check the copy against sensos's file; the U6T-1d evidence of a real run
to check that the form made here has the keys of a real verdict.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
from gv_support import make_env, make_verdict

from vendor.sensos_gate_verdict import validate
from vendor.sensos_gate_verdict.validate import Validator, verdict_errors

VENDOR = Path(validate.__file__).parent
SCHEMA_RELATIVE = "docs/architecture/contracts/nvs/gate-verdict.schema.json"
EVIDENCE = Path.home() / "vaults/20260124/experiment-evidence/U6T-1-RUN"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_lock_names_the_files_as_they_are() -> None:
    lock = json.loads((VENDOR / "LOCK.json").read_text(encoding="utf-8"))
    for name, entry in lock["files"].items():
        assert _sha(VENDOR / name) == entry["vendored_sha256"], name
    schema = lock["files"]["gate-verdict.schema.json"]
    assert schema["source_sha256"] == schema["vendored_sha256"]  # a byte-for-byte copy


def test_the_validator_accepts_the_verdicts_of_every_kind_and_refuses_damage() -> None:
    env = make_env()
    window = make_env(schema={"id": "msr.trajectory-window", "version": "0.10.0"})
    for v in (
        make_verdict(env),
        make_verdict(env, result="FAIL"),
        make_verdict(
            env,
            result="PASS_UNVERIFIED",
            unverifiable=(("points_bytes", "blob_not_supplied"),),
        ),
        make_verdict(window),
    ):
        assert verdict_errors(v) == []
    good = make_verdict(env)
    for damage in (
        lambda v: v.update(extra=1),
        lambda v: v.update(result="WAIVED"),
        lambda v: v.pop("gate"),
        lambda v: v["layers"]["L1"].update(status="PASS_UNVERIFIED"),
        lambda v: v.update(result="FAIL"),
        lambda v: v["subject"].update(content_id="cnt:short"),
        lambda v: v.update(waived=[{"x": 1}]),
        lambda v: v["inputs"].update(stored=[1]),
    ):
        bad = copy.deepcopy(good)
        damage(bad)
        assert verdict_errors(bad), damage


def test_a_schema_with_a_keyword_the_validator_lacks_is_refused_at_load() -> None:
    with pytest.raises(ValueError, match="does not implement"):
        Validator({"type": "object", "dependentRequired": {"a": ["b"]}})


def _shape(x: Any, opaque: tuple[str, ...] = ("lookup", "packages")) -> Any:
    if isinstance(x, dict):
        return {k: (None if k in opaque else _shape(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_shape(x[0])] if x else []
    return type(x).__name__ if x is not None else "null"


def test_the_form_made_here_has_the_keys_of_a_real_verdict_of_u6t1d() -> None:
    found = sorted(EVIDENCE.glob("u6t1-*/verdicts/final_norm.json"))
    if not found:
        pytest.skip(f"no U6T-1d evidence under {EVIDENCE.name}")
    real = json.loads(found[-1].read_text(encoding="utf-8"))
    mine = make_verdict(make_env())

    def keys(x: Any, path: str = "") -> set[str]:
        if not isinstance(x, dict):
            return set()
        out = set()
        for k, v in x.items():
            here = f"{path}.{k}"
            out.add(here)
            if k not in ("lookup", "packages"):
                out |= keys(v, here)
        return out

    assert keys(real) == keys(mine)
    assert verdict_errors(real) == []  # a real verdict fits the vendored schema
    # HEKB's own JCS (vendored) reproduces the real verdict's two hashes: the same bytes as sensos's JCS in the child
    from graph.formal import form_problem

    for path in found:
        assert form_problem(json.loads(path.read_text(encoding="utf-8"))) is None, path


# ---- against jsonschema (GATE_PY) --------------------------------------------------------------------


def _mutants(v: dict[str, Any]) -> list[Any]:
    out: list[Any] = [copy.deepcopy(v)]

    def visit(node: Any, path: list[Any]) -> None:
        containers = [node] if isinstance(node, (dict, list)) else []
        if isinstance(node, dict):
            for key in list(node):
                for variant in (None, 1, "x", [], {}, True):
                    m = copy.deepcopy(v)
                    _at(m, path)[key] = variant
                    out.append(m)
                m = copy.deepcopy(v)
                del _at(m, path)[key]
                out.append(m)
                visit(node[key], [*path, key])
            m = copy.deepcopy(v)
            _at(m, path)["zz_extra"] = 1
            out.append(m)
        elif isinstance(node, list):
            for i in range(len(node)):
                visit(node[i], [*path, i])
            m = copy.deepcopy(v)
            _at(m, path).append(copy.deepcopy(node[0]) if node else "x")
            out.append(m)
        del containers

    visit(v, [])
    return out


def _at(root: Any, path: list[Any]) -> Any:
    for p in path:
        root = root[p]
    return root


@pytest.mark.skipif(
    not os.environ.get("GATE_PY") or not os.environ.get("SENSOS_TOOLS"),
    reason="GATE_PY and SENSOS_TOOLS are not set",
)
def test_the_validator_agrees_with_jsonschema_on_damaged_verdicts() -> None:
    sensos = Path(os.environ["SENSOS_TOOLS"]).resolve().parent / SCHEMA_RELATIVE
    if not sensos.is_file():
        pytest.skip("SENSOS_TOOLS has no gate-verdict.schema.json")
    assert sensos.read_bytes() == (VENDOR / "gate-verdict.schema.json").read_bytes(), (
        "the vendored copy differs from sensos"
    )
    env = make_env()
    window = make_env(schema={"id": "msr.trajectory-window", "version": "0.10.0"})
    base = [
        make_verdict(env),
        make_verdict(env, result="FAIL"),
        make_verdict(
            env,
            result="PASS_UNVERIFIED",
            unverifiable=(("points_bytes", "blob_not_supplied"),),
        ),
        make_verdict(window),
    ]
    docs = [m for b in base for m in _mutants(b)]
    script = (
        "import json, sys\nfrom jsonschema import Draft202012Validator as V\n"
        "schema = json.load(open(sys.argv[1]))\nv = V(schema)\n"
        "print(json.dumps([not any(True for _ in v.iter_errors(d)) for d in json.load(sys.stdin)]))\n"
    )
    done = subprocess.run(
        [os.environ["GATE_PY"], "-c", script, str(sensos)],
        input=json.dumps(docs),
        capture_output=True,
        text=True,
        check=True,
    )
    theirs: list[bool] = json.loads(done.stdout)
    mine = [not verdict_errors(d) for d in docs]
    different = [i for i, (a, b) in enumerate(zip(mine, theirs, strict=True)) if a != b]
    assert len(docs) > 500
    assert not different, [
        (i, mine[i], theirs[i], verdict_errors(docs[i])[:2]) for i in different[:3]
    ]
