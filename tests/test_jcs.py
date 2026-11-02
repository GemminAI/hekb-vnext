# ruff: noqa: E501
"""graph.jcs: RFC 8785 canonical JSON for the values CTP seals (no floating point), and its use by the HEXT projection.

The projection regression compares ``sealed_bytes``, ``_state_hash`` and the node ids / ``blob_sha256`` of
``project_document`` with a copy of the implementation that ``graph/hext_projection.py`` had before it called
``graph.jcs`` (``_legacy_*`` below), on every document the projection tests use. Those tests use the P0 example
documents of GemminAI/hext (test data) and are skipped when they are absent.
"""

import glob
import hashlib
import json
from pathlib import Path

import pytest

from graph import chainlog
from graph.hext_projection import _state_hash, project_document, sealed_bytes
from graph.jcs import (
    MAX_SAFE_INTEGER,
    FloatNotAllowed,
    IntegerOutOfRange,
    JcsError,
    canonical_bytes,
    canonicalize,
)
from graph.model import GraphError
from graph.store import GraphStore

EX = Path.home() / "GemminAI" / "hext" / "examples" / "v2"
needs_hext_docs = pytest.mark.skipif(not (EX / "source-three-sentences.hext2.json").exists(), reason="P0 examples absent")


# ---- RFC 8785 ----------------------------------------------------------------------------------------
def test_rfc8785_section_3_2_3_sorts_keys_by_utf16_code_units():
    """RFC 8785 section 3.2.3: the order is by UTF-16 code units, so U+1F600 (D83D DE00) comes before U+FB33."""
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\u0080": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    assert list(json.loads(canonicalize(value))) == ["\r", "1", "\u0080", "ö", "€", "\U0001f600", "דּ"]
    assert canonicalize(value).startswith('{"\\r":"Carriage Return","1":"One","\u0080":"Control","ö":')


def test_rfc8785_appendix_e_string_and_literals():
    """RFC 8785 Appendix E: the string member and the literals (its number array holds floats, so it is not used)."""
    s = "€$\u000f\nA'B\"\\\\\"/"
    assert canonicalize(s) == '"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"'
    assert canonicalize({"literals": [None, True, False], "string": s}) == '{"literals":[null,true,false],"string":"€$\\u000f\\nA\'B\\"\\\\\\\\\\"/"}'


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ({}, "{}"),
        ([], "[]"),
        ({"a": {}, "b": []}, '{"a":{},"b":[]}'),
        ({"b": 1, "a": 2, "B": 3}, '{"B":3,"a":2,"b":1}'),
        ([[], [[]], {"x": [{}]}], '[[],[[]],{"x":[{}]}]'),
        ({"z": {"y": {"x": [1, {"w": None}]}}}, '{"z":{"y":{"x":[1,{"w":null}]}}}'),
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (-7, "-7"),
        (123456789, "123456789"),
        ("", '""'),
        ("a\"b\\c/d", '"a\\"b\\\\c/d"'),
        ("\b\f\n\r\t", '"\\b\\f\\n\\r\\t"'),
        ("\u0000\u0001\u001f", '"\\u0000\\u0001\\u001f"'),
        ("\u007f  ", '"\u007f  "'),
        ("é€\U0001f600", '"é€\U0001f600"'),
        ((1, 2), "[1,2]"),
    ],
)
def test_canonical_text(value, text):
    assert canonicalize(value) == text
    assert canonical_bytes(value) == text.encode("utf-8")
    assert json.loads(canonicalize(value)) == (list(value) if isinstance(value, tuple) else value)


def test_a_boolean_is_not_an_integer_and_zero_is_not_false():
    assert canonicalize([True, 1, False, 0]) == "[true,1,false,0]"


def test_input_order_and_container_type_do_not_change_the_output():
    assert canonicalize({"a": 1, "b": [1, 2]}) == canonicalize({"b": [1, 2], "a": 1}) == canonicalize({"b": (1, 2), "a": 1})


# ---- refusals ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.0, 1e300, float("nan"), float("inf"), float("-inf")])
def test_a_float_is_refused_at_any_depth(value):
    for wrapped in (value, [value], {"k": {"k": [value]}}):
        with pytest.raises(FloatNotAllowed):
            canonicalize(wrapped)


def test_the_float_refusal_is_a_jcs_error_and_a_graph_error():
    assert issubclass(FloatNotAllowed, JcsError) and issubclass(JcsError, GraphError)
    assert issubclass(IntegerOutOfRange, JcsError)


@pytest.mark.parametrize("n", [MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER, 0, 2**31, 2**52])
def test_integers_inside_the_safe_range_pass(n):
    assert canonicalize(n) == str(n) and canonicalize([n]) == f"[{n}]"
    assert MAX_SAFE_INTEGER == 9007199254740991


@pytest.mark.parametrize("n", [MAX_SAFE_INTEGER + 1, -MAX_SAFE_INTEGER - 1, 2**64, -(10**30)])
def test_integers_outside_the_safe_range_are_refused(n):
    with pytest.raises(IntegerOutOfRange, match="pass it as a string"):
        canonicalize({"n": n})
    assert canonicalize(str(n)) == f'"{n}"'  # as a string it is fine


@pytest.mark.parametrize("bad", [{1: "x"}, {None: 1}, {b"k": 1}])
def test_object_keys_must_be_strings(bad):
    with pytest.raises(JcsError, match="not a string"):
        canonicalize(bad)


@pytest.mark.parametrize("bad", [{1, 2}, b"x", object(), 1 + 2j])
def test_other_types_are_refused(bad):
    with pytest.raises(JcsError, match="not a canonicalizable value"):
        canonicalize([bad])


@pytest.mark.parametrize("bad", ["\ud800", "a\udfffb", {"\ud83d": 1}])
def test_a_lone_surrogate_is_refused(bad):
    with pytest.raises(JcsError, match="lone surrogate"):
        canonicalize(bad)


# ---- what is NOT changed ----------------------------------------------------------------------------------
def test_the_chain_log_serialisation_and_general_writes_still_accept_floats(tmp_path):
    assert chainlog.canonical({"b": 0.5, "a": "é"}) == '{"a":"\\u00e9","b":0.5}'  # D5: ensure_ascii, floats allowed
    s = GraphStore(tmp_path)
    assert s.add_node("n", "trajectory", attrs={"score": 0.5, "big": 2**60}) is True
    assert s.add_edge("measured_under", "n", "ps", {"w": 0.25}) is True
    assert s.nodes()[0]["attrs"] == {"score": 0.5, "big": 2**60}
    import graph.store

    assert "jcs" not in Path(chainlog.__file__).read_text() + Path(graph.store.__file__).read_text()


# ---- the HEXT projection gives the same output as before ---------------------------------------------------
def _legacy_q(s):
    out = ['"']
    for ch in s:
        o = ord(ch)
        if 0xD800 <= o <= 0xDFFF:
            raise GraphError("lone surrogate")
        out.append({'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\u{o:04x}" if o < 0x20 else ch))
    out.append('"')
    return "".join(out)


def _legacy_enc(v):
    """graph/hext_projection.py before graph.jcs (3a796f3), unchanged."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return _legacy_q(v)
    if isinstance(v, list):
        return "[" + ",".join(_legacy_enc(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(_legacy_q(k) + ":" + _legacy_enc(v[k]) for k in sorted(v, key=lambda k: k.encode("utf-16-be"))) + "}"
    raise GraphError(f"not a sealable value: {type(v).__name__}")


def _legacy_state_hash(doc):
    d = json.loads(json.dumps(doc))
    d["final_seal"].pop("state_hash", None)
    d["final_seal"].pop("signature", None)
    return "sha256:" + hashlib.sha256(_legacy_enc(d).encode("utf-8")).hexdigest()


def _documents():
    docs = sorted(glob.glob(str(EX / "*.hext2.json")))
    docs += sorted(glob.glob(str(Path.home() / "vaults" / "20260124" / "experiment-evidence" / "HEXT-V2-GEN001" / "docs" / "*.json")))
    return [Path(p) for p in docs]


@needs_hext_docs
def test_the_legacy_copy_and_the_new_function_agree_on_every_document():
    docs = _documents()
    assert len(docs) >= 3
    for path in docs:
        doc = json.loads(path.read_bytes())
        assert canonicalize(doc) == _legacy_enc(doc), path.name


@needs_hext_docs
def test_projection_outputs_are_unchanged_for_every_input():
    for path in _documents():
        raw = path.read_bytes()
        if "closure-failed" in path.name:  # an annotation the projection tests do not use
            continue
        doc = json.loads(raw)
        assert _state_hash(doc) == _legacy_state_hash(doc) == doc["final_seal"]["state_hash"]
        sb = sealed_bytes(raw)
        assert sb == _legacy_enc(doc).encode("utf-8")
        out = project_document(raw, resolve_session=lambda s: None)
        node = next(n for n in out["nodes"] if n["node_type"] == "hext_document")
        assert node["attrs"]["blob_sha256"] == hashlib.sha256(sb).hexdigest()
        assert node["node_id"] == "hextdoc:" + doc["final_seal"]["state_hash"]


@needs_hext_docs
@pytest.mark.parametrize("name", ["float-in-sealed-value", "exponent-in-sealed-value"])
def test_a_document_with_a_float_is_still_refused(name):
    raw = (EX / "invalid" / f"{name}.hext2.json").read_bytes()
    with pytest.raises(GraphError, match="not a sealable value"):
        sealed_bytes(raw)
    with pytest.raises(GraphError):
        project_document(raw, resolve_session=lambda s: None)
