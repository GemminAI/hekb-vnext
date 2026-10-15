import json

from graph.rebuild_check import compare, digest_of

B = {"git_commit": "c1", "rules": {"R": "s1"}}
BASE = {"build": B, "inputs": {"i1": "a", "i2": "b"}, "outputs": {"o1": "x", "o2": "y"}}


def test_rebuild_equal():
    r = compare(BASE, {"i1": "a", "i2": "b"}, {"o1": "x", "o2": "y"}, B)
    assert r["verdict"] == "REBUILD_EQUAL" and r["files_compared"] == 2


def test_changed_input_makes_no_claim_about_outputs():
    r = compare(BASE, {"i1": "a", "i2": "CHANGED"}, {"o1": "different", "o2": "y"}, B)
    assert r["verdict"] == "INPUTS_CHANGED" and r["changed_inputs"] == ["i2"]


def test_added_or_missing_input_counts_as_changed():
    assert compare(BASE, {"i1": "a"}, {"o1": "x", "o2": "y"}, B)["verdict"] == "INPUTS_CHANGED"
    assert compare(BASE, {"i1": "a", "i2": "b", "i3": "c"}, {"o1": "x", "o2": "y"}, B)["verdict"] == "INPUTS_CHANGED"


def test_output_difference_is_told_apart_by_build_identity():
    same = compare(BASE, {"i1": "a", "i2": "b"}, {"o1": "x", "o2": "Z"}, B)
    assert same["verdict"] == "OUTPUT_DIFFERS" and same["differing_files"] == ["o2"] and same["reading"] == "non-determinism"
    other = compare(BASE, {"i1": "a", "i2": "b"}, {"o1": "x", "o2": "Z"}, {"git_commit": "c2", "rules": {"R": "s1"}})
    assert other["verdict"] == "OUTPUT_DIFFERS" and other["same_code_and_rules"] is False


def test_output_file_set_mismatch():
    r = compare(BASE, {"i1": "a", "i2": "b"}, {"o1": "x", "o3": "y"}, B)
    assert r["verdict"] == "OUTPUT_FILES_MISMATCH" and r["missing"] == ["o2"] and r["extra"] == ["o3"]


def test_digest_of_is_order_independent():
    assert digest_of({"a": "1", "b": "2"}) == digest_of({"b": "2", "a": "1"})


def test_bind_baseline_keeps_original_and_records_the_commit_mapping():
    from graph.rebuild_check import bind_baseline

    base = {"build": {"git_commit": "orig", "working_tree_dirty": False, "rules": {}},
            "inputs": {"i": "a"}, "outputs": {"o": "x"}}
    snapshot = json.dumps(base, sort_keys=True)
    b = bind_baseline(base, remote_commit="rem", mapping=[{"original": "orig", "remote": "rem", "subject": "s"}],
                      tree_difference=["docs/x.md"])
    assert json.dumps(base, sort_keys=True) == snapshot  # the original is not modified
    assert b["build"]["git_commit"] == "rem" and b["bound_to_remote_commit"]["original_commit"] == "orig"
    assert b["bound_to_remote_commit"]["original_on_remote"] is False
    assert b["inputs"] == base["inputs"] and b["outputs"] == base["outputs"]
