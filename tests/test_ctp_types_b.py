# ruff: noqa: E501
"""CTP episode records, second half (graph/rng.py, graph/selection.py, graph/ctp.py): assignment and experience.

The decisions these tests pin are those of 2026-11-03 (Tomo and Claude): no shuffle, selected_ids in drawn order, no
``order`` field, ``order_considered`` true, rng seeds are strings, one counter per selection.
"""

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # plain `pytest` does not put the repo root on sys.path

from graph import ctp, rng, selection  # noqa: E402
from graph.blobs import write_blob  # noqa: E402
from graph.ctp import CtpError  # noqa: E402
from graph.jcs import canonical_bytes  # noqa: E402
from graph.kindmap import check_stored_kind, d70_class  # noqa: E402
from graph.model import EDGE_TYPES, GraphError  # noqa: E402
from graph.store import GraphStore  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPEC_HASH = "sha256:" + "5" * 64
PRODUCER = {"component": "selector", "version": "1", "config_sha256": "c" * 64}
FORMAT = {"template_id": "slate-v1", "version": "1"}


@pytest.fixture
def store(tmp_path):
    s = GraphStore(tmp_path, v01=False, blob_dir=tmp_path)
    s.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    return s


def blob(store, items):
    return write_blob(store.blob_dir, selection.candidates_blob(items))


def write_episode(store, **over):
    fields = {"experiment_spec_hash": SPEC_HASH, "item_id": "i1", "bundle_id": "b1", "split": "dev", "condition": "E0", "replicate": 0, "rng_seed": 7, **over}
    return store.write_ctp_node("episode", fields, producer=PRODUCER)


def write_context(store, ep):
    fields = {"episode_id": ep, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False}
    return store.write_ctp_node("context_observation", fields, producer=PRODUCER)


def write_step(store, ep, step="f2"):
    sha = write_blob(store.blob_dir, f"context:{ep}/{step}".encode())
    fields = {"episode_id": ep, "step": step, "text_ref": "blob:" + "3" * 64, "input_context_ref": f"blob:{sha}", "input_context_sha256": sha}
    return store.write_ctp_node("step_output", fields, producer=PRODUCER)


CONDITIONS = ["E0", "E1", "E2", "E3", "E4"]


def assignment_fields(store, candidates=CONDITIONS, seed="a-seed", policy_params=None, **over):
    fields = selection.assignment_fields(SPEC_HASH, "i1", 0, candidates, blob(store, candidates), seed=seed, stratum="b1", policy_id="uniform", policy_version="1", policy_params=policy_params)
    fields.update(over)
    if "assignment_policy_fingerprint" not in over:  # a changed algorithm, stratum or params is meant to be a consistent record, unless the test says otherwise
        fields["assignment_policy_fingerprint"] = selection.assignment_fingerprint(fields)
    return fields


def write_assignment(store, **over):
    return store.write_ctp_node("assignment", assignment_fields(store, **over), producer=PRODUCER)


@pytest.fixture
def past(store):
    """Four earlier episodes and an outcome of the second one: what an experience may draw."""
    eps = [write_episode(store, item_id=f"past{i}", condition="E1") for i in range(4)]
    out = store.write_ctp_node("outcome", {"episode_id": eps[1], "carry": {"f1": "lost"}, "constraint_compliance": "FAIL", "two_axis": "LOST_VIOLATED"}, producer=PRODUCER)
    return {"episodes": eps, "outcome": out, "candidates": [*eps, out]}


@pytest.fixture
def ep(store):
    return write_episode(store, item_id="now")


def experience_fields(store, ep, candidates, k=3, seed="e-seed", **over):
    fields = selection.experience_fields(
        ep, candidates, blob(store, candidates), k=k, seed=seed, policy_id="uniform-k", policy_version="1",
        presentation_mode="contrast", presentation_format=FORMAT,
    )
    return {**fields, **over}


def write_experience(store, ep, candidates, ctx=None, **over):
    ctx = ctx or write_context(store, ep)
    return store.write_ctp_node("experience", experience_fields(store, ep, candidates, **over), producer=PRODUCER, input_nodes=(ctx,))


def attrs_of(store, node_id):
    return next(n for n in store.nodes() if n["node_id"] == node_id)["attrs"]


def ref_word(seed, counter):
    """The number of the stream, written out by hand (not through graph.rng)."""
    return int.from_bytes(hashlib.sha256(('{"counter":%d,"seed":"%s"}' % (counter, seed)).encode()).digest(), "big")


# ---- the random source ------------------------------------------------------------------------------------------
def test_the_words_are_sha256_of_the_jcs_of_seed_and_counter():
    assert rng.word("abc", 0) == int("679cb86daa4ad5c97d3a14ce4d9a42ad61972b93e67e378bbf73d9950b14ee6b", 16)
    assert rng.word("abc", 1) == int("bdf02e4bfe350a166d07a1b360f485df65663695d3a4a35cb1ccbea9bd4a38b7", 16)
    assert all(rng.word(s, c) == ref_word(s, c) for s in ("a", "é", "x" * 128) for c in (0, 1, 99))


def test_a_choice_is_the_first_accepted_word_modulo_m():
    m = 10
    s = rng.Stream("abc")
    assert s.below(m) == ref_word("abc", 0) % m and s.counter == 1
    assert rng.choose_ordered(["a", "b", "c", "d"], 4, "abc") == _reference_order(["a", "b", "c", "d"], "abc")


def _reference_order(items, seed):
    left, counter, out = list(items), 0, []
    while left:
        m = len(left)
        limit = (2**256 // m) * m
        while True:
            r = ref_word(seed, counter)
            counter += 1
            if r < limit:
                break
        out.append(left.pop(r % m))
    return out


def test_a_word_at_or_above_floor_2_256_over_m_times_m_is_rejected_and_counted(monkeypatch):
    m = 2**255 + 1  # floor(2**256 / m) == 1, so the limit is m itself
    words = iter([2**256 - 1, m, 7])
    monkeypatch.setattr(rng, "word", lambda seed, counter: next(words))
    s = rng.Stream("x")
    assert s.below(m) == 7 and s.counter == 3  # both rejected words were used up and counted
    monkeypatch.setattr(rng, "word", lambda seed, counter: m - 1)
    assert rng.Stream("x").below(m) == m - 1  # just below the limit is accepted


def test_the_counter_starts_at_zero_for_every_selection_and_the_draw_is_deterministic():
    assert rng.choose_ordered(list("abcde"), 3, "s") == rng.choose_ordered(list("abcde"), 3, "s")
    assert rng.choose_one(list("abcde"), "s") == list("abcde")[ref_word("s", 0) % 5]


@pytest.mark.parametrize("seed", ["", 7, None, "x" * 129, b"a"])
def test_a_seed_is_a_string_of_one_to_128_characters(seed):
    with pytest.raises(rng.RngError):
        rng.Stream(seed)


def test_k_must_fit_the_candidates():
    for k in (0, 6, True, 1.0, -1):
        with pytest.raises(rng.RngError):
            rng.choose_ordered(list("abcde"), k, "s")


def test_each_candidate_is_drawn_to_each_position_about_equally_often_over_many_seeds():
    cands, k, n_seeds = list("abcde"), 3, 6000
    positions = Counter()
    sets = Counter()
    for i in range(n_seeds):
        drawn = rng.choose_ordered(cands, k, f"seed-{i}")
        assert len(set(drawn)) == k
        positions.update((pos, c) for pos, c in enumerate(drawn))
        sets[frozenset(drawn)] += 1
    p = 1 / len(cands)
    tol = 5 * math.sqrt(n_seeds * p * (1 - p))  # five standard deviations: the seeds are fixed, so this is deterministic
    for pos in range(k):
        for c in cands:
            assert abs(positions[(pos, c)] - n_seeds * p) < tol, (pos, c, positions[(pos, c)])
    assert len(sets) == math.comb(5, 3)
    p = 1 / math.comb(5, 3)
    assert all(abs(v - n_seeds * p) < 5 * math.sqrt(n_seeds * p * (1 - p)) for v in sets.values())


def test_every_order_of_a_full_draw_and_every_condition_are_about_equally_likely():
    n_seeds = 12000
    orders = Counter(tuple(rng.choose_ordered(list("abcd"), 4, f"o{i}")) for i in range(n_seeds))
    assert len(orders) == 24
    p = 1 / 24
    assert all(abs(v - n_seeds * p) < 5 * math.sqrt(n_seeds * p * (1 - p)) for v in orders.values())
    conds = Counter(rng.choose_one(CONDITIONS, f"c{i}") for i in range(7000))
    p = 1 / 5
    assert set(conds) == set(CONDITIONS) and all(abs(v - 7000 * p) < 5 * math.sqrt(7000 * p * (1 - p)) for v in conds.values())


# ---- registration ------------------------------------------------------------------------------------------------
def test_the_two_types_are_selections_without_observed_at_and_the_edge_is_registered(store, ep, past):
    assert {"assignment", "experience"} <= set(ctp.CTP_NODE_TYPES) and "presented_to" in ctp.CTP_EDGE_TYPES and "presented_to" in EDGE_TYPES
    for node_type in ("assignment", "experience"):
        assert d70_class({"node_type": node_type, "kind": "anything"}) == "selection"
        with pytest.raises(Exception, match="kind must be"):
            check_stored_kind({"node_id": "x", "node_type": node_type, "kind": "observation"})
    a, e = write_assignment(store), write_experience(store, ep, past["candidates"])
    for node_id in (a, e):
        rec = next(n for n in store.nodes() if n["node_id"] == node_id)
        assert set(rec) == {"node_id", "node_type", "kind", "verification", "attrs"} and (rec["kind"], rec["verification"]) == ("selection", "unverified")
        assert "observed_at" not in rec["attrs"] and rec["attrs"]["schema"] == "ctp.datamodel/0.3"
    for node_type, fields, inputs in (("assignment", assignment_fields(store), ()), ("experience", experience_fields(store, ep, past["candidates"]), (write_context(store, ep),))):
        with pytest.raises(CtpError, match="observed_at"):
            store.build_ctp_node(node_type, fields, producer=PRODUCER, input_nodes=inputs, observed_at="2026-11-03T00:00:00Z")


# ---- assignment ---------------------------------------------------------------------------------------------------
def test_an_assignment_draws_one_condition_with_probability_one_over_m(store):
    node = write_assignment(store)
    a = attrs_of(store, node)
    assert a["condition"] == CONDITIONS[ref_word("a-seed", 0) % 5] == selection.byte_sorted(CONDITIONS)[ref_word("a-seed", 0) % 5]
    assert a["assignment_probability"] == {"num": "1", "den": "5"} and a["probability_source"] == "selector"
    assert a["assignment_algorithm"] == {"name": "uniform_within_stratum", "version": "1"} and a["assignment_rng"] == {"algorithm": "sha256_rejection_v1", "seed": "a-seed"}
    assert write_assignment(store) == node and node.startswith("assignment:")


def test_the_candidates_of_an_assignment_are_read_from_the_blob_in_byte_order(store):
    node = write_assignment(store, candidates=["E4", "E2", "E0"], seed="s1")  # given unsorted: the blob is sorted
    assert attrs_of(store, node)["assignment_probability"] == {"num": "1", "den": "3"}
    assert json.loads((store.blob_dir / "blobs" / attrs_of(store, node)["assignment_candidate_set_hash"][:2] / attrs_of(store, node)["assignment_candidate_set_hash"][2:4] / attrs_of(store, node)["assignment_candidate_set_hash"]).read_bytes()) == ["E0", "E2", "E4"]


def other_condition(seed="a-seed"):
    return next(c for c in CONDITIONS if c != CONDITIONS[ref_word(seed, 0) % 5])


@pytest.mark.parametrize(
    "bad, match",
    [
        ({"condition": "WRONG"}, "recomputation: the condition"),
        ({"assignment_probability": {"num": "1", "den": "4"}}, "recomputation: assignment_probability"),
        ({"assignment_probability": {"num": "2", "den": "10"}}, "irreducible"),
        ({"assignment_probability": {"num": "0", "den": "1"}}, "recomputation: assignment_probability"),
        ({"assignment_probability": 0.2}, "floating-point"),
        ({"assignment_probability": {"num": 1.0, "den": 5}}, "floating-point"),
        ({"assignment_probability": "1/5"}, "fraction"),
        ({"assignment_probability": {"num": 1, "den": 5}}, "decimal strings"),
        ({"assignment_probability": {"num": "01", "den": "5"}}, "decimal strings"),
        ({"assignment_probability": {"num": "1", "den": "05"}}, "decimal strings"),
        ({"assignment_probability": {"num": "-1", "den": "5"}}, "decimal strings"),
        ({"assignment_probability": {"num": "1", "den": "0"}}, "den must not be 0"),
        ({"assignment_policy_fingerprint": "f" * 64}, "recomputation: assignment_policy_fingerprint"),
        ({"assignment_policy_params": 3}, "assignment_policy_params must be an object"),
        ({"assignment_algorithm": {"name": "uniform_within_stratum", "version": "2"}}, "assignment_algorithm must be"),
        ({"assignment_algorithm": {"name": "something_else", "version": "1"}}, "assignment_algorithm must be"),
        ({"assignment_rng": {"algorithm": "random.Random", "seed": "a-seed"}}, "sha256_rejection_v1"),
        ({"assignment_rng": {"algorithm": "sha256_rejection_v1", "seed": 5}}, "seed"),
        ({"assignment_rng": {"algorithm": "sha256_rejection_v1", "seed": "x" * 129}}, "seed"),
        ({"probability_source": "unobserved"}, "probability_source"),
        ({"probability_source": "llm_self_report"}, "probability_source"),
        ({"probability_source": "single_candidate"}, "only for a candidate set of one"),
        ({"assignment_candidates_ref": "blob:" + "9" * 64, "assignment_candidate_set_hash": "9" * 64}, "is not in the blob location"),
        ({"assignment_candidate_set_hash": "9" * 64}, "candidate set hash is not the sha256"),
        ({"assignment_candidates_ref": "nope"}, "must be 'blob:"),
        ({"replicate": -1}, "replicate"),
        ({"experiment_spec_hash": "sha256:" + "8" * 64}, "HEXT source document"),
    ],
)
def test_an_assignment_that_does_not_match_its_recomputation_is_refused(store, bad, match):
    with pytest.raises(GraphError, match=match):
        write_assignment(store, **bad)
    assert not [n for n in store.nodes() if n["node_type"] == "assignment"]


def test_an_assignment_with_a_different_seed_draws_again_and_a_wrong_condition_for_that_seed_is_refused(store):
    seed = next(s for s in (f"x{i}" for i in range(50)) if CONDITIONS[ref_word(s, 0) % 5] != CONDITIONS[ref_word("a-seed", 0) % 5])
    write_assignment(store, seed=seed)
    with pytest.raises(CtpError, match="recomputation"):
        write_assignment(store, seed=seed, condition=other_condition(seed))


def test_the_stratum_is_a_label_that_does_not_enter_the_draw_but_does_enter_the_node_id(store):
    a, b = write_assignment(store, assignment_stratum="b1"), write_assignment(store, assignment_stratum="b2")
    assert attrs_of(store, a)["condition"] == attrs_of(store, b)["condition"] and a != b


def test_a_missing_or_unsorted_or_duplicated_candidate_blob_is_refused(store, tmp_path):
    fields = assignment_fields(store)
    for data in (b'["E1","E0"]', b'["E0","E0"]', b"[]", b"[1]", b"not json", b'{"a":1}'):
        sha = write_blob(tmp_path, data)
        with pytest.raises(CtpError, match="candidates"):
            store.write_ctp_node("assignment", {**fields, "assignment_candidates_ref": f"blob:{sha}", "assignment_candidate_set_hash": sha}, producer=PRODUCER)
    bare = GraphStore(tmp_path / "bare", v01=False)  # no blob location
    bare.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    with pytest.raises(CtpError, match="no blob location"):
        bare.write_ctp_node("assignment", fields, producer=PRODUCER)


def test_single_candidate_is_allowed_only_when_the_blob_holds_one_candidate(store):
    node = write_assignment(store, candidates=["E0"], probability_source="single_candidate")
    a = attrs_of(store, node)
    assert a["condition"] == "E0" and a["assignment_probability"] == {"num": "1", "den": "1"}
    assert write_assignment(store, candidates=["E0"], seed="other")  # selector is fine for one candidate too
    with pytest.raises(CtpError, match="only for a candidate set of one"):
        write_assignment(store, candidates=["E0", "E1"], probability_source="single_candidate")


def test_an_assignment_has_no_input_node_and_its_input_facts_are_exactly_its_definition(store):
    node = write_assignment(store)
    a = attrs_of(store, node)
    assert a["input_nodes"] == [] and a["input_facts"] == {"definition": selection.definition("assignment", a)}
    with pytest.raises(CtpError, match="definition"):
        store.build_ctp_node("assignment", assignment_fields(store), producer=PRODUCER, input_facts={"x": 1})
    fresh = store.build_ctp_node("assignment", assignment_fields(store, seed="fresh"), producer=PRODUCER)  # built, not stored: a stored id would be a collision first
    forged = {**fresh["attrs"], "input_facts": {"definition": {**fresh["attrs"]["input_facts"]["definition"], "assignment_stratum": "other"}}}
    with pytest.raises(CtpError, match="exactly the selection's definition"):
        store.add_node(fresh["node_id"], "assignment", kind="selection", verification="unverified", attrs=forged)
    ep = write_episode(store)
    with pytest.raises(CtpError, match="no input node"):
        store.write_ctp_node("assignment", assignment_fields(store), producer=PRODUCER, input_nodes=(ep,))


def test_a_changed_definition_is_another_selection_event_with_the_same_identity_fields(store):
    a = write_assignment(store)
    b = write_assignment(store, assignment_policy_version="2")
    c = write_assignment(store, candidates=CONDITIONS + ["E5"])
    assert len({a, b, c}) == 3 and attrs_of(store, a)["experiment_spec_hash"] == attrs_of(store, c)["experiment_spec_hash"]


def test_an_assignment_selects_the_episode_that_carries_its_condition(store):
    node = write_assignment(store)
    cond = attrs_of(store, node)["condition"]
    ok = write_episode(store, condition=cond)
    assert store.add_edge("selects", node, ok) is True and store.add_edge("selects", node, ok) is False
    wrong_cond = write_episode(store, condition=other_condition())
    with pytest.raises(CtpError, match="condition"):
        store.add_edge("selects", node, wrong_cond)
    with pytest.raises(CtpError, match="item_id"):
        store.add_edge("selects", node, write_episode(store, condition=cond, item_id="other"))
    with pytest.raises(CtpError, match="replicate"):
        store.add_edge("selects", node, write_episode(store, condition=cond, replicate=1))
    ctx = write_context(store, ok)
    with pytest.raises(CtpError, match="selects the episode"):
        store.add_edge("selects", node, ctx)
    with pytest.raises(CtpError, match="governance record selects"):
        store.add_edge("selects", ok, node)


# ---- experience ---------------------------------------------------------------------------------------------------
def test_an_experience_draws_k_in_order_without_replacement_and_records_the_probabilities(store, ep, past):
    node = write_experience(store, ep, past["candidates"])
    a = attrs_of(store, node)
    cands = selection.byte_sorted(past["candidates"])
    assert a["selected_ids"] == _reference_order(cands, "e-seed")[:3] == rng.choose_ordered(cands, 3, "e-seed")
    assert len(set(a["selected_ids"])) == 3 and set(a["selected_ids"]) <= set(cands) and "order" not in a
    n, k = 5, 3
    assert a["conditional_selection_probabilities"] == [{"num": "1", "den": "5"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]
    assert a["slate_probability"] == {"set": {"num": "1", "den": str(math.comb(n, k))}, "ordered": {"num": "1", "den": "60"}}
    assert a["selection_algorithm"] == {"name": "uniform_ordered_k_subset_without_replacement", "version": "1"}
    assert (a["replacement"], a["order_considered"], a["ordering_rule"], a["retrieval_space"], a["probability_source"]) == (False, True, "node_id_bytewise_ascending", "R-text", "selector")
    assert a["arm_id"].startswith("arm:") and node.startswith("experience:") and write_experience(store, ep, past["candidates"]) == node


def test_the_ordered_probability_is_the_set_probability_over_k_factorial(store, ep, past):
    for k in (1, 2, 3, 4, 5):
        a = attrs_of(store, write_experience(store, ep, past["candidates"], k=k, seed=f"k{k}"))
        n = 5
        assert a["slate_probability"]["set"]["den"] == str(math.comb(n, k))
        assert a["slate_probability"]["ordered"]["den"] == str(math.factorial(n) // math.factorial(n - k))
        assert math.gcd(int(a["slate_probability"]["ordered"]["num"]), int(a["slate_probability"]["ordered"]["den"])) == 1
        assert int(a["slate_probability"]["ordered"]["den"]) == int(a["slate_probability"]["set"]["den"]) * math.factorial(k)
        assert [p["den"] for p in a["conditional_selection_probabilities"]] == [str(n - i) for i in range(k)]


def tampered(store, ep, past, seed="e-seed", **over):
    """The fields of a valid draw, then ``over`` written on top of them (so k, rng and the rest can be made wrong).

    The policy fingerprint follows the overridden settings unless the test gives one: it is checked on its own below.
    """
    fields = {**experience_fields(store, ep, past["candidates"], k=min(3, len(past["candidates"])), seed=seed), **over}
    if "selection_policy_fingerprint" not in over:
        fields["selection_policy_fingerprint"] = selection.selection_fingerprint(fields)
    return store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(write_context(store, ep),))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda a: {"selected_ids": list(reversed(a["selected_ids"]))}, "recomputation: the 3 drawn"),
        (lambda a: {"selected_ids": sorted(a["selected_ids"])}, "recomputation: the 3 drawn"),
        (lambda a: {"selected_ids": a["selected_ids"][:2]}, "recomputation: the 3 drawn"),
        (lambda a: {"selected_ids": ["nope", *a["selected_ids"][1:]]}, "recomputation: the 3 drawn"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "1", "den": "5"}, {"num": "1", "den": "4"}, {"num": "1", "den": "2"}]}, "recomputation: conditional"),
        (lambda a: {"conditional_selection_probabilities": a["conditional_selection_probabilities"][:2]}, "recomputation: conditional"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "2", "den": "10"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "irreducible"),
        (lambda a: {"conditional_selection_probabilities": [0.2, 0.25, 0.3333]}, "floating-point"),
        (lambda a: {"slate_probability": {"set": {"num": "1", "den": "10"}, "ordered": {"num": "1", "den": "59"}}}, "recomputation: slate_probability"),
        (lambda a: {"slate_probability": {"set": {"num": "1", "den": "60"}, "ordered": {"num": "1", "den": "60"}}}, "recomputation: slate_probability"),
        (lambda a: {"slate_probability": {"set": {"num": "2", "den": "20"}, "ordered": {"num": "1", "den": "60"}}}, "irreducible"),
        (lambda a: {"slate_probability": {"set": 0.1, "ordered": 0.0166}}, "floating-point"),
        (lambda a: {"slate_probability": {"set": {"num": "1", "den": "10"}}}, "slate_probability must be"),
        (lambda a: {"arm_id": "arm:" + "0" * 64}, "arm_id"),
        (lambda a: {"presentation_mode": "violation_only"}, "arm_id"),
        (lambda a: {"presentation_mode": "everything"}, "presentation_mode"),
        (lambda a: {"presentation_format": {"template_id": "slate-v1"}}, "presentation_format"),
        (lambda a: {"presentation_format": {"template_id": "slate-v1", "version": 1}}, "presentation_format"),
        (lambda a: {"k": 0}, "k must be"),
        (lambda a: {"k": 6}, "k must be"),
        (lambda a: {"k": True}, "k must be"),
        (lambda a: {"k": 3.0}, "floating-point"),
        (lambda a: {"replacement": True}, "replacement must be false"),
        (lambda a: {"order_considered": False}, "order_considered must be true"),
        (lambda a: {"ordering_rule": "as_given"}, "ordering_rule"),
        (lambda a: {"selection_algorithm": {"name": "uniform_k_subset_without_replacement", "version": "1"}}, "selection_algorithm must be"),
        (lambda a: {"selection_algorithm": {"name": "uniform_ordered_k_subset_without_replacement", "version": "2"}}, "selection_algorithm must be"),
        (lambda a: {"rng": {"algorithm": "mt19937", "seed": "e-seed"}}, "sha256_rejection_v1"),
        (lambda a: {"rng": {"algorithm": "sha256_rejection_v1", "seed": 1234}}, "seed"),
        (lambda a: {"rng": {"algorithm": "sha256_rejection_v1", "seed": "s" * 129}}, "seed"),
        (lambda a: {"rng": {"algorithm": "sha256_rejection_v1", "seed": "other-seed"}}, "recomputation"),
        (lambda a: {"probability_source": "unobserved"}, "probability_source"),
        (lambda a: {"probability_source": "single_candidate"}, "only for a candidate set of one"),
        (lambda a: {"retrieval_space": "R-vector"}, "retrieval_space"),
        (lambda a: {"selection_policy_fingerprint": ""}, "selection_policy_fingerprint"),
        (lambda a: {"selection_policy_fingerprint": "f" * 64}, "recomputation: selection_policy_fingerprint"),
        (lambda a: {"selection_policy_params": {"pool": "recent"}, "selection_policy_fingerprint": a["selection_policy_fingerprint"]}, "recomputation: selection_policy_fingerprint"),  # the params changed, the fingerprint did not
        (lambda a: {"selection_policy_params": [1]}, "selection_policy_params must be an object"),
        (lambda a: {"selection_policy_params": {"x": 0.5}}, "floating-point"),
        (lambda a: {"conditional_selection_probabilities": [{"num": 1, "den": 5}, {"num": 1, "den": 4}, {"num": 1, "den": 3}]}, "decimal strings"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "01", "den": "5"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "decimal strings"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "+1", "den": "5"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "decimal strings"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "-1", "den": "5"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "decimal strings"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "1", "den": "0"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "den must not be 0"),
        (lambda a: {"conditional_selection_probabilities": [{"num": "2", "den": "10"}, {"num": "1", "den": "4"}, {"num": "1", "den": "3"}]}, "irreducible"),
        (lambda a: {"slate_probability": {"set": {"num": 1, "den": 10}, "ordered": {"num": 1, "den": 60}}}, "decimal strings"),
        (lambda a: {"slate_probability": {"set": {"num": "1", "den": "10"}, "ordered": {"num": "01", "den": "60"}}}, "decimal strings"),
        (lambda a: {"slate_probability": {"set": {"num": "3", "den": "30"}, "ordered": {"num": "1", "den": "60"}}}, "irreducible"),
        (lambda a: {"candidates_ref": "blob:" + "9" * 64, "candidate_set_hash": "9" * 64}, "is not in the blob location"),
    ],
)
def test_an_experience_that_differs_from_its_recomputation_is_refused(store, ep, past, mutate, match):
    good = experience_fields(store, ep, past["candidates"])
    # the "other-seed" case must really draw something else, or the record would be valid
    over = mutate(good)
    if "rng" in over and over["rng"].get("seed") == "other-seed":
        assert rng.choose_ordered(selection.byte_sorted(past["candidates"]), 3, "other-seed") != good["selected_ids"]
    with pytest.raises(GraphError, match=match):
        tampered(store, ep, past, **over)
    assert not [n for n in store.nodes() if n["node_type"] == "experience"]


def test_r_traj_is_refused_with_its_reason_code_and_the_other_spaces_pass(store, ep, past):
    with pytest.raises(CtpError, match="PREFILL_TRAJECTORY_NOT_AVAILABLE"):
        tampered(store, ep, past, retrieval_space="R-traj")
    nodes = {space: write_experience(store, ep, past["candidates"], retrieval_space=space) for space in ("R-text", "R-struct")}
    assert len(set(nodes.values())) == 2  # the space is part of the definition, hence of the node_id
    assert selection.PREFILL_TRAJECTORY_NOT_AVAILABLE == "PREFILL_TRAJECTORY_NOT_AVAILABLE"


def test_a_missing_or_malformed_candidates_blob_or_no_blob_location_is_refused(store, ep, past, tmp_path):
    ctx = write_context(store, ep)
    fields = experience_fields(store, ep, past["candidates"])
    for data in (b'["b","a","c"]', b'["a","a","b"]', b"[]", b'[1,2]', b"nonsense"):
        sha = write_blob(tmp_path, data)
        with pytest.raises(CtpError, match="candidates"):
            store.write_ctp_node("experience", {**fields, "candidates_ref": f"blob:{sha}", "candidate_set_hash": sha}, producer=PRODUCER, input_nodes=(ctx,))
    assert ctx and not [n for n in store.nodes() if n["node_type"] == "experience"]
    bare = GraphStore(tmp_path / "bare", v01=False)
    bare.add_node(f"hextdoc:{SPEC_HASH}", "hext_document", attrs={"hext_kind": "source", "state_hash": SPEC_HASH})
    bare_ep = write_episode(bare, item_id="now")
    bare_ctx = write_context(bare, bare_ep)
    with pytest.raises(CtpError, match="no blob location"):
        bare.write_ctp_node("experience", {**fields, "episode_id": bare_ep}, producer=PRODUCER, input_nodes=(bare_ctx,))


def test_the_candidate_blob_must_be_the_one_the_hash_names(store, ep, past, tmp_path):
    fields = experience_fields(store, ep, past["candidates"])
    sha = fields["candidate_set_hash"]
    (tmp_path / "blobs" / sha[:2] / sha[2:4] / sha).write_bytes(b'["tampered"]')
    with pytest.raises(CtpError, match="does not hash to its name"):
        store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(write_context(store, ep),))


def test_a_single_candidate_experience(store, ep, past):
    one = [past["episodes"][0]]
    node = write_experience(store, ep, one, k=1, probability_source="single_candidate")
    a = attrs_of(store, node)
    assert a["selected_ids"] == one and a["slate_probability"] == {"set": {"num": "1", "den": "1"}, "ordered": {"num": "1", "den": "1"}}
    assert a["conditional_selection_probabilities"] == [{"num": "1", "den": "1"}]
    with pytest.raises(CtpError, match="only for a candidate set of one"):
        write_experience(store, ep, past["candidates"], probability_source="single_candidate")
    with pytest.raises(CtpError, match="k must be"):
        tampered(store, ep, {"candidates": one}, k=2)


# ---- policy params and fingerprints; large n and k -------------------------------------------------------------------
def test_params_that_differ_alone_make_another_fingerprint_another_input_hash_and_another_node_id(store, ep, past):
    base = write_experience(store, ep, past["candidates"])
    ctx = write_context(store, ep)
    ids = {}
    for name, params in (("none", {}), ("a", {"recency": "7"}), ("b", {"recency": "8"}), ("nested", {"w": {"x": ["1", "2"]}})):
        fields = selection.experience_fields(ep, past["candidates"], blob(store, past["candidates"]), k=3, seed="e-seed", policy_id="uniform-k", policy_version="1", policy_params=params, presentation_mode="contrast", presentation_format=FORMAT)
        ids[name] = store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(ctx,))
        got = attrs_of(store, ids[name])
        assert got["selection_policy_params"] == params
        assert got["selection_policy_fingerprint"] == hashlib.sha256(canonical_bytes({f: got[f] for f in ("selection_algorithm", "k", "replacement", "order_considered", "ordering_rule", "selection_policy_params")})).hexdigest()
    assert ids["none"] == base and len(set(ids.values())) == 4
    hashes = {attrs_of(store, i)["input_hash"] for i in ids.values()}
    fps = {attrs_of(store, i)["selection_policy_fingerprint"] for i in ids.values()}
    assert len(hashes) == 4 and len(fps) == 4
    assert len({tuple(attrs_of(store, i)["selected_ids"]) for i in ids.values()}) == 1  # the draw is the same: only the params differ


def test_assignment_params_and_fingerprint(store):
    ids = {}
    for name, params in (("none", {}), ("a", {"w": "1"}), ("b", {"w": "2"})):
        fields = selection.assignment_fields(SPEC_HASH, "i1", 0, CONDITIONS, blob(store, CONDITIONS), seed="a-seed", stratum="b1", policy_id="uniform", policy_version="1", policy_params=params)
        ids[name] = store.write_ctp_node("assignment", fields, producer=PRODUCER)
        got = attrs_of(store, ids[name])
        assert got["assignment_policy_fingerprint"] == hashlib.sha256(canonical_bytes({f: got[f] for f in ("assignment_algorithm", "assignment_stratum", "assignment_policy_params")})).hexdigest()
    assert ids["none"] == write_assignment(store) and len(set(ids.values())) == 3
    assert len({attrs_of(store, i)["assignment_policy_fingerprint"] for i in ids.values()}) == 3
    assert len({attrs_of(store, i)["condition"] for i in ids.values()}) == 1  # params and stratum never enter the draw
    stratum = write_assignment(store, assignment_stratum="b2")  # a changed stratum changes the fingerprint too
    assert attrs_of(store, stratum)["assignment_policy_fingerprint"] != attrs_of(store, ids["none"])["assignment_policy_fingerprint"]


@pytest.mark.parametrize("n, k", [(100, 48), (200, 12), (1000, 7)])
def test_a_large_selection_is_stored_and_matches_its_recomputation(store, ep, n, k):
    cands = [f"episode:{i:04d}" for i in range(n)]
    node = write_experience(store, ep, cands, k=k, seed=f"big-{n}-{k}")
    a = attrs_of(store, node)
    assert a["selected_ids"] == _reference_order(selection.byte_sorted(cands), f"big-{n}-{k}")[:k] and len(set(a["selected_ids"])) == k
    assert a["slate_probability"]["set"] == {"num": "1", "den": str(math.comb(n, k))}
    assert a["slate_probability"]["ordered"] == {"num": "1", "den": str(math.factorial(n) // math.factorial(n - k))}
    assert int(a["slate_probability"]["ordered"]["den"]) > 2**53  # a bare integer would not have been storable
    assert [p["den"] for p in a["conditional_selection_probabilities"]] == [str(n - i) for i in range(k)]
    assert write_experience(store, ep, cands, k=k, seed=f"big-{n}-{k}") == node  # idempotent
    wrong = {**experience_fields(store, ep, cands, k=k, seed=f"big-{n}-{k}"), "slate_probability": {"set": {"num": "1", "den": str(math.comb(n, k) + 1)}, "ordered": a["slate_probability"]["ordered"]}}
    with pytest.raises(CtpError, match="recomputation: slate_probability"):
        store.write_ctp_node("experience", wrong, producer=PRODUCER, input_nodes=(write_context(store, ep),))


def test_params_changed_without_the_fingerprint_are_refused_for_an_assignment(store):
    good = assignment_fields(store)
    with pytest.raises(CtpError, match="recomputation: assignment_policy_fingerprint"):
        store.write_ctp_node("assignment", {**good, "assignment_policy_params": {"pool": "x"}}, producer=PRODUCER)
    with pytest.raises(CtpError, match="recomputation: assignment_policy_fingerprint"):
        store.write_ctp_node("assignment", {**good, "assignment_stratum": "b9"}, producer=PRODUCER)  # the stratum is in the material too


def test_an_integer_fraction_is_not_a_probability_even_when_it_is_small(store):
    with pytest.raises(CtpError, match="decimal strings"):
        write_assignment(store, assignment_probability={"num": 1, "den": 5})


def test_the_arm_is_the_presented_content_and_the_node_is_the_selection_event(store, past):
    cands, k = selection.byte_sorted(past["candidates"])[:3], 2  # n=3, k=2: six possible slates, so seeds that agree are easy to find
    by_slate = {}
    for i in range(60):
        by_slate.setdefault(tuple(rng.choose_ordered(cands, k, f"w{i}")), []).append(f"w{i}")
    slate, seeds = next((s, v) for s, v in by_slate.items() if len(v) >= 2)
    eps = [write_episode(store, item_id=f"arm{i}") for i in range(2)]
    nodes = []
    for e, seed, policy in zip(eps, seeds[:2], ("policy-a", "policy-b"), strict=True):
        fields = selection.experience_fields(e, cands, blob(store, cands), k=k, seed=seed, policy_id=policy, policy_version="1",
                                             presentation_mode="contrast", presentation_format=FORMAT)
        nodes.append(store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(write_context(store, e),)))
    a, b = (attrs_of(store, n) for n in nodes)
    assert nodes[0] != nodes[1]  # another episode, another policy, another rng seed: another selection event
    assert tuple(a["selected_ids"]) == tuple(b["selected_ids"]) == slate and a["arm_id"] == b["arm_id"]
    assert a["arm_id"] == "arm:" + hashlib.sha256(canonical_bytes({"selected_ids": list(slate), "presentation_mode": "contrast", "presentation_format": FORMAT})).hexdigest()
    # the arm moves with the presented content only: mode, format and the order of the ids
    assert selection.arm_id(slate, "unrelated", FORMAT) != a["arm_id"]
    assert selection.arm_id(slate, "contrast", {"template_id": "slate-v2", "version": "1"}) != a["arm_id"]
    assert selection.arm_id(tuple(reversed(slate)), "contrast", FORMAT) != a["arm_id"]


def test_an_experience_needs_the_context_observation_of_its_own_episode(store, ep, past):
    fields = experience_fields(store, ep, past["candidates"])
    other = write_episode(store, item_id="elsewhere")
    with pytest.raises(CtpError, match="exactly one input node"):
        store.write_ctp_node("experience", fields, producer=PRODUCER)  # no input node
    with pytest.raises(CtpError, match="exactly one input node"):
        store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(write_context(store, other),))  # another episode's context
    with pytest.raises(CtpError, match="exactly one input node"):
        store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(ep,))  # not a context_observation
    with pytest.raises(CtpError, match="is not a stored node|is not stored"):
        store.write_ctp_node("experience", fields, producer=PRODUCER, input_nodes=("context_observation:" + "0" * 64,))
    with pytest.raises(CtpError, match="not a stored episode"):
        store.write_ctp_node("experience", {**fields, "episode_id": "episode:" + "0" * 64}, producer=PRODUCER, input_nodes=(write_context(store, ep),))
    with pytest.raises(CtpError, match="definition"):
        store.build_ctp_node("experience", fields, producer=PRODUCER, input_nodes=(write_context(store, ep),), input_facts={"queried_with": ["t"]})  # no queried_with, no free facts


def test_the_input_hash_covers_the_whole_definition_and_the_context_record(store, ep, past):
    base = write_experience(store, ep, past["candidates"])
    assert attrs_of(store, base)["input_facts"] == {"definition": selection.definition("experience", attrs_of(store, base))}
    assert set(attrs_of(store, base)["input_facts"]["definition"]) == set(selection.EXPERIENCE_DEFINITION)
    variants = {
        "policy version": dict(selection_policy_version="2"),
        "policy id": dict(selection_policy_id="uniform-other"),
        "retrieval space": dict(retrieval_space="R-struct"),
    }
    ids = {name: write_experience(store, ep, past["candidates"], **over) for name, over in variants.items()}
    assert len({base, *ids.values()}) == 4
    ids["k"] = write_experience(store, ep, past["candidates"], k=2)
    with_params = selection.experience_fields(ep, past["candidates"], blob(store, past["candidates"]), k=3, seed="e-seed", policy_id="uniform-k", policy_version="1",
                                              policy_params={"temperature": "1"}, presentation_mode="contrast", presentation_format=FORMAT)
    ids["params"] = store.write_ctp_node("experience", with_params, producer=PRODUCER, input_nodes=(write_context(store, ep),))
    ids["candidates"] = write_experience(store, ep, past["candidates"][:4], k=3)
    ids["seed"] = write_experience(store, ep, past["candidates"], seed="another")
    assert len({base, *ids.values()}) == 8
    fresh = store.build_ctp_node("experience", experience_fields(store, ep, past["candidates"], seed="fresh"), producer=PRODUCER, input_nodes=(write_context(store, ep),))  # built, not stored
    forged = {**fresh["attrs"], "input_facts": {"definition": {**fresh["attrs"]["input_facts"]["definition"], "k": 2}}}
    with pytest.raises(CtpError, match="exactly the selection's definition"):
        store.add_node(fresh["node_id"], "experience", kind="selection", verification="unverified", attrs=forged)
    forged_arm = {**fresh["attrs"], "arm_id": "arm:" + "1" * 64}
    with pytest.raises(CtpError, match="arm_id"):
        store.add_node(fresh["node_id"], "experience", kind="selection", verification="unverified", attrs=forged_arm)


def test_floats_and_out_of_range_integers_are_refused_in_both_types(store, ep, past):
    ctx = write_context(store, ep)
    a = store.build_ctp_node("assignment", assignment_fields(store), producer=PRODUCER)
    e = store.build_ctp_node("experience", experience_fields(store, ep, past["candidates"]), producer=PRODUCER, input_nodes=(ctx,))
    for node in (a, e):
        for over in ({"refs": {"score": 0.5}}, {"refs": {"n": 2**60}}, {"refs": {"deep": [{"x": -(2**53)}]}}):
            with pytest.raises(CtpError, match="floating-point|outside"):
                store.add_node(node["node_id"], node["node_type"], kind="selection", verification="unverified", attrs={**node["attrs"], **over})
        assert store.add_node(node["node_id"], node["node_type"], kind="selection", verification="unverified", attrs=node["attrs"])


# ---- edges ---------------------------------------------------------------------------------------------------------
def test_an_experience_selects_what_it_drew_from_another_episode_and_never_its_own(store, ep, past):
    node = write_experience(store, ep, past["candidates"], k=5)
    drawn = attrs_of(store, node)["selected_ids"]
    for target in past["candidates"]:
        assert store.add_edge("selects", node, target) is True and store.add_edge("selects", node, target) is False
    assert set(drawn) == set(past["candidates"])
    own_outcome = store.write_ctp_node("outcome", {"episode_id": ep, "carry": {}, "constraint_compliance": "PASS", "two_axis": "CARRIED_COMPLIED"}, producer=PRODUCER)
    own_cands = [*past["candidates"], ep, own_outcome]
    mine = write_experience(store, ep, own_cands, k=7, seed="own")
    assert set(attrs_of(store, mine)["selected_ids"]) == set(own_cands)
    with pytest.raises(CtpError, match="its own episode"):
        store.add_edge("selects", mine, ep)
    with pytest.raises(CtpError, match="its own episode"):
        store.add_edge("selects", mine, own_outcome)
    assert store.add_edge("selects", mine, past["outcome"]) is True


def test_an_experience_selects_only_episodes_or_outcomes_that_it_drew(store, ep, past):
    node = write_experience(store, ep, past["candidates"], k=2)
    drawn = set(attrs_of(store, node)["selected_ids"])
    not_drawn = next(c for c in past["candidates"] if c not in drawn)
    with pytest.raises(CtpError, match="did not draw"):
        store.add_edge("selects", node, not_drawn)
    ctx = write_context(store, ep)
    with pytest.raises(CtpError, match="earlier episode or outcome"):
        store.add_edge("selects", node, ctx)
    step = write_step(store, ep)
    with pytest.raises(CtpError, match="earlier episode or outcome"):
        store.add_edge("selects", node, step)


def test_presented_to_goes_to_the_f2_step_of_the_same_episode(store, ep, past):
    node = write_experience(store, ep, past["candidates"])
    f2, f1, f3 = (write_step(store, ep, s) for s in ("f2", "f1", "f3"))
    assert store.add_edge("presented_to", node, f2) is True and store.add_edge("presented_to", node, f2) is False
    for wrong in (f1, f3):
        with pytest.raises(CtpError, match="f2 step_output"):
            store.add_edge("presented_to", node, wrong)
    other = write_episode(store, item_id="elsewhere")
    with pytest.raises(CtpError, match="another episode"):
        store.add_edge("presented_to", node, write_step(store, other))
    with pytest.raises(CtpError, match="f2 step_output"):
        store.add_edge("presented_to", f2, node)
    with pytest.raises(CtpError, match="f2 step_output"):
        store.add_edge("presented_to", write_assignment(store), f2)


def test_presented_to_needs_the_blob_of_the_stage_input_to_be_there(store, ep, past, tmp_path):
    node = write_experience(store, ep, past["candidates"])
    f2 = write_step(store, ep)
    sha = attrs_of(store, f2)["input_context_sha256"]
    (tmp_path / "blobs" / sha[:2] / sha[2:4] / sha).unlink()  # the stage input went missing after the step was written
    with pytest.raises(CtpError, match="is not in the blob location"):
        store.add_edge("presented_to", node, f2)


def test_a_whole_slate_chain_with_an_assignment(store, past):
    assignment = write_assignment(store)
    ep = write_episode(store, condition=attrs_of(store, assignment)["condition"], item_id="i1")
    assert store.add_edge("selects", assignment, ep)
    exp = write_experience(store, ep, past["candidates"], k=2)
    f2 = write_step(store, ep)
    assert store.add_edge("presented_to", exp, f2)
    for target in attrs_of(store, exp)["selected_ids"]:
        assert store.add_edge("selects", exp, target)
    assert store.nondeterminism() == [] and store.axis_consistency().ok


# ---- determinism ----------------------------------------------------------------------------------------------------
CHILD = r"""
import json, sys
sys.path.insert(0, {root!r})
from pathlib import Path
from graph import selection
from graph.blobs import write_blob
from graph.store import GraphStore
root = Path(sys.argv[1])
s = GraphStore(root, v01=False, blob_dir=root)
spec = "sha256:" + "5" * 64
s.add_node("hextdoc:" + spec, "hext_document", attrs={{"hext_kind": "source", "state_hash": spec}})
P = {{"component": "s", "version": "1", "config_sha256": "c" * 64}}
def ep(i, cond):
    return s.write_ctp_node("episode", {{"experiment_spec_hash": spec, "item_id": i, "bundle_id": "b", "split": "fit", "condition": cond, "replicate": 0, "rng_seed": 1}}, producer=P)
conds = ["E0", "E1", "E2"]
csha = write_blob(root, selection.candidates_blob(conds))
a = s.write_ctp_node("assignment", selection.assignment_fields(spec, "i1", 0, conds, csha, seed="child-seed", stratum="b", policy_id="u", policy_version="1"), producer=P)
cond = next(n for n in s.nodes() if n["node_id"] == a)["attrs"]["condition"]
e = ep("i1", cond)
past = [ep(f"p{{i}}", "E0") for i in range(4)]
ssha = write_blob(root, selection.candidates_blob(past))
ctx = s.write_ctp_node("context_observation", {{"episode_id": e, "context_ref": "blob:" + "1" * 64, "knowledge_ids": [], "constraint_annotation_ids": [], "task_ref": "t", "contains_experience": False}}, producer=P)
x = s.write_ctp_node("experience", selection.experience_fields(e, past, ssha, k=3, seed="child-seed", policy_id="u", policy_version="1", presentation_mode="contrast", presentation_format={{"template_id": "t", "version": "1"}}), producer=P, input_nodes=(ctx,))
s.add_edge("selects", a, e)
print(json.dumps({{"ids": [a, e, x], "nodes": (root / "graph" / "nodes.jsonl").read_text(), "edges": (root / "graph" / "edges.jsonl").read_text()}}))
"""


def test_ids_and_logs_are_identical_in_two_new_processes(tmp_path):
    outs = []
    for i, seed in enumerate(("1", "2")):
        done = subprocess.run([sys.executable, "-c", CHILD.format(root=str(ROOT)), str(tmp_path / f"p{i}")], capture_output=True, text=True, check=True, env={**os.environ, "PYTHONHASHSEED": seed})
        outs.append(json.loads(done.stdout))
    assert outs[0] == outs[1] and outs[0]["ids"][0].startswith("assignment:") and outs[0]["ids"][2].startswith("experience:")


def test_candidates_are_ordered_by_their_utf8_bytes():
    items = ["b", "\u00e9", "a", "Z", "\U0001f600", "\uffff"]
    assert selection.byte_sorted(items) == ["Z", "a", "b", "\u00e9", "\uffff", "\U0001f600"]  # not by code point: U+FFFF < U+1F600 in both, but UTF-16 order would differ


# ---- regression: the real data dirs ------------------------------------------------------------------------------
EV = Path.home() / "vaults" / "20260124" / "experiment-evidence" / "GRAPH-ALIGN001"
REAL = [EV / "graph_data", EV / "graph_data_v2", Path("/Volumes/SSD1TB/graph002_work/data"),
        Path("/Volumes/SSD1TB/graph002_work/data_samecontent"), Path("/Volumes/SSD1TB/d71_work/hekb_lib")]


@pytest.mark.parametrize("src", REAL, ids=lambda p: p.name)
def test_real_data_rewrites_byte_identically_with_the_second_half_registered(tmp_path, src):
    """Copies are written in tmp_path; the originals are only read. The projection layer's own log is read and re-checked."""
    if not (src / "graph" / "nodes.jsonl").exists():
        pytest.skip("real data dir absent")
    from graph.chainlog import ChainLog

    out = GraphStore(tmp_path / "copy", v01=False)
    for line in (src / "graph" / "nodes.jsonl").open():
        r = json.loads(line)["record"]
        out.add_node(r["node_id"], r["node_type"], kind=r["kind"], verification=r["verification"], attrs=r["attrs"])
    for line in (src / "graph" / "edges.jsonl").open():
        r = json.loads(line)["record"]
        out.add_edge(r["type"], r["src"], r["dst"], r.get("attrs", {}))
    for name in ("nodes.jsonl", "edges.jsonl"):
        assert (tmp_path / "copy" / "graph" / name).read_bytes() == (src / "graph" / name).read_bytes()
    layer = src / "graph" / "hext_projection" / "nodes.jsonl"
    if layer.exists():
        before = layer.read_bytes()
        for rec in ChainLog(layer).records():
            check_stored_kind(rec)
        assert layer.read_bytes() == before
