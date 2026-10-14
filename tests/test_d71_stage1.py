# ruff: noqa: E501
import datetime

from d71 import tasks
from d71.summarize import summarize_v1

THETA = 0.013114872304912739


def _rec(t, n, v, step):
    ports = {f"P{i:02d}": {"result": {"x": 1.0}} for i in (1, 9, 13, 19, 25, 31, 35)}
    ports["P28"] = {"result": {"mean_vector": [3.0, 4.0]}}
    ports["P18"] = {"result": {"n": n, "v_norm": v, "window": 10}}
    ports["P03"] = {"result": {"mean_step": step}}
    return {"token_index": t, "ports": ports}


def test_summarize_v1_golden():
    records = [_rec(t, min(t + 1, 8), 0.001 * (t + 1) if t >= 7 else None, 1.0 + t) for t in range(10)]
    pieces = ["a", " b", "\n", "d", "e", "f", "g", "h", "i", "j"]
    got = summarize_v1(session="s1", object_id="0123456789abcdef", records=records, pieces=pieces, theta=THETA)
    assert got == (
        'Run s1 (node 0123456789ab, summarize_v1).\n'
        '- Generated tokens: 10. Full measurement window (n=8) from token 7.\n'
        '- mean_step (P03): median 5.5, max 10 at token 9 ("j").\n'
        '- P18 v_norm (full window only): median 0.009, max 0.01 at token 9 ("j"); '
        'first above the fit-reference threshold 0.01311: none.\n'
        '- Port medians P01 1, P03 5.5, P09 1, P13 1, P18 0.009, P19 1, P25 1, P28 5, P31 1, P35 1.\n'
        '- Time axis: event-time-v1 (1 s per generated token; synthetic, not wall-clock). '
        'No interpretation and no correctness is stated.'
    )


def test_deficient_item_checks():
    rec = {
        "prompt_id": "p", "kind": "remainder", "difficulty": "easy",
        "params": {"a": 5, "b": 5, "m": 13}, "expected": "5",
        "prompt": "What is the remainder when 5^5 is divided by 13? " + tasks.ANSWER_INSTRUCTION,
    }
    item = tasks.build_item(rec)
    assert item["valid"] and item["check_i_rerender_and_answer"]
    d = item["deficient"]
    assert d["removed"] in ("a", "b", "m") and "given earlier" in d["body"]
    assert not tasks.removed_literal_present(d["body"], d["removed_literal"])


def test_answers_match_make_task_formulas():
    assert tasks.answer("nth_prime", {"n": 9}) == "23"
    assert tasks.answer("pow2", {"n": 19}) == "524288"
    assert tasks.answer("pi_digits", {"decimals": 4}) == "3.1415"
    assert tasks.answer("day_of_week", {"date": "2024-01-01"}) == "Monday"
    assert tasks.answer("kth_largest", {"numbers": [149, 198, 613, 284, 583, 473, 678], "k": 2}) == "613"
    assert datetime.date(2024, 1, 1).weekday() == 0


def test_derivability_and_retrieval_are_deterministic():
    from d71 import context as C

    assert C.derivable("pi_digits", {"decimals": 10}, {"decimals": 8})
    assert not C.derivable("pi_digits", {"decimals": 4}, {"decimals": 8})
    assert C.derivable("nth_prime", {"n": 12}, {"n": 14})
    assert not C.derivable("nth_prime", {"n": 12}, {"n": 15})
    lib = [
        {"session": f"s{i}", "prompt_id": f"p{i}", "prompt_sha256": f"h{i}", "kind": "pow2",
         "params": {"n": 8 + 3 * i}, "body": "b", "extracted": "x"}
        for i in range(6)
    ]
    item = C.ItemPublic("q", "hq", "pow2", "easy", {"n": 12}, "b", "d")
    a = C.retrieve_b(item, lib)
    b = C.retrieve_b(item, list(reversed(lib)))
    assert a is not None and [e["session"] for e in a] == [e["session"] for e in b]
    assert [e["params"]["n"] for e in a] == [8, 17, 20]  # 11 and 14 are within 2 of 12 (derivable)


def test_cprime_is_deterministic_and_other_kind():
    from d71 import context as C

    lib = [
        {"session": f"{k}{i}", "prompt_id": f"{k}{i}", "prompt_sha256": "h", "kind": k, "params": {}, "body": "", "extracted": ""}
        for k in ("a", "b", "c") for i in range(4)
    ]
    item = C.ItemPublic("zz", "hz", "d", "easy", {}, "b", None)
    count = lambda t: len(t.split())  # noqa: E731
    obs = lambda s: f"obs {s} " + "w " * 5  # noqa: E731
    cs = C.select_cprime(item, lib, count, C.obs_section(lib[:3], obs), obs)[0]
    cs2 = C.select_cprime(item, list(reversed(lib)), count, C.obs_section(lib[:3], obs), obs)[0]
    assert cs is not None and [e["session"] for e in cs] == [e["session"] for e in cs2]
    assert all(e["kind"] != "d" for e in cs) and len({e["session"] for e in cs}) == 3


def test_m2_vocab_and_cprime_rule_hashes_are_stable():
    from d71 import context as C
    from d71 import m2_vocab, skills

    assert len(m2_vocab.vocab_sha256()) == 64 and m2_vocab.vocab_sha256() == m2_vocab.vocab_sha256()
    assert C.CPRIME_RULE_SHA256 == __import__("hashlib").sha256(C.CPRIME_RULE.encode()).hexdigest()
    assert all(s["selection_id"] == "" for s in skills.SKILLS)  # not approved
