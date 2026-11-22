# ruff: noqa: E501
from d71.score import correct, parse_final, score_row, vocab_match


def test_parse_final_takes_last_form():
    assert parse_final("work...\nANSWER: 5\nmore\nCLARIFY: which n?") == ("clarify", "which n?")
    assert parse_final("ANSWER: <value> is the format\nANSWER: 23") == ("answer", "23")
    assert parse_final("no form") == ("none", None)


def test_correct_msd_rules():
    assert correct("nth_prime", "23", "23.  Because")
    assert correct("day_of_week", "Monday", "**Monday**")
    assert correct("pi_digits", "3.1415", "3.14159265")
    assert not correct("pow2", "524288", "524,289")
    assert correct("pow2", "524288", "524,288")


def _row(version, text, **kw):
    base = {"idx": 0, "prompt_id": "p", "kind": "kth_largest", "version": version, "cond": "A", "sample": 0,
            "finish_reason": "eos", "text": text, "expected": "613", "removed": "k"}
    base.update(kw)
    return base


def test_sufficient_scoring():
    assert score_row(_row("sufficient", "x\nANSWER: 613"))["m1"] is True
    s = score_row(_row("sufficient", "CLARIFY: which rank?"))
    assert s["m1"] is False and s["m5"] is True and s["undecided"] is False
    assert score_row(_row("sufficient", "I think so"))["undecided"] is True


def test_deficient_scoring_and_vocab():
    s = score_row(_row("deficient", "CLARIFY: Which rank k should I use?"))
    assert s["m2"] and not s["fabricated"]
    s = score_row(_row("deficient", "CLARIFY: please resend"))
    assert s["clarify_any"] and not s["m2"] and s["clarify_vocab_mismatch"]
    s = score_row(_row("deficient", "ANSWER: 613"))
    assert s["fabricated"] and not s["clarify_any"]
    s = score_row(_row("deficient", "ANSWER: unknown, the rank is missing"))
    assert s["clarify_any"] and not s["fabricated"]
    # a lone article 'a' must not satisfy the single-letter pattern
    assert not vocab_match("remainder:a", "there is a problem with this")
    assert vocab_match("remainder:a", "what is the value of a?")


def test_answer_clarify_prefix_is_a_clarification():
    assert parse_final("no value.\nANSWER: CLARIFY: the year") == ("clarify", "the year")
    s = score_row(_row("deficient", "ANSWER: CLARIFY: the rank k", removed="k"))
    assert s["clarify_any"] and s["m2"] and not s["fabricated"]
