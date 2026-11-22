"""Skills S1-S3 (DRAFT, v0.2-A 2.6). ``selection_id`` is empty until Tomo's approval (HANKO)."""

SKILLS = [
    {
        "skill_id": "S1", "version": "0.1-draft", "selection_id": "",
        "applies_to": "any numeric question with a single final value",
        "text": (
            "Work through the calculation step by step, one step per line. "
            "Put the final value on the last line as `ANSWER: <value>` and write nothing after it."
        ),
    },
    {
        "skill_id": "S2", "version": "0.1-draft", "selection_id": "",
        "applies_to": "any question; governs missing information",
        "text": (
            "Before computing, list the values the question gives. If a value needed to answer is not "
            "among them, do not guess it or assume a typical value: end with `CLARIFY: <the missing "
            "value>` instead of an ANSWER line."
        ),
    },
    {
        "skill_id": "S3", "version": "0.1-draft", "selection_id": "",
        "applies_to": "any numeric question with a single final value",
        "text": (
            "After you have a value, check it once by a different route (compute it again another way, "
            "or test that it satisfies the question). If the check disagrees, redo the calculation "
            "before answering."
        ),
    },
]


def skills_block() -> str:
    lines = ["Procedures (version-fixed):"]
    for s in SKILLS:
        lines.append(f"[{s['skill_id']} v{s['version']}] {s['text']}")
    return "\n".join(lines)
