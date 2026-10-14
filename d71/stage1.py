# ruff: noqa: E501
#!/usr/bin/env python3
"""EXP-D71-CONTEXT-001 stage 1: split, pools, deficient versions, library ingest, summaries, contexts,
leak checks, the 40-item selection. No model is used; MSD0001 and INFO001 are read only.

    python3 -m d71.stage1 --out DIR [--skip-ingest]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from d71 import context as C  # noqa: E402
from d71 import m2_vocab, skills, summarize, tasks  # noqa: E402
from d71.data import load_records  # noqa: E402

EVID = Path.home() / "vaults/20260124/experiment-evidence"
INFO001 = Path("/Volumes/SSD1TB/info001_work/run/hekb_data")
WORK = Path("/Volumes/SSD1TB/d71_work")
HEKB_LIB = WORK / "hekb_lib"
TOKENIZER_DIR = WORK / "tokenizer"
THETA_FILE = EVID / "EXP-C18-RECAL001/c18_recal001_results.json"
N_SELECT = 40
#: kinds whose answer space is so small that a case answer equal to the expected answer is expected by chance
SMALL_ANSWER_SPACE = {"day_of_week", "remainder"}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=EVID / "EXP-D71-CONTEXT-001/stage1")
    ap.add_argument("--skip-ingest", action="store_true")
    args = ap.parse_args()
    out: Path = args.out
    import shutil

    shutil.rmtree(out / "contexts", ignore_errors=True)
    (out / "contexts").mkdir(parents=True, exist_ok=True)

    data = load_records()
    recs = data["records"]
    split_info: dict[str, Any] = {"locked_sessions_excluded_by_id": data["locked_count"]}

    # ---- 1. split ------------------------------------------------------------------------------
    sides: dict[str, set[str]] = collections.defaultdict(set)
    for r in recs:
        sides[r["prompt_id"]].add(r["side"])
    crossing = sorted(p for p, s in sides.items() if len(s) > 1)
    fit_sessions = [r for r in recs if r["side"] == "fit" and r["prompt_id"] not in crossing]
    dev_sessions = [r for r in recs if r["side"] == "dev" and r["prompt_id"] not in crossing]
    lib_by_prompt: dict[str, dict[str, Any]] = {}
    undecided = [r["session"] for r in fit_sessions if r["verdict_correct"] is not True]
    for r in sorted(fit_sessions, key=lambda r: sha(r["session"])):
        if r["verdict_correct"] is True:  # only machine-verified correct sessions may be retrieved
            lib_by_prompt.setdefault(r["prompt_id"], r)
    library = [
        {
            "session": r["session"], "prompt_id": r["prompt_id"], "prompt_sha256": r["prompt_sha256"],
            "kind": r["kind"], "difficulty": r["difficulty"], "params": r["params"],
            "body": r["prompt"][: -len(" " + tasks.ANSWER_INSTRUCTION)],
            "extracted": r["extracted"],
        }
        for r in lib_by_prompt.values() if r["kind"] in tasks.KINDS
    ]
    eval_recs: dict[str, dict[str, Any]] = {}
    for r in sorted(dev_sessions, key=lambda r: sha(r["session"])):
        if r["kind"] in tasks.KINDS:
            eval_recs.setdefault(r["prompt_id"], r)
    split_info.update(
        crossing_prompt_ids=crossing,
        fit_library_sessions=len(fit_sessions),
        fit_library_prompts_verified=len(lib_by_prompt),
        fit_verified_correct_sessions=sum(1 for r in fit_sessions if r['verdict_correct'] is True),
        fit_without_machine_verdict=undecided,
        library_kinds=dict(collections.Counter(e["kind"] for e in library)),
        eval_pool_prompts=len(eval_recs),
        eval_pool_kinds=dict(collections.Counter(r["kind"] for r in eval_recs.values())),
        excluded_kind_no_library=sorted({r["kind"] for r in dev_sessions} - set(tasks.KINDS)),
    )
    (out / "split.json").write_text(json.dumps(
        {**split_info, "library_sessions": [r["session"] for r in fit_sessions],
         "library_entries": library, "eval_pool": sorted(eval_recs)}, indent=1))

    # ---- 2. deficient versions + checks --------------------------------------------------------
    items: dict[str, dict[str, Any]] = {}
    for pid, r in eval_recs.items():
        it = tasks.build_item({k: r[k] for k in ("prompt_id", "kind", "difficulty", "params", "prompt", "expected")})
        it["prompt_sha256"] = r["prompt_sha256"]
        items[pid] = it
    check_stats = {
        "pool": len(items),
        "i_pass": sum(1 for i in items.values() if i["check_i_rerender_and_answer"]),
        "valid_deficient": sum(1 for i in items.values() if i["valid"]),
        "first_attempt_failed": sum(1 for i in items.values() if i["valid"] and len(i["attempts"]) > 1),
        "by_kind_valid": dict(collections.Counter(i["kind"] for i in items.values() if i["valid"])),
    }
    (out / "items.json").write_text(json.dumps(items, indent=1))

    # ---- 3. library ingest (graph_ingest: objects copied byte-identically; graph built) ---------
    obj_ids = json.loads((EVID / "PORT38-INFO001/regeneration_summary.json").read_text())["object_ids"]
    sel = {"sessions": [r["session"] for r in fit_sessions], "object_id": {r["session"]: obj_ids[r["session"]] for r in fit_sessions}}
    (out / "library_selection.json").write_text(json.dumps(sel, indent=1))
    if not args.skip_ingest:
        cmd = [sys.executable, str(Path(__file__).resolve().parent.parent / "tools/graph_ingest.py"),
               "--src", str(INFO001), "--dst", str(HEKB_LIB), "--selection", str(out / "library_selection.json"),
               "--report", str(out / "library_ingest_report.json")]
        subprocess.run(cmd, check=True)

    # ---- 4. summaries --------------------------------------------------------------------------
    from graph.trace import GraphQuery

    theta = float(json.loads(THETA_FILE.read_text())["theta_fit_p95"])
    q = GraphQuery(HEKB_LIB)
    summaries: dict[str, dict[str, str]] = {}
    for e in library:
        traj = q.trajectory_of(e["session"])
        cl = q._out[(traj, "has_causal_log")][0]["dst"].removeprefix("blob:")
        st = q._out[(traj, "has_source_text")][0]["dst"].removeprefix("blob:")
        records = q._json_blob(cl)
        st_blob = q._json_blob(st)
        pieces = [st_blob["text"][a:b] for a, b in st_blob["token_char_ranges"]]
        text = summarize.summarize_v1(session=e["session"], object_id=traj, records=records, pieces=pieces, theta=theta)
        summaries[e["session"]] = {"text": text, "sha256": sha(text), "object_id": traj}
    fp: dict[str, Any] = dict(summarize.version_fingerprint())
    fp["theta_source"] = str(THETA_FILE)
    fp["theta_source_sha256"] = hashlib.sha256(THETA_FILE.read_bytes()).hexdigest()
    fp["theta"] = theta
    (out / "summaries.json").write_text(json.dumps({"fingerprint": fp, "summaries": summaries}, indent=1))

    # ---- 5. tokenizer, contexts, leak checks ---------------------------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    count = lambda t: len(tok.encode(t, add_special_tokens=False))  # noqa: E731
    system_prompt = data["system_prompt"]
    obs_of = lambda s: summaries[s]["text"]  # noqa: E731

    def ctx_tokens(user: str) -> int:
        ids = tok.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=True,
        )
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        return len(ids)

    skills_text = skills.skills_block()
    fit_sessions_set = {r["session"] for r in fit_sessions}
    fail_reasons: collections.Counter[str] = collections.Counter()
    coincidences: collections.Counter[str] = collections.Counter()
    literal_in_context: collections.Counter[str] = collections.Counter()
    built: dict[str, dict[str, Any]] = {}
    token_rows: list[dict[str, Any]] = []
    for pid, it in items.items():
        if not it["valid"]:
            fail_reasons["no_valid_deficient"] += 1
            continue
        pub = C.ItemPublic(pid, it["prompt_sha256"], it["kind"], it["difficulty"], it["params"],
                           it["sufficient_body"], it["deficient"]["body"])
        cases = C.retrieve_b(pub, library)
        if cases is None:
            fail_reasons["fewer_than_3_B_candidates"] += 1
            continue
        c_obs = C.obs_section(cases, obs_of)
        cprime, note = C.select_cprime(pub, library, count, c_obs, obs_of)
        if cprime is None:
            fail_reasons["no_Cprime_" + note.split()[0]] += 1
            continue
        evidence = C.cases_section(cases)
        cprime_obs = C.obs_section(cprime, obs_of)
        problems: list[str] = []
        # leak checks (2.7)
        equal_cases = [c["session"] for c in cases if c["extracted"] == it["expected"]]
        if equal_cases and it["kind"] not in SMALL_ANSWER_SPACE:
            problems.append("L3_case_answer_equals_expected")
        if equal_cases and it["kind"] in SMALL_ANSWER_SPACE:
            coincidences[it["kind"]] += 1  # allowed: a small answer space makes equality unavoidable (2.7-3)
        if C.leak_literal(evidence + "\n" + c_obs + "\n" + cprime_obs, it["expected"]):
            literal_in_context[it["kind"]] += 1  # information only (numbers such as 5 occur everywhere)
        for c in cases + cprime:
            if c["session"] not in fit_sessions_set:
                problems.append("L5_not_fit")
        for c in cases:
            if C.derivable(it["kind"], c["params"], it["params"]):
                problems.append("L2_derivable")
        for text in (c_obs, cprime_obs, skills_text):
            if it["sufficient_body"] in text or it["deficient"]["body"] in text:
                problems.append("L4_question_in_context")
        if problems:
            for p in set(problems):
                fail_reasons[p] += 1
            continue
        conds: dict[str, dict[str, Any]] = {}
        for version, body in (("sufficient", it["sufficient_body"]), ("deficient", it["deficient"]["body"])):
            parts: dict[str, list[str]] = {
                "A": [],
                "B": [evidence],
                "C": [evidence, c_obs],
                "Cprime": [evidence, cprime_obs],
                "D": [evidence, c_obs, skills_text],
            }
            for cond, part_list in parts.items():
                user = C.assemble(cond, body, part_list)
                n = ctx_tokens(user)
                conds[f"{version}:{cond}"] = {"user": user, "tokens": n, "sha256": sha(user)}
                token_rows.append({"prompt_id": pid, "kind": it["kind"], "version": version, "cond": cond, "tokens": n})
        built[pid] = {
            "cases": [c["session"] for c in cases], "cprime": [c["session"] for c in cprime],
            "cprime_note": note, "conditions": conds,
            "obs_tokens_C": count(c_obs), "obs_tokens_Cprime": count(cprime_obs),
        }

    pool = len(items)
    usable = len(built)
    fail_frac = 1 - usable / pool
    # ---- 6. selection of 40 (deterministic, 7 kinds, easy/hard half) ---------------------------
    groups: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for pid in built:
        groups[(items[pid]["kind"], items[pid]["difficulty"])].append(pid)
    for g in groups.values():
        g.sort(key=lambda p: sha("D71|sel|" + p))
    chosen: list[str] = []
    kinds = sorted(tasks.KINDS)
    for tier in ("easy", "hard"):
        idx = {k: 0 for k in kinds}
        picked = 0
        progress = True
        while picked < N_SELECT // 2 and progress:
            progress = False
            for kind in kinds:
                pool_ = groups.get((kind, tier), [])
                if picked < N_SELECT // 2 and idx[kind] < len(pool_):
                    chosen.append(pool_[idx[kind]])
                    idx[kind] += 1
                    picked += 1
                    progress = True
    selection = {
        "n": len(chosen), "rule": "per tier (easy, hard: 20 each) round-robin over sorted kinds; within (kind,tier) ascending "
        "sha256('D71|sel|'+prompt_id); items must have a complete context set",
        "items": [{"prompt_id": p, "kind": items[p]["kind"], "difficulty": items[p]["difficulty"]} for p in chosen],
        "by_kind": dict(collections.Counter(items[p]["kind"] for p in chosen)),
        "by_tier": dict(collections.Counter(items[p]["difficulty"] for p in chosen)),
    }
    (out / "selection40.json").write_text(json.dumps(selection, indent=1))
    for p in chosen:
        d = out / "contexts" / p
        d.mkdir(exist_ok=True, parents=True)
        (d / "context_set.json").write_text(json.dumps({
            "prompt_id": p, "kind": items[p]["kind"], "difficulty": items[p]["difficulty"],
            "system_prompt_sha256": sha(system_prompt), "sufficient_body": items[p]["sufficient_body"],
            "deficient": items[p]["deficient"], "expected": items[p]["expected"], **built[p],
        }, indent=1))

    # ---- 7. statistics ------------------------------------------------------------------------
    import statistics

    def mean_by(cond: str, version: str) -> float:
        v = [r["tokens"] for r in token_rows if r["cond"] == cond and r["version"] == version]
        return round(statistics.mean(v), 1) if v else float("nan")

    diffs = [abs(b["obs_tokens_C"] - b["obs_tokens_Cprime"]) / b["obs_tokens_C"] for b in built.values()]
    stats = {
        "pool": pool, "usable_items": usable, "dropped": pool - usable, "dropped_fraction": round(fail_frac, 4),
        "stop_rule_20pct_exceeded": fail_frac > 0.20, "drop_reasons": dict(fail_reasons),
        "deficient_checks": check_stats,
        "mean_context_tokens": {f"{v}:{c}": mean_by(c, v) for v in ("sufficient", "deficient") for c in ("A", "B", "C", "Cprime", "D")},
        "C_vs_Cprime_observation_token_relative_diff": {
            "mean": round(statistics.mean(diffs), 4) if diffs else None, "max": round(max(diffs), 4) if diffs else None,
        },
        "mean_obs_tokens": {
            "C": round(statistics.mean(b["obs_tokens_C"] for b in built.values()), 1) if built else None,
            "Cprime": round(statistics.mean(b["obs_tokens_Cprime"] for b in built.values()), 1) if built else None,
        },
        "mean_full_context_token_diff_C_minus_Cprime": round(
            mean_by("C", "sufficient") - mean_by("Cprime", "sufficient"), 1),
        "L3_coincidence_allowed_small_answer_space": dict(coincidences),
        "expected_literal_in_context_info_only": dict(literal_in_context),
        "usable_by_kind": dict(collections.Counter(items[p]["kind"] for p in built)),
        "usable_by_tier": dict(collections.Counter(items[p]["difficulty"] for p in built)),
        "skills_max_tokens": max(count(s["text"]) for s in skills.SKILLS),
        "skills_tokens": {s["skill_id"]: count(s["text"]) for s in skills.SKILLS},
        "summary_tokens_mean": round(statistics.mean(count(s["text"]) for s in summaries.values()), 1),
    }
    (out / "pool_stats.json").write_text(json.dumps(stats, indent=1))
    (out / "retrieval_config.json").write_text(json.dumps(
        {"config": C.RETRIEVAL_CONFIG, "retrieval_config_sha256": C.retrieval_config_sha256(
            {"summarizer_source_sha256": fp["source_sha256"], "summarizer_template_sha256": fp["template_sha256"]}),
         "cprime_rule_sha256": C.CPRIME_RULE_SHA256}, indent=1))
    (out / "cprime_rule.txt").write_text(C.CPRIME_RULE)
    (out / "m2_vocab.json").write_text(json.dumps(
        {"version": m2_vocab.VERSION, "sha256": m2_vocab.vocab_sha256(), "vocab": m2_vocab.VOCAB},
        indent=1, ensure_ascii=False))
    (out / "skills_draft.json").write_text(json.dumps(skills.SKILLS, indent=1, ensure_ascii=False))
    (out / "token_counts.csv").write_text(
        "prompt_id,kind,version,cond,tokens\n"
        + "\n".join(f"{r['prompt_id']},{r['kind']},{r['version']},{r['cond']},{r['tokens']}" for r in token_rows))
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
