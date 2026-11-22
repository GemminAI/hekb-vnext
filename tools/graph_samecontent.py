#!/usr/bin/env python3
"""HEKB-GRAPH002: the same session measured twice with only ``experiment_id`` changed.

Takes object ids from two (or more) source data dirs, imports them into one new data dir and
reports ``object_id`` / ``content_id`` / ``same_content_as``.

    python3 tools/graph_samecontent.py --dst DIR --session S SRC_DIR:OBJECT_ID [SRC_DIR:OBJECT_ID ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph import catalog  # noqa: E402
from graph.ingest import ingest_session, link_same_content  # noqa: E402
from graph.source_text import Decoder  # noqa: E402
from graph.store import GraphStore  # noqa: E402
from tools.graph_ingest import MSD, TOKENIZER, WINDOW_8_V1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("items", nargs="+", help="SRC_DIR:OBJECT_ID")
    args = parser.parse_args()
    assignment = json.loads((MSD / "splits.json").read_text())["assignment"]
    store = GraphStore(args.dst)
    decoder = Decoder(TOKENIZER)
    rows = []
    for item in args.items:
        src, oid = item.rsplit(":", 1)
        rows.append(
            ingest_session(
                src=Path(src), dst=args.dst, store=store, session_id=args.session, object_id=oid,
                msd_dir=MSD / "sessions", decoder=decoder, window_policy=WINDOW_8_V1, assignment=assignment,
            )
        )
    linked = link_same_content(store)
    store.sync()
    catalog.write(store, args.dst)
    same = [e for e in store.edges() if e["type"] == "same_content_as"]
    report = {
        "object_ids": [r["object_id"] for r in rows],
        "content_ids": [r["content_id"] for r in rows],
        "object_ids_distinct": len({r["object_id"] for r in rows}) == len(rows),
        "same_content_edges": same,
        "linked_now": linked,
    }
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
