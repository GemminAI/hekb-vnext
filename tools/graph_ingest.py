#!/usr/bin/env python3
"""HEKB-GRAPH002: import the selected sessions into a NEW data dir and build the graph.

    python3 tools/graph_ingest.py --src SRC_DATA_DIR --dst NEW_DATA_DIR --selection selection.json

``--src`` is only read. ``--dst`` must not hold a graph yet unless ``--append`` is given.
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

MSD = Path("/Volumes/SSD1TB/EXP-MSD0001 v1.1")
TOKENIZER = Path.home() / (
    ".cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/"
    "8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/tokenizer.json"
)
WINDOW_8_V1 = {
    "name": "window-8-v1",
    "sha256": "feac008258153b77c0cf61f1920eb36041d603330751733e637ee36d6ddd7ae3",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--msd", type=Path, default=MSD)
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if (args.dst / "graph" / "nodes.jsonl").exists() and not args.append:
        raise SystemExit(f"{args.dst} already has a graph (use --append)")
    sel = json.loads(args.selection.read_text())
    assignment = json.loads((args.msd / "splits.json").read_text())["assignment"]
    store = GraphStore(args.dst)
    decoder = Decoder(args.tokenizer)
    rows = []
    for s in sel["sessions"]:
        rows.append(
            ingest_session(
                src=args.src,
                dst=args.dst,
                store=store,
                session_id=s,
                object_id=sel["object_id"][s],
                msd_dir=args.msd / "sessions",
                decoder=decoder,
                window_policy=WINDOW_8_V1,
                assignment=assignment,
            )
        )
    linked = link_same_content(store)
    store.sync()
    index = catalog.write(store, args.dst)
    summary = {
        "sessions": len(rows),
        "same_content_edges": linked,
        "nodes": len(store.nodes_log),
        "edges": len(store.edges_log),
        "spaces": len(index["by_space_id"]),
        "policy_sets": len(index["by_policy_set"]),
        "rows": rows,
    }
    if args.report:
        args.report.write_text(json.dumps(summary, indent=1))
    print({k: v for k, v in summary.items() if k != "rows"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
