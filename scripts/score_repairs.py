#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.mock import feedback_from_scores, heuristic_scores
from fate.schema import TaskMode
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score repair candidates with the configured verifier.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verifier", choices=["heuristic"], default="heuristic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.candidates):
        metadata = dict(row.get("metadata", {}))
        mode = TaskMode.coerce(metadata.get("mode", row.get("mode", "benign")))
        scores = heuristic_scores(row["trajectory"], mode)
        row["scores"] = scores.asdict()
        row["verifier_feedback"] = feedback_from_scores(scores)
        rows.append(row)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} scored repair candidates to {args.output}")


if __name__ == "__main__":
    main()
