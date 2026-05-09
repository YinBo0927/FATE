#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.schema import RepairCandidate
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and normalize officially scored repairs.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.candidates):
        if "scores" not in row:
            raise ValueError(
                "repair candidate is missing verifier scores; run the official benchmark verifier first"
            )
        candidate = RepairCandidate.from_mapping(row)
        row["scores"] = candidate.scores.asdict()
        rows.append(row)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} scored repair candidates to {args.output}")


if __name__ == "__main__":
    main()
