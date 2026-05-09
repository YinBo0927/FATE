#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.prompts import build_repair_prompt
from fate.schema import FailureRecord
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert failure records to repair prompts.")
    parser.add_argument("--failures", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.failures):
        failure = FailureRecord.from_mapping(row)
        rows.append(
            {
                "failure_id": failure.failure_id,
                "prompt": build_repair_prompt(failure),
                "mode": failure.mode.value,
                "metadata": dict(failure.metadata),
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} repair prompts to {args.output}")


if __name__ == "__main__":
    main()

