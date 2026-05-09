#!/usr/bin/env python
from __future__ import annotations

import argparse

from datasets import load_dataset

from fate.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ATBench test split to JSONL.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="ATBench")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset("AI45Research/ATBench", args.config, split=args.split)
    rows = []
    for item in dataset:
        rows.append(
            {
                "id": item.get("id"),
                "trajectory": item.get("contents") or item.get("content"),
                "label": item.get("label"),
                "risk_source": item.get("risk_source"),
                "failure_mode": item.get("failure_mode"),
                "real_world_harm": item.get("real_world_harm"),
                "reason": item.get("reason"),
                "tool_used": item.get("tool_used"),
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} ATBench rows to {args.output}")


if __name__ == "__main__":
    main()

