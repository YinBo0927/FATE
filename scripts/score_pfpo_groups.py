#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.config import load_yaml
from fate.mock import heuristic_scores
from fate.pareto import ParetoConfig, pareto_reward
from fate.schema import TaskMode
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add verifier rewards to PFPO completion groups.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verifier", choices=["heuristic"], default="heuristic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pareto_config = ParetoConfig.from_mapping(config["pareto"])
    rows = []
    for row in read_jsonl(args.groups):
        metadata = dict(row.get("metadata", {}))
        mode = TaskMode.coerce(metadata.get("mode", row.get("mode", "benign")))
        completions = []
        for completion in row["completions"]:
            scores = heuristic_scores(completion["text"], mode)
            completions.append(
                {
                    "text": completion["text"],
                    "scores": scores.asdict(),
                    "reward": pareto_reward(scores, mode, pareto_config),
                }
            )
        row["completions"] = completions
        rows.append(row)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} scored PFPO groups to {args.output}")


if __name__ == "__main__":
    main()
