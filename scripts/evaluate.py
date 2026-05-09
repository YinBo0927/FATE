#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fate.schema import ObjectiveScores
from fate.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize replay quality metrics.")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.replay)
    metrics = {
        "num_replay_examples": len(rows),
        "mean_security": _mean_score(rows, "security"),
        "mean_utility": _mean_score(rows, "utility"),
        "mean_over_refusal": _mean_score(rows, "over_refusal"),
        "mean_trajectory_control": _mean_score(rows, "trajectory_control"),
        "mean_pareto_reward": sum(float(row["pareto_reward"]) for row in rows) / max(len(rows), 1),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def _mean_score(rows: list[dict], name: str) -> float:
    if not rows:
        return 0.0
    return sum(ObjectiveScores.from_mapping(row["scores"]).asdict()[name] for row in rows) / len(rows)


if __name__ == "__main__":
    main()
