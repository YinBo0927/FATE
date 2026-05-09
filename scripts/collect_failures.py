#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.mock import feedback_from_scores, heuristic_scores
from fate.schema import TaskMode
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or normalize failure trajectories.")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in read_jsonl(args.tasks):
        mode = TaskMode.coerce(item["mode"])
        trajectory = item.get("failed_trajectory") or item.get("trajectory")
        scores = heuristic_scores(trajectory, mode)
        rows.append(
            {
                "failure_id": item.get("failure_id") or item.get("task_id") or item.get("id"),
                "task": item["task"],
                "failed_trajectory": trajectory,
                "mode": mode.value,
                "scores": scores.asdict(),
                "verifier_feedback": feedback_from_scores(scores),
                "metadata": dict(item.get("metadata", {})),
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} failures to {args.output}")


if __name__ == "__main__":
    main()
