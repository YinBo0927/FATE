#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.schema import FailureRecord, ObjectiveScores, TaskMode
from fate.utils.io import read_jsonl, write_jsonl
from fate.verifiers import failure_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect failures from scored benchmark rollouts.")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for item in read_jsonl(args.rollouts):
        mode = TaskMode.coerce(item["mode"])
        scores = ObjectiveScores.from_mapping(item["scores"])
        if not args.keep_all and not failure_violation(scores, mode):
            continue
        failure = FailureRecord(
            failure_id=str(item.get("failure_id") or item.get("task_id") or item.get("id")),
            task=str(item["task"]),
            failed_trajectory=str(item.get("failed_trajectory") or item.get("trajectory")),
            mode=mode,
            scores=scores,
            verifier_feedback=dict(item.get("verifier_feedback", {})),
            metadata=dict(item.get("metadata", {})),
        )
        rows.append(
            {
                "failure_id": failure.failure_id,
                "task": failure.task,
                "failed_trajectory": failure.failed_trajectory,
                "mode": failure.mode.value,
                "scores": failure.scores.asdict(),
                "verifier_feedback": dict(failure.verifier_feedback),
                "metadata": dict(failure.metadata),
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} failures to {args.output}")


if __name__ == "__main__":
    main()
