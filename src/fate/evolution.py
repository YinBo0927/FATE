from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fate.pareto import ParetoConfig, group_by_failure, pareto_reward, select_repair_targets
from fate.prompts import build_repair_prompt
from fate.schema import FailureRecord, RepairCandidate, ReplayExample
from fate.utils.io import read_jsonl, write_jsonl


def build_replay(
    failures: Sequence[FailureRecord],
    candidates: Sequence[RepairCandidate],
    pareto_config: ParetoConfig,
    top_n: int = 1,
) -> list[ReplayExample]:
    by_failure = group_by_failure(candidates)
    replay: list[ReplayExample] = []
    for failure in failures:
        selected = select_repair_targets(
            by_failure.get(failure.failure_id, []),
            mode=failure.mode,
            config=pareto_config,
            top_n=top_n,
        )
        prompt = build_repair_prompt(failure)
        for candidate in selected:
            replay.append(
                ReplayExample(
                    failure_id=failure.failure_id,
                    candidate_id=candidate.candidate_id,
                    prompt=prompt,
                    target=candidate.trajectory,
                    scores=candidate.scores,
                    pareto_reward=pareto_reward(candidate.scores, failure.mode, pareto_config),
                    metadata={
                        "mode": failure.mode.value,
                        **dict(failure.metadata),
                        **dict(candidate.metadata),
                    },
                )
            )
    return replay


def build_replay_from_files(
    failures_path: str | Path,
    candidates_path: str | Path,
    pareto_config: ParetoConfig,
    output_path: str | Path,
    top_n: int = 1,
) -> list[ReplayExample]:
    failures = [FailureRecord.from_mapping(row) for row in read_jsonl(failures_path)]
    candidates = [RepairCandidate.from_mapping(row) for row in read_jsonl(candidates_path)]
    replay = build_replay(failures, candidates, pareto_config, top_n=top_n)
    write_jsonl(output_path, [example.asdict() for example in replay])
    return replay

