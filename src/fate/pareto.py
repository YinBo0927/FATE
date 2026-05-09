from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from fate.schema import ObjectiveScores, RepairCandidate, TaskMode

OBJECTIVES = ("security", "utility", "over_refusal", "trajectory_control")


@dataclass(frozen=True)
class ModeParetoConfig:
    weights: Mapping[str, float]
    thresholds: Mapping[str, float]


@dataclass(frozen=True)
class ParetoConfig:
    modes: Mapping[TaskMode, ModeParetoConfig]
    beta: float = 1.0
    lambda_shortfall: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ParetoConfig":
        modes = {}
        for mode_name, mode_data in data["modes"].items():
            mode = TaskMode.coerce(mode_name)
            modes[mode] = ModeParetoConfig(
                weights={k: float(v) for k, v in mode_data["weights"].items()},
                thresholds={k: float(v) for k, v in mode_data["thresholds"].items()},
            )
        return cls(
            modes=modes,
            beta=float(data.get("beta", 1.0)),
            lambda_shortfall=float(data.get("lambda_shortfall", 0.5)),
        )


def is_feasible(candidate: RepairCandidate, mode: TaskMode, config: ParetoConfig) -> bool:
    thresholds = config.modes[mode].thresholds
    scores = candidate.scores
    return (
        scores.utility >= thresholds["utility"]
        and scores.over_refusal >= thresholds["over_refusal"]
        and scores.trajectory_control >= thresholds["trajectory_control"]
    )


def dominates(left: ObjectiveScores, right: ObjectiveScores) -> bool:
    left_values = dict(left.items())
    right_values = dict(right.items())
    no_worse = all(left_values[name] >= right_values[name] for name in OBJECTIVES)
    strictly_better = any(left_values[name] > right_values[name] for name in OBJECTIVES)
    return no_worse and strictly_better


def pareto_front(
    candidates: Sequence[RepairCandidate], mode: TaskMode, config: ParetoConfig
) -> list[RepairCandidate]:
    feasible = [candidate for candidate in candidates if is_feasible(candidate, mode, config)]
    front: list[RepairCandidate] = []
    for candidate in feasible:
        if any(dominates(other.scores, candidate.scores) for other in feasible):
            continue
        front.append(candidate)
    return front


def pareto_reward(scores: ObjectiveScores, mode: TaskMode, config: ParetoConfig) -> float:
    weights = config.modes[mode].weights
    values = dict(scores.items())
    weighted_sum = sum(weights[name] * values[name] for name in OBJECTIVES)
    max_shortfall = max(weights[name] * (1.0 - values[name]) for name in OBJECTIVES)
    return weighted_sum - config.lambda_shortfall * max_shortfall


def supervision_distribution(
    candidates: Sequence[RepairCandidate], mode: TaskMode, config: ParetoConfig
) -> list[tuple[RepairCandidate, float]]:
    front = pareto_front(candidates, mode, config)
    if not front:
        return []
    logits = [
        candidate.proposal_logprob
        + config.beta * pareto_reward(candidate.scores, mode=mode, config=config)
        for candidate in front
    ]
    probs = _softmax(logits)
    return list(zip(front, probs, strict=True))


def select_repair_targets(
    candidates: Sequence[RepairCandidate],
    mode: TaskMode,
    config: ParetoConfig,
    top_n: int = 1,
) -> list[RepairCandidate]:
    distribution = supervision_distribution(candidates, mode, config)
    ranked = sorted(
        distribution,
        key=lambda item: (
            item[1],
            pareto_reward(item[0].scores, mode=mode, config=config),
            item[0].candidate_id,
        ),
        reverse=True,
    )
    return [candidate for candidate, _ in ranked[:top_n]]


def group_by_failure(candidates: Iterable[RepairCandidate]) -> dict[str, list[RepairCandidate]]:
    grouped: dict[str, list[RepairCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.failure_id, []).append(candidate)
    return grouped


def _softmax(logits: Sequence[float]) -> list[float]:
    max_logit = max(logits)
    exps = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exps)
    return [value / total for value in exps]

