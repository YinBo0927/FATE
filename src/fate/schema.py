from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskMode(str, Enum):
    BENIGN = "benign"
    ATTACKED_LEGITIMATE = "attacked_legitimate"
    HARMFUL_REQUEST = "harmful_request"

    @classmethod
    def coerce(cls, value: str | "TaskMode") -> "TaskMode":
        if isinstance(value, TaskMode):
            return value
        normalized = value.strip().lower().replace("-", "_")
        return cls(normalized)


@dataclass(frozen=True)
class ObjectiveScores:
    security: float
    utility: float
    over_refusal: float
    trajectory_control: float

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ObjectiveScores":
        aliases = {
            "security": ("security", "sec", "zsec"),
            "utility": ("utility", "util", "zutil"),
            "over_refusal": ("over_refusal", "over-refusal", "or", "zor"),
            "trajectory_control": ("trajectory_control", "control", "ctrl", "zctrl"),
        }
        values: dict[str, float] = {}
        for field_name, keys in aliases.items():
            for key in keys:
                if key in data:
                    values[field_name] = float(data[key])
                    break
            else:
                raise KeyError(f"missing objective score: {field_name}")
        return cls(**values).clipped()

    def clipped(self) -> "ObjectiveScores":
        return ObjectiveScores(
            security=_clip01(self.security),
            utility=_clip01(self.utility),
            over_refusal=_clip01(self.over_refusal),
            trajectory_control=_clip01(self.trajectory_control),
        )

    def asdict(self) -> dict[str, float]:
        return asdict(self)

    def items(self) -> tuple[tuple[str, float], ...]:
        return (
            ("security", self.security),
            ("utility", self.utility),
            ("over_refusal", self.over_refusal),
            ("trajectory_control", self.trajectory_control),
        )


@dataclass
class FailureRecord:
    failure_id: str
    task: str
    failed_trajectory: str
    mode: TaskMode
    scores: ObjectiveScores
    verifier_feedback: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FailureRecord":
        return cls(
            failure_id=str(data.get("failure_id") or data.get("id")),
            task=str(data["task"]),
            failed_trajectory=str(data.get("failed_trajectory") or data.get("trajectory")),
            mode=TaskMode.coerce(data["mode"]),
            scores=ObjectiveScores.from_mapping(data["scores"]),
            verifier_feedback=dict(data.get("verifier_feedback", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RepairCandidate:
    failure_id: str
    candidate_id: str
    trajectory: str
    scores: ObjectiveScores
    proposal_logprob: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RepairCandidate":
        return cls(
            failure_id=str(data["failure_id"]),
            candidate_id=str(data.get("candidate_id") or data.get("id")),
            trajectory=str(data["trajectory"]),
            scores=ObjectiveScores.from_mapping(data["scores"]),
            proposal_logprob=float(data.get("proposal_logprob", 0.0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class ReplayExample:
    failure_id: str
    candidate_id: str
    prompt: str
    target: str
    scores: ObjectiveScores
    pareto_reward: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def asdict(self) -> dict[str, Any]:
        return {
            "failure_id": self.failure_id,
            "candidate_id": self.candidate_id,
            "prompt": self.prompt,
            "target": self.target,
            "scores": self.scores.asdict(),
            "pareto_reward": self.pareto_reward,
            "metadata": dict(self.metadata),
        }


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

