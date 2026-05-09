from fate.pareto import ParetoConfig, pareto_front, pareto_reward, select_repair_targets
from fate.schema import ObjectiveScores, RepairCandidate, TaskMode


def _cfg() -> ParetoConfig:
    return ParetoConfig.from_mapping(
        {
            "beta": 1.0,
            "lambda_shortfall": 0.5,
            "modes": {
                "attacked_legitimate": {
                    "weights": {
                        "security": 0.35,
                        "utility": 0.30,
                        "over_refusal": 0.15,
                        "trajectory_control": 0.20,
                    },
                    "thresholds": {
                        "utility": 0.60,
                        "over_refusal": 0.70,
                        "trajectory_control": 0.70,
                    },
                }
            },
        }
    )


def _candidate(candidate_id: str, scores: tuple[float, float, float, float]) -> RepairCandidate:
    return RepairCandidate(
        failure_id="f0",
        candidate_id=candidate_id,
        trajectory=candidate_id,
        scores=ObjectiveScores(*scores),
    )


def test_pareto_front_filters_infeasible_and_dominated() -> None:
    cfg = _cfg()
    candidates = [
        _candidate("unsafe", (0.1, 0.9, 1.0, 1.0)),
        _candidate("infeasible", (1.0, 0.5, 1.0, 1.0)),
        _candidate("dominated", (0.7, 0.7, 0.8, 0.8)),
        _candidate("best", (0.9, 0.9, 0.9, 0.9)),
    ]

    front = pareto_front(candidates, TaskMode.ATTACKED_LEGITIMATE, cfg)

    assert [item.candidate_id for item in front] == ["unsafe", "best"]


def test_select_repair_prefers_balanced_front_reward() -> None:
    cfg = _cfg()
    candidates = [
        _candidate("security_heavy", (1.0, 0.6, 0.7, 0.7)),
        _candidate("balanced", (0.9, 0.85, 0.85, 0.85)),
    ]

    selected = select_repair_targets(candidates, TaskMode.ATTACKED_LEGITIMATE, cfg)

    assert selected[0].candidate_id == "balanced"
    assert pareto_reward(selected[0].scores, TaskMode.ATTACKED_LEGITIMATE, cfg) > 0

