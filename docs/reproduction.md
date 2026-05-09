# Reproduction Guide

This repo expects benchmark execution logs in JSONL. That choice keeps the FATE implementation independent from fast-changing benchmark APIs while preserving the exact algorithmic steps.

## Required JSONL Schemas

### Failure Records

```json
{
  "failure_id": "agentdojo-dev-0001",
  "task": "User task text",
  "failed_trajectory": "Serialized action-observation trajectory",
  "mode": "attacked_legitimate",
  "scores": {
    "security": 0.0,
    "utility": 0.25,
    "over_refusal": 1.0,
    "trajectory_control": 0.4
  },
  "verifier_feedback": {
    "security": "followed an injected instruction",
    "utility": "did not complete the original task",
    "over_refusal": "did not refuse the benign task",
    "trajectory_control": "attempted an unauthorized state change"
  },
  "metadata": {
    "benchmark": "agentdojo",
    "split": "dev"
  }
}
```

### Repair Candidates

```json
{
  "failure_id": "agentdojo-dev-0001",
  "candidate_id": "agentdojo-dev-0001-k0",
  "trajectory": "Serialized corrected action-observation trajectory",
  "proposal_logprob": -12.4,
  "scores": {
    "security": 0.95,
    "utility": 0.85,
    "over_refusal": 1.0,
    "trajectory_control": 0.9
  },
  "metadata": {
    "benchmark": "agentdojo"
  }
}
```

### Replay Records

`scripts/build_replay.py` writes:

```json
{
  "failure_id": "agentdojo-dev-0001",
  "candidate_id": "agentdojo-dev-0001-k0",
  "prompt": "Repair prompt",
  "target": "Selected repaired trajectory",
  "scores": {
    "security": 0.95,
    "utility": 0.85,
    "over_refusal": 1.0,
    "trajectory_control": 0.9
  },
  "pareto_reward": 0.71,
  "metadata": {
    "mode": "attacked_legitimate"
  }
}
```

### PFPO Group Records

`scripts/train_pfpo.py` expects grouped completions that have already been re-scored by a verifier and converted to Pareto rewards:

```json
{
  "prompt_id": "agentdojo-dev-0001",
  "prompt": "Repair prompt",
  "completions": [
    {"text": "candidate trajectory 0", "reward": 0.61},
    {"text": "candidate trajectory 1", "reward": 0.74}
  ]
}
```

The paper samples `G=8` completions per prompt. Rewards should use the same front-only Pareto reward after benchmark-specific scoring.

## Benchmark Procedure

1. Split official benchmark tasks into dev/test. The paper mines failures only on dev.
2. Roll out the current policy on dev.
3. Convert failed trajectories to `failures.jsonl`.
4. Build repair prompts.
5. Generate `K=8` same-policy repairs for each failure.
6. Re-execute or re-score candidates using official verifiers.
7. Build Pareto replay.
8. Train SFT, then PFPO.
9. Evaluate held-out AgentDojo/AgentHarm.
10. Evaluate ATBench externally without using it for training.

## Metrics

- AgentDojo: ASR lower is better, TSR higher is better, BRR lower is better.
- AgentHarm: HCR lower is better, VRR higher is better, SafeScore higher is better.
- ATBench-C: accuracy, precision, recall, F1.
- ATBench-F: risk source, failure mode, real-world harm diagnosis accuracy.
