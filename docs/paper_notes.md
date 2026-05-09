# Paper Notes

Paper: `On-Policy Self-Evolution via Failure Trajectories for Agentic Safety Alignment`

## Method

FATE builds supervision from failed agent trajectories rather than response-only labels.

At round `t`:

1. Roll out current policy `pi_t` on the development split.
2. Keep verifier-scored failures `f = (x, a, z(x, a))`.
3. Build repair prompt `p_f = Prompt(x, a, z(x, a))`.
4. Sample `K` same-policy repair candidates from `pi_t(. | p_f)`.
5. Re-score each repair candidate with benchmark verifiers.
6. Remove infeasible candidates that fail protected utility, over-refusal, or trajectory-control thresholds.
7. Keep the Pareto front under four objectives: security, utility, over-refusal control, trajectory control.
8. Use a front-only tie-breaker to select balanced replay targets.
9. Train by SFT on replay, then PFPO with group-relative Pareto rewards.

## Objective Vector

Each trajectory is scored as:

```text
z(x, a) = (security, utility, over_refusal, trajectory_control)
```

The objectives are benchmark-specific:

- AgentDojo: injection success, original task success, benign refusal, valid tool/state checks.
- AgentHarm: harmful compliance, benign/safe completion, valid refusal, valid tool/state checks.
- ATBench: safe/unsafe labels plus fine-grained diagnosis labels. It is external evaluation only.

## Pareto Replay

For task mode `tau`, feasibility thresholds are applied to:

```text
utility >= kappa_utility(tau)
over_refusal >= kappa_over_refusal(tau)
trajectory_control >= kappa_trajectory_control(tau)
```

Security is used for Pareto comparison and front scoring, not as a feasibility threshold.

The front-only reward is:

```text
r_pf = sum_m w_m(tau) z_m - lambda * max_m w_m(tau) * (1 - z_m)
```

The paper's supervision distribution keeps probability mass only on the Pareto front and reweights candidates by proposal probability and `exp(beta * r_pf)`.

## Default Hyperparameters

- `T = 2` self-evolution rounds.
- `K = 8` repair candidates per failure.
- `G = 8` PFPO completions per prompt.
- `2048` selected repairs per round.
- LoRA updates.
- SFT learning rate `2e-5`.
- PFPO learning rate `1e-6`.
- PFPO clip coefficient `0.2`.
- KL coefficient `0.01`.
- Proposal temperature `0.7`.
- 3 random seeds.

