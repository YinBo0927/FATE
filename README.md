# FATE

## Framework

The framework collects failed trajectories, asks the current policy to generate repairs, filters repaired trajectories with verifier scores and Pareto-front selection, then updates the policy with SFT and PFPO.

![FATE framework](assets/fate_figure1.png)

## Experimental Results

### Qwen3-8B-Instruct Main Results

| Benchmark | Metric | Base | FATE | Direction |
| --- | --- | ---: | ---: | --- |
| AgentDojo | ASR | 0.812 | 0.540 | lower is better |
| AgentDojo | TSR | 0.132 | 0.392 | higher is better |
| AgentDojo | BRR | 0.104 | 0.082 | lower is better |
| AgentHarm | HCR | 0.719 | 0.125 | lower is better |
| AgentHarm | VRR | 0.156 | 0.812 | higher is better |
| AgentHarm | SafeScore | 0.241 | 0.870 | higher is better |

### External Trajectory-Safety Generalization

| Model | ATBench-C F1 | Risk Source | Failure Mode | Real-World Harm |
| --- | ---: | ---: | ---: | ---: |
| AgentDoG-Qwen3-4B | 71.1 | 46.8 | 16.5 | 40.6 |
| Qwen3-8B-Instruct + FATE | 79.5 | 49.2 | 18.4 | 43.1 |

## Reproduction

## Step 1. Clone

Clone the repository and enter the project directory.

```bash
git clone https://github.com/YinBo0927/FATE.git
cd FATE
```

## Step 2. Create Environment

Create a clean conda environment, install dependencies, and expose the local `src/` package path.

```bash
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

For GPU training, install the PyTorch build that matches your CUDA driver before running SFT or PFPO.

## Step 3. Download Datasets

Download the external benchmark assets before running any experiment.

```bash
python scripts/download_datasets.py \
  --datasets agentharm atbench agentdojo \
  --output-dir data/raw
```

## Step 4. Create Output Folders

Create folders for the first self-evolution round and evaluation outputs.

```bash
mkdir -p artifacts/round_0 results
```

## Step 5. Run Benchmark Rollouts

Run the current policy on the benchmark development split with the official benchmark runners and verifiers. Export the scored rollout records to:

```text
artifacts/round_0/dev_rollouts.scored.jsonl
```

Each record must contain `task`, `trajectory`, `mode`, and verifier `scores`. See [docs/reproduction.md](docs/reproduction.md) for the JSONL schema.

## Step 6. Collect Failures

Filter the scored rollout file and keep trajectories that violate at least one objective.

```bash
python scripts/collect_failures.py \
  --rollouts artifacts/round_0/dev_rollouts.scored.jsonl \
  --output artifacts/round_0/failures.jsonl
```

## Step 7. Build Repair Prompts

Build repair prompts from each failure trajectory and its verifier feedback.

```bash
python scripts/make_repair_prompts.py \
  --failures artifacts/round_0/failures.jsonl \
  --output artifacts/round_0/repair_prompts.jsonl
```

## Step 8. Generate Repairs

Generate same-policy repair candidates with the configured Hugging Face model.

```bash
python scripts/generate_repairs.py \
  --config configs/qwen3_8b_fate.yaml \
  --model-name-or-path Qwen/Qwen3-8B-Instruct \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/repair_candidates.jsonl
```

## Step 9. Score Repairs

Run the official benchmark verifier on `repair_candidates.jsonl`, then save verifier-scored candidates to:

```text
artifacts/round_0/repair_candidates.verified.jsonl
```

Normalize and validate the scored candidate file.

```bash
python scripts/score_repairs.py \
  --candidates artifacts/round_0/repair_candidates.verified.jsonl \
  --output artifacts/round_0/scored_repair_candidates.jsonl
```

## Step 10. Build Replay

Apply feasibility filtering and Pareto-front selection to build repair replay data.

```bash
python scripts/build_replay.py \
  --config configs/qwen3_8b_fate.yaml \
  --failures artifacts/round_0/failures.jsonl \
  --candidates artifacts/round_0/scored_repair_candidates.jsonl \
  --output artifacts/round_0/replay.jsonl
```

## Step 11. Train SFT

Run supervised fine-tuning on the selected repair replay pairs.

```bash
python scripts/train_sft.py \
  --config configs/qwen3_8b_fate.yaml \
  --replay artifacts/round_0/replay.jsonl \
  --output-dir artifacts/round_0/sft
```

## Step 12. Sample PFPO Groups

Sample grouped completions from the SFT policy for PFPO optimization.

```bash
python scripts/make_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --model-name-or-path artifacts/round_0/sft \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/pfpo_groups.unscored.jsonl
```

## Step 13. Score PFPO Groups

Run the official benchmark verifier on each PFPO completion and add `scores` to every completion. Save the result to:

```text
artifacts/round_0/pfpo_groups.verified.jsonl
```

Convert verifier scores into Pareto rewards.

```bash
python scripts/score_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.verified.jsonl \
  --output artifacts/round_0/pfpo_groups.scored.jsonl
```

## Step 14. Train PFPO

Run PFPO using the verifier-scored grouped completions.

```bash
python scripts/train_pfpo.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.scored.jsonl \
  --sft-checkpoint artifacts/round_0/sft \
  --output-dir artifacts/round_0/pfpo
```

## Step 15. Evaluate

Evaluate replay quality or adapt this step to official benchmark test-set evaluation.

```bash
python scripts/evaluate.py \
  --replay artifacts/round_0/replay.jsonl \
  --output results/replay_metrics.json
```
