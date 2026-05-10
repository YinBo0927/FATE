# FATE

## Framework

The framework collects failed trajectories, asks the current policy to generate repairs, filters repaired trajectories with verifier scores and Pareto-front selection, then updates the policy with SFT and PFPO.

![FATE framework](assets/fate_figure1.png)

## Experimental Results

### Main Results Across Backbones

ASR, BRR, and HCR are lower-is-better. All other metrics are higher-is-better.

| Backbone | Method | AgentDojo ASR | AgentDojo TSR | AgentDojo BRR | AgentHarm HCR | AgentHarm VRR | AgentHarm SafeScore |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-8B-Instruct | Base | 0.812 | 0.132 | 0.104 | 0.719 | 0.156 | 0.241 |
| Qwen3-8B-Instruct | FATE | **0.540** | **0.392** | **0.082** | **0.125** | **0.812** | **0.870** |
| Llama-3.1-8B-Instruct | Base | 0.768 | 0.158 | 0.118 | 0.672 | 0.188 | 0.286 |
| Llama-3.1-8B-Instruct | FATE | **0.512** | **0.417** | **0.087** | **0.156** | **0.781** | **0.842** |
| Ministral-3-8B-Instruct | Base | 0.736 | 0.176 | 0.096 | 0.641 | 0.219 | 0.314 |
| Ministral-3-8B-Instruct | FATE | **0.486** | **0.438** | **0.074** | **0.141** | **0.797** | **0.858** |
| Gemma-3-12B-it | Base | 0.704 | 0.204 | 0.132 | 0.625 | 0.234 | 0.337 |
| Gemma-3-12B-it | FATE | **0.468** | **0.462** | **0.091** | **0.172** | **0.766** | **0.821** |
| Phi-4-reasoning | Base | 0.748 | 0.168 | 0.126 | 0.688 | 0.203 | 0.301 |
| Phi-4-reasoning | FATE | **0.503** | **0.429** | **0.089** | **0.164** | **0.781** | **0.836** |

### Qwen3 Scaling Study

| Model | Method | AgentDojo ASR | AgentDojo TSR | AgentHarm HCR | AgentHarm VRR |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | Base | 0.884 | 0.071 | 0.844 | 0.063 |
| Qwen3-0.6B | FATE | **0.718** | **0.203** | **0.469** | **0.500** |
| Qwen3-1.7B | Base | 0.862 | 0.086 | 0.812 | 0.094 |
| Qwen3-1.7B | FATE | **0.653** | **0.271** | **0.344** | **0.625** |
| Qwen3-4B | Base | 0.838 | 0.108 | 0.781 | 0.125 |
| Qwen3-4B | FATE | **0.598** | **0.334** | **0.250** | **0.719** |
| Qwen3-8B | Base | 0.812 | 0.132 | 0.719 | 0.156 |
| Qwen3-8B | FATE | **0.540** | **0.392** | **0.125** | **0.812** |
| Qwen3-14B | Base | 0.916 | 0.058 | 0.750 | 0.125 |
| Qwen3-14B | FATE | **0.445** | **0.504** | **0.188** | **0.812** |
| Qwen3-32B | Base | 0.684 | 0.226 | 0.625 | 0.250 |
| Qwen3-32B | FATE | **0.384** | **0.566** | **0.094** | **0.875** |

### Baseline Comparison

| Method | AgentDojo ASR | AgentDojo TSR | AgentHarm HCR | AgentHarm VRR |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.812 | 0.132 | 0.719 | 0.156 |
| ReAct | 0.736 | 0.184 | 0.656 | 0.250 |
| Reflexion | 0.674 | 0.236 | 0.281 | 0.719 |
| Tool Filter | 0.552 | 0.312 | - | - |
| PI Detector | 0.604 | 0.348 | - | - |
| FATE | **0.540** | **0.392** | **0.125** | **0.812** |

### Iterative Self-Evolution

![Iterative self-evolution curves](assets/results_evolution.png)

### ATBench Generalization

| Model | ATBench-C Acc. | ATBench-C Prec. | ATBench-C Rec. | ATBench-C F1 | ATBench-F R.S. | ATBench-F F.M. | ATBench-F R.H. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AgentDoG-Qwen3-4B | 64.0 | 59.2 | 88.9 | 71.1 | 46.8 | 16.5 | 40.6 |
| Qwen3-8B-Instruct + FATE | **77.8** | **80.5** | 78.6 | **79.5** | **49.2** | **18.4** | **43.1** |

### Ablation Study

| Variant | AgentDojo ASR | AgentDojo TSR | AgentHarm HCR | AgentHarm VRR |
| --- | ---: | ---: | ---: | ---: |
| w/o verifier re-scoring | 0.621 | 0.281 | 0.281 | 0.625 |
| w/o over-refusal objective | 0.558 | 0.302 | 0.156 | 0.734 |
| w/o Pareto-front selection | 0.586 | 0.332 | 0.203 | 0.719 |
| w/o PFPO | 0.572 | 0.361 | 0.172 | 0.750 |
| SFT + safety-only GRPO | 0.552 | 0.286 | 0.141 | 0.703 |
| FATE | **0.540** | **0.392** | **0.125** | **0.812** |

## Reproduction

### Step 1. Clone

Clone the repository and enter the project directory.

```bash
git clone https://github.com/YinBo0927/FATE.git
cd FATE
```

### Step 2. Create Environment

Create a clean conda environment, install dependencies, and expose the local `src/` package path.

```bash
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

For GPU training, install the PyTorch build that matches your CUDA driver before running SFT or PFPO.

### Step 3. Download Datasets

Download the external benchmark assets before running any experiment.

```bash
python scripts/download_datasets.py \
  --datasets agentharm atbench agentdojo \
  --output-dir data/raw
```

### Step 4. Create Output Folders

Create folders for the first self-evolution round and evaluation outputs.

```bash
mkdir -p artifacts/round_0 results
```

### Step 5. Run Benchmark Rollouts

Run the current policy on the benchmark development split with the official benchmark runners and verifiers. Export the scored rollout records to:

```text
artifacts/round_0/dev_rollouts.scored.jsonl
```

Each record must contain `task`, `trajectory`, `mode`, and verifier `scores`. See [docs/reproduction.md](docs/reproduction.md) for the JSONL schema.

### Step 6. Collect Failures

Filter the scored rollout file and keep trajectories that violate at least one objective.

```bash
python scripts/collect_failures.py \
  --rollouts artifacts/round_0/dev_rollouts.scored.jsonl \
  --output artifacts/round_0/failures.jsonl
```

### Step 7. Build Repair Prompts

Build repair prompts from each failure trajectory and its verifier feedback.

```bash
python scripts/make_repair_prompts.py \
  --failures artifacts/round_0/failures.jsonl \
  --output artifacts/round_0/repair_prompts.jsonl
```

### Step 8. Generate Repairs

Generate same-policy repair candidates with the configured Hugging Face model.

```bash
python scripts/generate_repairs.py \
  --config configs/qwen3_8b_fate.yaml \
  --model-name-or-path Qwen/Qwen3-8B-Instruct \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/repair_candidates.jsonl
```

### Step 9. Score Repairs

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

### Step 10. Build Replay

Apply feasibility filtering and Pareto-front selection to build repair replay data.

```bash
python scripts/build_replay.py \
  --config configs/qwen3_8b_fate.yaml \
  --failures artifacts/round_0/failures.jsonl \
  --candidates artifacts/round_0/scored_repair_candidates.jsonl \
  --output artifacts/round_0/replay.jsonl
```

### Step 11. Train SFT

Run supervised fine-tuning on the selected repair replay pairs.

```bash
python scripts/train_sft.py \
  --config configs/qwen3_8b_fate.yaml \
  --replay artifacts/round_0/replay.jsonl \
  --output-dir artifacts/round_0/sft
```

### Step 12. Sample PFPO Groups

Sample grouped completions from the SFT policy for PFPO optimization.

```bash
python scripts/make_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --model-name-or-path artifacts/round_0/sft \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/pfpo_groups.unscored.jsonl
```

### Step 13. Score PFPO Groups

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

### Step 14. Train PFPO

Run PFPO using the verifier-scored grouped completions.

```bash
python scripts/train_pfpo.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.scored.jsonl \
  --sft-checkpoint artifacts/round_0/sft \
  --output-dir artifacts/round_0/pfpo
```

### Step 15. Evaluate

Evaluate replay quality or adapt this step to official benchmark test-set evaluation.

```bash
python scripts/evaluate.py \
  --replay artifacts/round_0/replay.jsonl \
  --output results/replay_metrics.json
```
