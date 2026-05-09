# FATE

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

## Step 3. Run Everything

Run the complete smoke pipeline with bundled sample data and mock model components.

```bash
bash scripts/run_all.sh
```

The full smoke run writes:

```text
results/smoke_metrics.json
```

## Step 4. Run Step by Step

Create output folders for intermediate artifacts and final metrics.

```bash
mkdir -p artifacts/round_0 results
```

Create a small local sample dataset that can run without external credentials.

```bash
python scripts/download_datasets.py \
  --datasets sample \
  --output-dir data/raw
```

Convert the sample task records into verifier-scored failure trajectories.

```bash
python scripts/collect_failures.py \
  --tasks data/raw/sample/tasks.jsonl \
  --output artifacts/round_0/failures.jsonl
```

Build repair prompts from the failure trajectories and verifier feedback.

```bash
python scripts/make_repair_prompts.py \
  --failures artifacts/round_0/failures.jsonl \
  --output artifacts/round_0/repair_prompts.jsonl
```

Generate same-policy repair candidates with the mock policy backend.

```bash
python scripts/generate_repairs.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/repair_candidates.jsonl
```

Score each repair candidate with the heuristic verifier used by the smoke run.

```bash
python scripts/score_repairs.py \
  --candidates artifacts/round_0/repair_candidates.jsonl \
  --output artifacts/round_0/scored_repair_candidates.jsonl
```

Apply feasibility filtering and Pareto-front selection to build repair replay data.

```bash
python scripts/build_replay.py \
  --config configs/qwen3_8b_fate.yaml \
  --failures artifacts/round_0/failures.jsonl \
  --candidates artifacts/round_0/scored_repair_candidates.jsonl \
  --output artifacts/round_0/replay.jsonl
```

Run the mock SFT stage and write a placeholder checkpoint for the smoke pipeline.

```bash
python scripts/train_sft.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --replay artifacts/round_0/replay.jsonl \
  --output-dir artifacts/round_0/sft
```

Sample grouped PFPO completions from the mock policy backend.

```bash
python scripts/make_pfpo_groups.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/pfpo_groups.unscored.jsonl
```

Score PFPO completions and convert verifier scores into Pareto rewards.

```bash
python scripts/score_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.unscored.jsonl \
  --output artifacts/round_0/pfpo_groups.scored.jsonl
```

Run the mock PFPO stage and write a placeholder checkpoint for the smoke pipeline.

```bash
python scripts/train_pfpo.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.scored.jsonl \
  --sft-checkpoint artifacts/round_0/sft \
  --output-dir artifacts/round_0/pfpo
```

Summarize replay quality into a metrics JSON file.

```bash
python scripts/evaluate.py \
  --replay artifacts/round_0/replay.jsonl \
  --output results/smoke_metrics.json
```

## Step 5. Download Real Datasets

Download the external benchmark assets used for real experiments.

```bash
python scripts/download_datasets.py \
  --datasets agentharm atbench agentdojo \
  --output-dir data/raw
```

## Step 6. Run with a Real HF Model

Use `--backend hf` instead of `--backend mock` for generation and training commands. This switches from the local mock backend to Hugging Face model loading.

Real runs require GPU-compatible PyTorch, Hugging Face access, benchmark rollout logs, and official benchmark verifiers.
