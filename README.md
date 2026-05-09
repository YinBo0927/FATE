# FATE

## Step 1. Clone

```bash
git clone https://github.com/YinBo0927/FATE.git
cd FATE
```

## Step 2. Create Environment

```bash
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

## Step 3. Run Everything

```bash
bash scripts/run_all.sh
```

The full smoke run writes:

```text
results/smoke_metrics.json
```

## Step 4. Run Step by Step

```bash
mkdir -p artifacts/round_0 results
```

```bash
python scripts/download_datasets.py \
  --datasets sample \
  --output-dir data/raw
```

```bash
python scripts/collect_failures.py \
  --tasks data/raw/sample/tasks.jsonl \
  --output artifacts/round_0/failures.jsonl
```

```bash
python scripts/make_repair_prompts.py \
  --failures artifacts/round_0/failures.jsonl \
  --output artifacts/round_0/repair_prompts.jsonl
```

```bash
python scripts/generate_repairs.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/repair_candidates.jsonl
```

```bash
python scripts/score_repairs.py \
  --candidates artifacts/round_0/repair_candidates.jsonl \
  --output artifacts/round_0/scored_repair_candidates.jsonl
```

```bash
python scripts/build_replay.py \
  --config configs/qwen3_8b_fate.yaml \
  --failures artifacts/round_0/failures.jsonl \
  --candidates artifacts/round_0/scored_repair_candidates.jsonl \
  --output artifacts/round_0/replay.jsonl
```

```bash
python scripts/train_sft.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --replay artifacts/round_0/replay.jsonl \
  --output-dir artifacts/round_0/sft
```

```bash
python scripts/make_pfpo_groups.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/pfpo_groups.unscored.jsonl
```

```bash
python scripts/score_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.unscored.jsonl \
  --output artifacts/round_0/pfpo_groups.scored.jsonl
```

```bash
python scripts/train_pfpo.py \
  --backend mock \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.scored.jsonl \
  --sft-checkpoint artifacts/round_0/sft \
  --output-dir artifacts/round_0/pfpo
```

```bash
python scripts/evaluate.py \
  --replay artifacts/round_0/replay.jsonl \
  --output results/smoke_metrics.json
```

## Step 5. Download Real Datasets

```bash
python scripts/download_datasets.py \
  --datasets agentharm atbench agentdojo \
  --output-dir data/raw
```

## Step 6. Run with a Real HF Model

Use `--backend hf` instead of `--backend mock` for generation and training commands.

Real runs require GPU-compatible PyTorch, Hugging Face access, benchmark rollout logs, and official benchmark verifiers.
