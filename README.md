# FATE

## Step 1. Create Environment

```bash
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
```

## Step 2. Run the End-to-End Pipeline

```bash
bash scripts/run_all.sh
```

This creates sample data, collects failures, builds repair prompts, generates repairs, scores repairs, builds replay data, runs mock SFT/PFPO checkpoints, and writes metrics to:

```text
results/smoke_metrics.json
```

## Step 3. Download Real Datasets

```bash
python scripts/download_datasets.py --datasets agentharm atbench agentdojo --output-dir data/raw
```

## Step 4. Use a Real HF Model

Replace the mock backend with `hf` in the generation and training commands:

```bash
python scripts/generate_repairs.py --backend hf --config configs/qwen3_8b_fate.yaml --prompts artifacts/round_0/repair_prompts.jsonl --output artifacts/round_0/repair_candidates.jsonl
python scripts/train_sft.py --backend hf --config configs/qwen3_8b_fate.yaml --replay artifacts/round_0/replay.jsonl --output-dir artifacts/round_0/sft
python scripts/make_pfpo_groups.py --backend hf --config configs/qwen3_8b_fate.yaml --prompts artifacts/round_0/repair_prompts.jsonl --output artifacts/round_0/pfpo_groups.unscored.jsonl
python scripts/train_pfpo.py --backend hf --config configs/qwen3_8b_fate.yaml --groups artifacts/round_0/pfpo_groups.scored.jsonl --sft-checkpoint artifacts/round_0/sft --output-dir artifacts/round_0/pfpo
```

For real runs, configure GPU-compatible PyTorch, Hugging Face access, benchmark-specific rollout logs, and official benchmark verifiers.
