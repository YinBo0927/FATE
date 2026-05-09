# FATE Reproduction

## Step 1. Create Environment

```bash
conda create -n fate python=3.10 -y
conda activate fate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
```

## Step 2. Download Datasets

```bash
python scripts/download_datasets.py --output-dir data/raw
```

To download only selected datasets:

```bash
python scripts/download_datasets.py --datasets agentharm atbench --output-dir data/raw
```

## Step 3. Prepare Failures

Convert rollout failures into:

```text
artifacts/round_0/failures.jsonl
```

See the required schema in [docs/reproduction.md](docs/reproduction.md).

## Step 4. Build Repair Prompts

```bash
python scripts/make_repair_prompts.py \
  --failures artifacts/round_0/failures.jsonl \
  --output artifacts/round_0/repair_prompts.jsonl
```

## Step 5. Generate Repairs

```bash
python scripts/generate_repairs.py \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/repair_candidates.jsonl
```

## Step 6. Score Repairs

Use the corresponding benchmark verifier to add the `scores` field to `repair_candidates.jsonl`, then save:

```text
artifacts/round_0/scored_repair_candidates.jsonl
```

## Step 7. Build Replay

```bash
python scripts/build_replay.py \
  --config configs/qwen3_8b_fate.yaml \
  --failures artifacts/round_0/failures.jsonl \
  --candidates artifacts/round_0/scored_repair_candidates.jsonl \
  --output artifacts/round_0/replay.jsonl
```

## Step 8. Train

```bash
python scripts/train_sft.py \
  --config configs/qwen3_8b_fate.yaml \
  --replay artifacts/round_0/replay.jsonl \
  --output-dir artifacts/round_0/sft

python scripts/make_pfpo_groups.py \
  --config configs/qwen3_8b_fate.yaml \
  --prompts artifacts/round_0/repair_prompts.jsonl \
  --output artifacts/round_0/pfpo_groups.unscored.jsonl
```

After adding verifier rewards to `pfpo_groups.unscored.jsonl`, run:

```bash
python scripts/train_pfpo.py \
  --config configs/qwen3_8b_fate.yaml \
  --groups artifacts/round_0/pfpo_groups.scored.jsonl \
  --sft-checkpoint artifacts/round_0/sft \
  --output-dir artifacts/round_0/pfpo
```
