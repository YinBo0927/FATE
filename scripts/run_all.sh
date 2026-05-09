#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/qwen3_8b_fate.yaml}"
ROUND_DIR="${ROUND_DIR:-artifacts/round_0}"
DATA_DIR="${DATA_DIR:-data/raw}"
RESULTS_DIR="${RESULTS_DIR:-results}"

mkdir -p "$ROUND_DIR" "$RESULTS_DIR"

python scripts/download_datasets.py --datasets sample --output-dir "$DATA_DIR"
python scripts/collect_failures.py \
  --tasks "$DATA_DIR/sample/tasks.jsonl" \
  --output "$ROUND_DIR/failures.jsonl"
python scripts/make_repair_prompts.py \
  --failures "$ROUND_DIR/failures.jsonl" \
  --output "$ROUND_DIR/repair_prompts.jsonl"
python scripts/generate_repairs.py \
  --backend mock \
  --config "$CONFIG" \
  --prompts "$ROUND_DIR/repair_prompts.jsonl" \
  --output "$ROUND_DIR/repair_candidates.jsonl"
python scripts/score_repairs.py \
  --candidates "$ROUND_DIR/repair_candidates.jsonl" \
  --output "$ROUND_DIR/scored_repair_candidates.jsonl"
python scripts/build_replay.py \
  --config "$CONFIG" \
  --failures "$ROUND_DIR/failures.jsonl" \
  --candidates "$ROUND_DIR/scored_repair_candidates.jsonl" \
  --output "$ROUND_DIR/replay.jsonl"
python scripts/train_sft.py \
  --backend mock \
  --config "$CONFIG" \
  --replay "$ROUND_DIR/replay.jsonl" \
  --output-dir "$ROUND_DIR/sft"
python scripts/make_pfpo_groups.py \
  --backend mock \
  --config "$CONFIG" \
  --prompts "$ROUND_DIR/repair_prompts.jsonl" \
  --output "$ROUND_DIR/pfpo_groups.unscored.jsonl"
python scripts/score_pfpo_groups.py \
  --config "$CONFIG" \
  --groups "$ROUND_DIR/pfpo_groups.unscored.jsonl" \
  --output "$ROUND_DIR/pfpo_groups.scored.jsonl"
python scripts/train_pfpo.py \
  --backend mock \
  --config "$CONFIG" \
  --groups "$ROUND_DIR/pfpo_groups.scored.jsonl" \
  --sft-checkpoint "$ROUND_DIR/sft" \
  --output-dir "$ROUND_DIR/pfpo"
python scripts/evaluate.py \
  --replay "$ROUND_DIR/replay.jsonl" \
  --output "$RESULTS_DIR/smoke_metrics.json"

echo "Pipeline complete. Metrics: $RESULTS_DIR/smoke_metrics.json"
