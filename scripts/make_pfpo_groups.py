#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.agents.hf_policy import HFPolicyClient
from fate.config import load_yaml
from fate.mock import MockPolicyClient
from fate.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample grouped PFPO completions.")
    parser.add_argument("--backend", choices=["mock", "hf"], default="mock")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    model_cfg = config["model"]
    gen_cfg = config["generation"]
    if args.backend == "hf":
        client = HFPolicyClient.from_pretrained(
            name_or_path=model_cfg["name_or_path"],
            torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            attn_implementation=model_cfg.get("attn_implementation"),
        )
    else:
        client = MockPolicyClient()
    rows = read_jsonl(args.prompts)
    outputs = []
    batch_size = int(config["evolution"].get("proposal_batch_size", 4))
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        groups = client.generate(
            [row["prompt"] for row in batch],
            max_new_tokens=int(model_cfg["max_new_tokens"]),
            temperature=float(gen_cfg["temperature"]),
            top_p=float(gen_cfg["top_p"]),
            num_return_sequences=int(gen_cfg["pfpo_group_size"]),
        )
        for row, completions in zip(batch, groups, strict=True):
            outputs.append(
                {
                    "prompt_id": row["failure_id"],
                    "prompt": row["prompt"],
                    "completions": [
                        {"text": completion.text, "reward": None} for completion in completions
                    ],
                    "metadata": {"mode": row["mode"], **dict(row.get("metadata", {}))},
                }
            )
    write_jsonl(args.output, outputs)
    print(f"wrote {len(outputs)} PFPO groups to {args.output}")


if __name__ == "__main__":
    main()
