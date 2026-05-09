#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.config import load_yaml
from fate.utils.io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT repair warmup.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fate.training.sft import CompletionOnlyExample, make_trainer

    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
    )
    lora_cfg = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        ),
    )
    examples = [
        CompletionOnlyExample(prompt=row["prompt"], target=row["target"])
        for row in read_jsonl(args.replay)
    ]
    trainer = make_trainer(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        output_dir=args.output_dir,
        max_length=int(model_cfg["max_prompt_tokens"]) + int(model_cfg["max_new_tokens"]),
        training_args=config["sft"],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
