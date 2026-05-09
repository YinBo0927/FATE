#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from fate.config import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFPO refinement entry point.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--sft-checkpoint", required=True)
    parser.add_argument("--reference-checkpoint")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from fate.training.offline_pfpo import PFPOGroupDataset, train_offline_pfpo

    model_cfg = config["model"]
    pfpo_cfg = config["pfpo"]
    trust_remote_code = bool(model_cfg.get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(args.sft_checkpoint, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_causal_model(args.sft_checkpoint, trust_remote_code=trust_remote_code)
    old_model = _load_causal_model(args.sft_checkpoint, trust_remote_code=trust_remote_code)
    ref_checkpoint = args.reference_checkpoint or model_cfg["name_or_path"]
    ref_model = _load_causal_model(ref_checkpoint, trust_remote_code=trust_remote_code)

    dataset = PFPOGroupDataset.from_jsonl(
        args.groups,
        tokenizer=tokenizer,
        max_length=int(model_cfg["max_prompt_tokens"]) + int(model_cfg["max_new_tokens"]),
    )
    train_offline_pfpo(
        model=model,
        old_model=old_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        output_dir=args.output_dir,
        learning_rate=float(pfpo_cfg["learning_rate"]),
        clip_epsilon=float(pfpo_cfg["clip_epsilon"]),
        kl_beta=float(pfpo_cfg["kl_beta"]),
        batch_size=int(pfpo_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(pfpo_cfg["gradient_accumulation_steps"]),
        max_steps=int(pfpo_cfg["max_steps"]),
    )


def _load_causal_model(path: str, *, trust_remote_code: bool):
    from transformers import AutoModelForCausalLM

    adapter_config = Path(path) / "adapter_config.json"
    if adapter_config.exists():
        try:
            from peft import AutoPeftModelForCausalLM

            return AutoPeftModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=trust_remote_code,
            )
        except ImportError:
            pass
    return AutoModelForCausalLM.from_pretrained(path, trust_remote_code=trust_remote_code)


if __name__ == "__main__":
    main()
