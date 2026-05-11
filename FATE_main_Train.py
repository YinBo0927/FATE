#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FATE SFT and PFPO training stages.")
    parser.add_argument("--config", default="configs/qwen3_8b_fate.yaml")
    parser.add_argument("--round-dir", default="artifacts/round_0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_dir = Path(args.round_dir)
    replay = round_dir / "replay.jsonl"
    prompts = round_dir / "repair_prompts.jsonl"
    sft_dir = round_dir / "sft"
    pfpo_unscored = round_dir / "pfpo_groups.unscored.jsonl"
    pfpo_verified = round_dir / "pfpo_groups.verified.jsonl"
    pfpo_scored = round_dir / "pfpo_groups.scored.jsonl"
    pfpo_dir = round_dir / "pfpo"

    require_file(replay)
    require_file(prompts)

    if not sft_dir.exists():
        run([
            sys.executable,
            "scripts/train_sft.py",
            "--config",
            args.config,
            "--replay",
            str(replay),
            "--output-dir",
            str(sft_dir),
        ])
    if not pfpo_unscored.exists():
        run([
            sys.executable,
            "scripts/make_pfpo_groups.py",
            "--config",
            args.config,
            "--model-name-or-path",
            str(sft_dir),
            "--prompts",
            str(prompts),
            "--output",
            str(pfpo_unscored),
        ])

    if not pfpo_verified.exists():
        print(f"created {pfpo_unscored}")
        print(f"score PFPO groups with official verifiers and save them to {pfpo_verified}")
        return

    run([
        sys.executable,
        "scripts/score_pfpo_groups.py",
        "--config",
        args.config,
        "--groups",
        str(pfpo_verified),
        "--output",
        str(pfpo_scored),
    ])
    run([
        sys.executable,
        "scripts/train_pfpo.py",
        "--config",
        args.config,
        "--groups",
        str(pfpo_scored),
        "--sft-checkpoint",
        str(sft_dir),
        "--output-dir",
        str(pfpo_dir),
    ])


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
