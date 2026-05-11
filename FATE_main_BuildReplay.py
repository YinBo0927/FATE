#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FATE repair replay from verifier-scored files.")
    parser.add_argument("--config", default="configs/qwen3_8b_fate.yaml")
    parser.add_argument("--round-dir", default="artifacts/round_0")
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round_dir = Path(args.round_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    rollouts = round_dir / "dev_rollouts.scored.jsonl"
    verified_repairs = round_dir / "repair_candidates.verified.jsonl"
    require_file(rollouts)

    failures = round_dir / "failures.jsonl"
    prompts = round_dir / "repair_prompts.jsonl"
    unverified_repairs = round_dir / "repair_candidates.unverified.jsonl"
    scored_repairs = round_dir / "scored_repair_candidates.jsonl"
    replay = round_dir / "replay.jsonl"

    run([sys.executable, "scripts/collect_failures.py", "--rollouts", str(rollouts), "--output", str(failures)])
    run([sys.executable, "scripts/make_repair_prompts.py", "--failures", str(failures), "--output", str(prompts)])

    if not verified_repairs.exists():
        if not unverified_repairs.exists():
            run([
                sys.executable,
                "scripts/generate_repairs.py",
                "--config",
                args.config,
                "--prompts",
                str(prompts),
                "--output",
                str(unverified_repairs),
            ])
        print(f"created {unverified_repairs}")
        print(f"score repair candidates with official verifiers and save them to {verified_repairs}")
        return

    run([sys.executable, "scripts/score_repairs.py", "--candidates", str(verified_repairs), "--output", str(scored_repairs)])
    run([
        sys.executable,
        "scripts/build_replay.py",
        "--config",
        args.config,
        "--failures",
        str(failures),
        "--candidates",
        str(scored_repairs),
        "--output",
        str(replay),
    ])
    run([sys.executable, "scripts/evaluate.py", "--replay", str(replay), "--output", str(results_dir / "replay_metrics.json")])


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing required file: {path}")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
