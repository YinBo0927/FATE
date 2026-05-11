#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets and output folders for FATE.")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--round-dir", default="artifacts/round_0")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["agentharm", "atbench", "agentdojo"],
        choices=["agentharm", "atbench", "agentdojo"],
    )
    parser.add_argument("--skip-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.round_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        cmd = [
            sys.executable,
            "scripts/download_datasets.py",
            "--output-dir",
            args.data_dir,
            "--datasets",
            *args.datasets,
        ]
        run(cmd)

    print(f"prepared {args.data_dir}, {args.round_dir}, and {args.results_dir}")
    print("next: create artifacts/round_0/dev_rollouts.scored.jsonl with the official benchmark runners")


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
