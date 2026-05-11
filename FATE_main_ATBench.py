#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ATBench for external trajectory-safety evaluation.")
    parser.add_argument("--output", default="data/raw/atbench/test.jsonl")
    parser.add_argument("--config", default="ATBench")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable,
        "scripts/export_atbench.py",
        "--output",
        args.output,
        "--config",
        args.config,
        "--split",
        args.split,
    ])


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
