#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge JSONL files in order.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output.open("w", encoding="utf-8") as dst:
        for input_path in args.inputs:
            path = Path(input_path)
            with path.open("r", encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        dst.write(line if line.endswith("\n") else f"{line}\n")
                        total += 1
    print(f"merged {total} rows to {output}")


if __name__ == "__main__":
    main()
