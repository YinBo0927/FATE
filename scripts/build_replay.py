#!/usr/bin/env python
from __future__ import annotations

import argparse

from fate.config import load_yaml
from fate.evolution import build_replay_from_files
from fate.pareto import ParetoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Pareto-front repair replay.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--failures", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-n", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pareto_config = ParetoConfig.from_mapping(config["pareto"])
    replay = build_replay_from_files(
        failures_path=args.failures,
        candidates_path=args.candidates,
        pareto_config=pareto_config,
        output_path=args.output,
        top_n=args.top_n,
    )
    print(f"wrote {len(replay)} replay examples to {args.output}")


if __name__ == "__main__":
    main()

