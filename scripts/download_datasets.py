#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import warnings
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import snapshot_download


AGENTHARM_REPO = "ai-safety-institute/AgentHarm"
ATBENCH_REPO = "AI45Research/ATBench"
AGENTDOJO_ZIP = "https://github.com/ethz-spylab/agentdojo/archive/refs/heads/main.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets used by this repo.")
    parser.add_argument("--output-dir", default="data/raw")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["agentharm", "atbench", "agentdojo"],
        choices=["agentharm", "atbench", "agentdojo"],
    )
    parser.add_argument("--hf-token", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "agentharm" in args.datasets:
        download_agentharm(output_dir / "agentharm", token=args.hf_token)
    if "atbench" in args.datasets:
        download_atbench(output_dir / "atbench", token=args.hf_token)
    if "agentdojo" in args.datasets:
        download_agentdojo(output_dir / "agentdojo")


def download_agentharm(output_dir: Path, token: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_repo = snapshot_download(
        repo_id=AGENTHARM_REPO,
        repo_type="dataset",
        local_dir=output_dir / "hub_snapshot",
        token=token,
    )
    _write_manifest(output_dir, AGENTHARM_REPO, local_repo)

    _try_export_hf_dataset(AGENTHARM_REPO, output_dir / "jsonl", token=token)


def download_atbench(output_dir: Path, token: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    local_repo = snapshot_download(
        repo_id=ATBENCH_REPO,
        repo_type="dataset",
        local_dir=output_dir / "hub_snapshot",
        token=token,
    )
    _write_manifest(output_dir, ATBENCH_REPO, local_repo)

    _try_export_hf_dataset(ATBENCH_REPO, output_dir / "jsonl", token=token)


def download_agentdojo(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "agentdojo-main.zip"
    urlretrieve(AGENTDOJO_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(output_dir)
    _write_manifest(output_dir, "ethz-spylab/agentdojo", str(output_dir / "agentdojo-main"))


def _export_dataset_dict(dataset: Dataset | DatasetDict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(dataset, DatasetDict):
        for split, split_dataset in dataset.items():
            split_dataset.to_json(output_dir / f"{split}.jsonl", force_ascii=False)
    else:
        dataset.to_json(output_dir / "data.jsonl", force_ascii=False)


def _try_export_hf_dataset(repo_id: str, output_dir: Path, token: str | None) -> None:
    try:
        dataset = load_dataset(repo_id, token=token)
    except Exception as exc:
        warnings.warn(
            f"Downloaded raw snapshot for {repo_id}, but load_dataset export failed: {exc}",
            stacklevel=2,
        )
        return
    _export_dataset_dict(dataset, output_dir)


def _write_manifest(output_dir: Path, source: str, local_path: str) -> None:
    manifest = {
        "source": source,
        "local_path": local_path,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
