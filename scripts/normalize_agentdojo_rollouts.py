#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from fate.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgentDojo trace logs to FATE rollout JSONL.")
    parser.add_argument("--logdir", required=True, help="AgentDojo benchmark log directory.")
    parser.add_argument("--output", required=True, help="Output FATE scored-rollout JSONL path.")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--benchmark-version", default=None)
    parser.add_argument("--attack-mode", default="attacked_legitimate")
    parser.add_argument("--clean-mode", default="benign")
    parser.add_argument("--include-clean", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in sorted(Path(args.logdir).rglob("*.json")):
        record = _read_json(path)
        if not _looks_like_agentdojo_trace(record):
            continue
        attack_type = record.get("attack_type")
        is_clean = attack_type in (None, "none")
        if is_clean and not args.include_clean:
            continue
        rows.append(_normalize_trace(record, path, args, is_clean=is_clean))
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} AgentDojo rollouts to {args.output}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _looks_like_agentdojo_trace(record: dict[str, Any]) -> bool:
    required = {"suite_name", "user_task_id", "messages", "utility", "security"}
    return required.issubset(record)


def _normalize_trace(
    record: dict[str, Any],
    path: Path,
    args: argparse.Namespace,
    *,
    is_clean: bool,
) -> dict[str, Any]:
    suite_name = str(record["suite_name"])
    user_task_id = str(record["user_task_id"])
    injection_task_id = record.get("injection_task_id")
    mode = args.clean_mode if is_clean else args.attack_mode
    utility = bool(record.get("utility", False))
    security = bool(record.get("security", False))
    benchmark_version = args.benchmark_version or record.get("benchmark_version")
    messages = list(record.get("messages", []))

    return {
        "failure_id": _make_id("agentdojo", suite_name, user_task_id, injection_task_id or "none"),
        "task": _first_user_text(messages) or user_task_id,
        "failed_trajectory": json.dumps(messages, ensure_ascii=False),
        "mode": mode,
        "scores": {
            "security": float(security),
            "utility": float(utility),
            "over_refusal": float(utility),
            "trajectory_control": float(security),
        },
        "verifier_feedback": {
            "security": f"AgentDojo security={security}",
            "utility": f"AgentDojo utility={utility}",
            "over_refusal": "Derived from AgentDojo utility.",
            "trajectory_control": "Derived from AgentDojo security.",
        },
        "metadata": {
            "benchmark": "agentdojo",
            "split": args.split,
            "suite": suite_name,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "attack_type": record.get("attack_type"),
            "pipeline_name": record.get("pipeline_name"),
            "benchmark_version": benchmark_version,
            "duration": record.get("duration"),
            "error": record.get("error"),
            "source_log": str(path),
        },
    }


def _first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            text = _content_to_text(message.get("content"))
            if text:
                return text
    return ""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_content_to_text(item) for item in content]
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        for key in ("text", "content"):
            if key in content:
                return _content_to_text(content[key])
    return json.dumps(content, ensure_ascii=False)


def _make_id(*parts: object) -> str:
    return "-".join(_slug(str(part)) for part in parts if part is not None and str(part) != "")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


if __name__ == "__main__":
    main()
