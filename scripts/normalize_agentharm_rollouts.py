#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from fate.utils.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AgentHarm Inspect logs to FATE rollout JSONL.")
    parser.add_argument("--logdir", required=True, help="Inspect log directory from AgentHarm.")
    parser.add_argument("--output", required=True, help="Output FATE scored-rollout JSONL path.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--mode", choices=["harmful_request", "benign"], default="harmful_request")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for sample, source_log in _iter_inspect_samples(Path(args.logdir)):
        row = _normalize_sample(sample, source_log, args)
        if row is not None:
            rows.append(row)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} AgentHarm rollouts to {args.output}")


def _iter_inspect_samples(logdir: Path) -> Iterable[tuple[dict[str, Any], Path]]:
    try:
        from inspect_ai.log import read_eval_log_samples
    except ImportError as exc:
        raise ImportError(
            "normalize_agentharm_rollouts.py requires inspect-evals/inspect-ai. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    for path in sorted([*logdir.rglob("*.eval"), *logdir.rglob("*.json")]):
        try:
            samples = read_eval_log_samples(
                str(path),
                all_samples_required=False,
                resolve_attachments="core",
            )
            for sample in samples:
                yield _to_plain(sample), path
        except Exception as exc:
            print(f"skip {path}: {exc}")


def _normalize_sample(
    sample: dict[str, Any],
    source_log: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    metadata = dict(sample.get("metadata") or {})
    sample_id = str(sample.get("id") or metadata.get("behavior_id") or metadata.get("id") or "")
    if not sample_id:
        sample_id = _slug(source_log.stem)

    task = _content_to_text(sample.get("input")) or str(metadata.get("behavior") or sample_id)
    trajectory = json.dumps(_trajectory_payload(sample), ensure_ascii=False)
    score_value = _extract_combined_score(sample)
    refusal_value = _extract_refusal_score(sample)

    if args.mode == "harmful_request":
        security = 1.0 - score_value
        utility = 1.0
        over_refusal = refusal_value
        trajectory_control = 1.0 - score_value
    else:
        security = 1.0
        utility = score_value
        over_refusal = 1.0 - refusal_value
        trajectory_control = score_value

    return {
        "failure_id": _make_id("agentharm", args.split, sample_id),
        "task": task,
        "failed_trajectory": trajectory,
        "mode": args.mode,
        "scores": {
            "security": _clip01(security),
            "utility": _clip01(utility),
            "over_refusal": _clip01(over_refusal),
            "trajectory_control": _clip01(trajectory_control),
        },
        "verifier_feedback": {
            "security": f"AgentHarm score={score_value}",
            "utility": f"AgentHarm score={score_value}",
            "over_refusal": f"AgentHarm refusal={refusal_value}",
            "trajectory_control": f"AgentHarm score={score_value}",
        },
        "metadata": {
            "benchmark": "agentharm",
            "split": args.split,
            "source_log": str(source_log),
            **metadata,
        },
    }


def _trajectory_payload(sample: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "input": sample.get("input"),
        "messages": sample.get("messages"),
        "output": sample.get("output"),
        "scores": sample.get("scores"),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _extract_combined_score(sample: dict[str, Any]) -> float:
    value = _find_score_value(sample.get("scores"), preferred=("score", "combined_scorer"))
    return _clip01(value)


def _extract_refusal_score(sample: dict[str, Any]) -> float:
    value = _find_score_value(sample.get("scores"), preferred=("refusal", "valid_refusal"))
    return _clip01(value)


def _find_score_value(scores: Any, preferred: tuple[str, ...]) -> float:
    plain = _to_plain(scores)
    if isinstance(plain, dict):
        for key in preferred:
            found = _find_key(plain, key)
            if found is not None:
                return _coerce_float(found)
        for value in plain.values():
            if isinstance(value, dict) and "value" in value:
                return _coerce_float(value["value"])
            if isinstance(value, (int, float, bool, str)):
                return _coerce_float(value)
    return 0.0


def _find_key(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            nested = value[key]
            if isinstance(nested, dict) and "value" in nested:
                return nested["value"]
            return nested
        for nested in value.values():
            found = _find_key(nested, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_key(nested, key)
            if found is not None:
                return found
    return None


def _coerce_float(value: Any) -> float:
    if isinstance(value, dict):
        if "score" in value:
            return _coerce_float(value["score"])
        if "value" in value:
            return _coerce_float(value["value"])
        if "refusal" in value:
            return _coerce_float(value["refusal"])
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text in {"yes", "true", "pass", "passed", "correct"}:
        return 1.0
    if text in {"no", "false", "fail", "failed", "incorrect", "none"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _content_to_text(content: Any) -> str:
    plain = _to_plain(content)
    if plain is None:
        return ""
    if isinstance(plain, str):
        return plain
    if isinstance(plain, list):
        parts = [_content_to_text(item) for item in plain]
        return "\n".join(part for part in parts if part)
    if isinstance(plain, dict):
        for key in ("text", "content", "input", "prompt"):
            if key in plain:
                text = _content_to_text(plain[key])
                if text:
                    return text
    return json.dumps(plain, ensure_ascii=False)


def _to_plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: _to_plain(nested) for key, nested in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    return value


def _make_id(*parts: object) -> str:
    return "-".join(_slug(str(part)) for part in parts if part is not None and str(part) != "")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower()


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


if __name__ == "__main__":
    main()
