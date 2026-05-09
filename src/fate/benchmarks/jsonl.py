from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from fate.schema import FailureRecord
from fate.utils.io import read_jsonl


@dataclass
class JsonlFailureAdapter:
    path: Path

    def iter_failures(self) -> Iterable[FailureRecord]:
        for row in read_jsonl(self.path):
            yield FailureRecord.from_mapping(row)

