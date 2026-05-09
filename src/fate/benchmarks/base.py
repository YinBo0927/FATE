from __future__ import annotations

from typing import Iterable, Protocol

from fate.schema import FailureRecord


class BenchmarkAdapter(Protocol):
    def iter_failures(self) -> Iterable[FailureRecord]:
        """Yield verifier-scored failures from an exported benchmark log."""

