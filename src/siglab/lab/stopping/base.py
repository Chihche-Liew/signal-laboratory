"""StoppingRule interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

from siglab.lab.archive import SignalArchive


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str


class StoppingRule(Protocol):
    def should_stop(self, archive: SignalArchive, meta: dict[str, Any]) -> StopDecision: ...
