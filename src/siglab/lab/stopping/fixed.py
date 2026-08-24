"""Fixed-generation-count stopping rule."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from siglab.lab.archive import SignalArchive
from siglab.lab.stopping.base import StopDecision


@dataclass
class FixedGens:
    """Stop when generation count reaches n."""
    n: int

    def should_stop(self, archive: SignalArchive, meta: dict[str, Any]) -> StopDecision:
        gen = int(meta.get("generation", archive.generation))
        if gen >= self.n:
            return StopDecision(stop=True, reason=f"fixed cap: {gen} >= {self.n}")
        return StopDecision(stop=False, reason=f"continue: {gen} < {self.n}")
