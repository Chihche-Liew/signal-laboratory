"""Reflector protocol."""
from __future__ import annotations

from typing import Protocol

from siglab.lab.archive import SignalArchive
from siglab.lab.context.base import ArchiveSummary
from siglab.lab.task import DiscoveryTask


class Reflector(Protocol):
    def summarize(
        self, *, archive: SignalArchive, task: DiscoveryTask,
        generation: int,
        intent: str | None = None,
    ) -> ArchiveSummary: ...
