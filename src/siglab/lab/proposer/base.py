"""Proposer protocol."""
from __future__ import annotations

from typing import Any, Protocol

from siglab.lab.archive import SignalArchive
from siglab.lab.proposer._result import ProposerResult
from siglab.lab.task import DiscoveryTask


class Proposer(Protocol):
    def propose(
        self, *,
        archive: SignalArchive,
        task: DiscoveryTask,
        generation: int,
        n: int,
        intent: str | None = None,
        progress: Any | None = None,
    ) -> ProposerResult:
        """Produce K proposals for this generation.

        Proposers manage their own multi-turn state internally (so
        single-agent can preserve its thinking chain and Debate can preserve
        per-persona chains without leaking state into the Loop).
        """
        ...
