from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from siglab.lab.archive import Proposal
from siglab.lab.task import DiscoveryTask


@dataclass(frozen=True)
class ParseDiagnostic:
    severity: str
    code: str
    message: str
    generation: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParsedProposalBatch:
    proposals: list[Proposal]
    diagnostics: list[ParseDiagnostic]
    raw_text: str


class ProposalParser(Protocol):
    def parse(
        self, text: str, *, task: DiscoveryTask, generation: int,
    ) -> ParsedProposalBatch:
        ...
