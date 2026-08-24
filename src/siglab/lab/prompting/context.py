from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from siglab.lab.archive import SignalArchive
from siglab.lab.task import DiscoveryTask


@dataclass(frozen=True)
class PromptRole:
    name: str
    instructions: str


@dataclass(frozen=True)
class PromptRenderResult:
    system_message: str
    user_message: str
    cache: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptContext:
    task: DiscoveryTask
    generation: int
    max_generations: int
    role: PromptRole
    archive: SignalArchive
    intent: str
    guidance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["archive"] = {
            "generation": self.archive.generation,
            "n_evaluated": len(self.archive.evaluated),
            "n_considered": len(self.archive.considered),
            "n_critic_notes": len(self.archive.critic_notes),
            "n_moderator_decisions": len(self.archive.moderator_decisions),
            "n_parse_diagnostics": len(self.archive.parse_diagnostics),
            "n_validation_diagnostics": len(self.archive.validation_diagnostics),
        }
        return data
