"""Prompt context passed between Reflector and Proposer."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ArchiveSummary:
    """The Reflector's output. Captures what should be shown to a Proposer."""
    text: str                    # the prose summary / instructions
    generation: int              # current generation index (0-based)
    max_generations: int         # total planned generations (for the budget line)
