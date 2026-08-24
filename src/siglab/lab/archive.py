"""Shared signal archive and associated dataclasses.

This is the blackboard that every component in a DiscoveryLoop writes to
and reads from. All dataclasses are deliberately simple (no methods beyond
basic constructors and one from_proposal factory) so they serialize to JSON
cleanly.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass
class Proposal:
    """A signal proposed by an agent but not yet evaluated."""
    name: str
    expression: str
    hypothesis: str
    expected_sign: Literal["positive", "negative"]
    reasoning: str
    interaction_type: Literal["composite", "conditional", "ratio", "novel"]
    theme_a: str
    theme_b: str
    author: str  # "proposer" | "proposer_1" | "peer_2" | etc


@dataclass
class EvaluatedSignal:
    """A proposal that has gone through the Evaluator."""
    name: str
    expression: str
    hypothesis: str
    expected_sign: str
    reasoning: str
    interaction_type: str
    theme_a: str
    theme_b: str
    author: str
    generation: int
    fmb_tstat: float | None = None
    ls_alpha: float | None = None
    ls_talpha: float | None = None
    ls_sharpe: float | None = None
    coverage: float | None = None
    error: str | None = None

    @classmethod
    def from_proposal(cls, p: Proposal, generation: int, **metrics) -> "EvaluatedSignal":
        return cls(
            name=p.name, expression=p.expression, hypothesis=p.hypothesis,
            expected_sign=p.expected_sign, reasoning=p.reasoning,
            interaction_type=p.interaction_type,
            theme_a=p.theme_a, theme_b=p.theme_b, author=p.author,
            generation=generation, **metrics,
        )


@dataclass
class CriticNote:
    """A critic agent's verdict on a specific proposal."""
    generation: int
    target_name: str
    verdict: Literal["accept", "flag"]
    score: int  # 1–5
    rationale: str


@dataclass
class ModeratorNote:
    """A moderator's selection rationale for a generation."""
    generation: int
    selected_names: list[str]
    rationale: str


@dataclass
class SignalArchive:
    """Shared, mutable state for a DiscoveryLoop run."""
    evaluated: list[EvaluatedSignal] = field(default_factory=list)
    considered: list[Proposal] = field(default_factory=list)
    critic_notes: list[CriticNote] = field(default_factory=list)
    moderator_decisions: list[ModeratorNote] = field(default_factory=list)
    parse_diagnostics: list[Any] = field(default_factory=list)
    validation_diagnostics: list[Any] = field(default_factory=list)
    generation: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()

    def add_evaluated(self, signals: list[EvaluatedSignal]) -> None:
        with self._lock:
            self.evaluated.extend(signals)

    def add_considered(self, proposals: list[Proposal]) -> None:
        with self._lock:
            self.considered.extend(proposals)

    def add_critic_notes(self, notes: list[CriticNote]) -> None:
        with self._lock:
            self.critic_notes.extend(notes)

    def add_moderator_note(self, note: ModeratorNote) -> None:
        with self._lock:
            self.moderator_decisions.append(note)

    def add_parse_diagnostics(self, diagnostics: list[Any]) -> None:
        with self._lock:
            self.parse_diagnostics.extend(diagnostics)

    def add_validation_diagnostics(self, diagnostics: list[Any]) -> None:
        with self._lock:
            self.validation_diagnostics.extend(diagnostics)

    def successful(self, threshold: float = 1.96) -> list[EvaluatedSignal]:
        """Return evaluated signals that are significant and have the predicted sign."""
        out = []
        for s in self.evaluated:
            if s.error is not None or s.fmb_tstat is None:
                continue
            if not np.isfinite(s.fmb_tstat):
                # NaN t with error=None happens when FMB has too few valid
                # months: abs(NaN) <= thr is False and NaN > 0 is False, so
                # NaN would otherwise register as a successful NEGATIVE
                # signal (P1-1).
                continue
            if abs(s.fmb_tstat) <= threshold:
                continue
            expected_pos = s.expected_sign == "positive"
            actual_pos = s.fmb_tstat > 0
            if expected_pos == actual_pos:
                out.append(s)
        return out
