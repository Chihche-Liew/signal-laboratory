"""Evaluator interface."""
from __future__ import annotations

from typing import Protocol

from siglab.lab.archive import Proposal, EvaluatedSignal


class Evaluator(Protocol):
    def evaluate(self, proposals: list[Proposal], generation: int) -> list[EvaluatedSignal]: ...
