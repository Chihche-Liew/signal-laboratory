"""Evaluator that delegates to the existing SignalEvaluator.evaluate_signal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from siglab.lab.archive import Proposal, EvaluatedSignal


@dataclass
class SignalAgentAdapter:
    """Wrap SignalEvaluator so it satisfies the Evaluator protocol.

    This is the canonical signal evaluation pipeline: monthly panel build,
    microcap/financial exclusions, value-weighted quintile sorts, FF5 alpha,
    and FMB regression.
    """
    agent: Any  # SignalEvaluator; typed loose to avoid a pytest-relevant import cycle
    exclude_financials: bool
    exclude_microcap: bool = True
    alpha_factor_model: int = 5

    def evaluate(self, proposals: list[Proposal], generation: int) -> list[EvaluatedSignal]:
        out: list[EvaluatedSignal] = []
        for p in proposals:
            theme_tag = f"{p.theme_a}_{p.theme_b}"
            kwargs = {
                "expression": p.expression,
                "name": p.name,
                "hypothesis": p.hypothesis,
                "theme": theme_tag,
                "exclude_financials": self.exclude_financials,
                "exclude_microcap": self.exclude_microcap,
                "alpha_factor_model": self.alpha_factor_model,
            }
            result = self.agent.evaluate_signal(**kwargs)
            out.append(EvaluatedSignal.from_proposal(
                p, generation=generation,
                fmb_tstat=result.fmb_tstat,
                ls_alpha=result.ls_alpha,
                ls_talpha=result.ls_talpha,
                ls_sharpe=result.ls_sharpe,
                coverage=result.coverage,
                error=result.error,
            ))
        return out
