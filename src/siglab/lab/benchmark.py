"""Benchmark metrics for comparing architectures.

Each architecture × pair run is summarized as one BenchmarkMetrics record;
the benchmark harness collects these records and emits a comparison table.

BenchmarkMetrics captures in-loop metrics only: raw success counts, token usage,
wall clock, stopping reason, and expression diversity. Stricter posthoc
metrics are populated later by reading completed experiment folders and
cross-referencing posthoc output files.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from siglab.lab.archive import SignalArchive


def _normalize_expression(expression: str) -> str:
    """Normalize expression text for exact duplicate checks."""
    return "".join(expression.lower().split())


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


@dataclass
class BenchmarkMetrics:
    """A comparable summary of one architecture × pair run."""
    architecture: str
    pair: str
    n_evaluated: int = 0
    n_raw_fmb_successes: int = 0
    raw_fmb_success_rate: float = 0.0
    # Posthoc jobs can populate these from completed robustness outputs.
    n_ff5_survivors: int = 0
    n_qfactor_survivors: int = 0
    n_cz_novel: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cached_input_tokens: int = 0
    wall_clock_sec: float = 0.0
    stopping_reason: str = ""
    stopping_generation: int = 0
    mean_pairwise_corr: float | None = None
    n_unique_expressions: int = 0
    n_duplicate_expressions: int = 0
    expression_uniqueness_rate: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def collect_metrics(
    *, architecture: str, pair: str, archive: SignalArchive,
    total_input_tokens: int, total_output_tokens: int,
    cached_input_tokens: int, wall_clock_sec: float,
    stopping_reason: str, stopping_generation: int,
) -> BenchmarkMetrics:
    """Compute a BenchmarkMetrics from a completed DiscoveryLoop run."""
    successful = archive.successful()
    n_evaluated = len(archive.evaluated)
    expressions = [
        _normalize_expression(signal.expression)
        for signal in archive.evaluated
        if signal.expression.strip()
    ]
    n_unique_expressions = len(set(expressions))
    n_duplicate_expressions = len(expressions) - n_unique_expressions
    expression_uniqueness_rate = _safe_rate(n_unique_expressions, len(expressions))
    return BenchmarkMetrics(
        architecture=architecture,
        pair=pair,
        n_evaluated=n_evaluated,
        n_raw_fmb_successes=len(successful),
        raw_fmb_success_rate=_safe_rate(len(successful), n_evaluated),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cached_input_tokens=cached_input_tokens,
        wall_clock_sec=wall_clock_sec,
        stopping_reason=stopping_reason,
        stopping_generation=stopping_generation,
        mean_pairwise_corr=None,
        n_unique_expressions=n_unique_expressions,
        n_duplicate_expressions=n_duplicate_expressions,
        expression_uniqueness_rate=expression_uniqueness_rate,
    )
