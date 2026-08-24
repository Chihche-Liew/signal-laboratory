"""Operator library for fundamental signal construction.

Defines the set of operations the LLM agent can compose to build signals.
Each operator takes accounting variable panels as input and returns a signal panel.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Operator:
    """Metadata for a signal construction operator."""
    name: str
    category: str          # ratio, growth, delta, cross_section, temporal, composite, math
    arity: int             # 1 = unary (one input), 2 = binary (two inputs)
    description: str       # For the LLM
    example: str           # Concrete example using Compustat mnemonics
    params: tuple[str, ...] = ()  # Named parameters (e.g., "lag", "window")


# ── Tier 1: Core Operators (always available) ────────────────────────────

RATIO = Operator(
    "RATIO", "ratio", 2,
    "Divide numerator by denominator to create a financial ratio. "
    "Observations where abs(denominator) <= 1e-10 become NaN; negative "
    "denominators are otherwise allowed and preserve sign.",
    "RATIO(SUB(revt, cogs), at) → gross profitability",
)

GROWTH = Operator(
    "GROWTH", "growth", 1,
    "Year-over-year percent change: (X_t - X_{t-lag}) / |X_{t-lag}|. "
    "Captures rate of expansion or contraction.",
    "GROWTH(at, 1) → asset growth rate",
    params=("lag",),
)

DELTA = Operator(
    "DELTA", "delta", 1,
    "Simple level change: X_t - X_{t-lag}. "
    "Use RATIO to normalize by assets or market cap.",
    "DELTA(dltt, 1) → change in long-term debt",
    params=("lag",),
)

IND_ADJ = Operator(
    "IND_ADJ", "cross_section", 1,
    "Industry-adjusted value: X_i minus the industry median at each "
    "year x 2-digit SIC group. Removes industry-level variation "
    "(important for ratios with strong industry effects).",
    "IND_ADJ(RATIO(sale, at)) → industry-adjusted asset turnover",
)

# ── Tier 2: Temporal Operators ───────────────────────────────────────────

MA = Operator(
    "MA", "temporal", 1,
    "Moving average over k years: mean(X_t, X_{t-1}, ..., X_{t-k+1}). "
    "Smooths noisy variables like earnings or R&D spending.",
    "MA(RATIO(ib, at), 3) → 3-year average ROA",
    params=("k",),
)

TREND = Operator(
    "TREND", "temporal", 1,
    "Linear trend slope of X over k years. "
    "Captures sustained improvement or deterioration.",
    "TREND(RATIO(sale, at), 5) → 5-year trend in asset turnover",
    params=("k",),
)

VOL = Operator(
    "VOL", "temporal", 1,
    "Rolling standard deviation over k years. "
    "Earnings volatility and cash flow volatility are quality indicators.",
    "VOL(RATIO(ib, at), 5) → earnings volatility",
    params=("k",),
)

ACCEL = Operator(
    "ACCEL", "temporal", 1,
    "Second difference: DELTA(DELTA(X)). Change in the change. "
    "Captures acceleration or deceleration of trends.",
    "ACCEL(RATIO(sale, at)) → acceleration of asset turnover",
)

LAG = Operator(
    "LAG", "temporal", 1,
    "Simple lag: X_{t-d}. Access prior period values.",
    "LAG(at, 1) → prior year total assets",
    params=("d",),
)

TS_MAX = Operator(
    "TS_MAX", "temporal", 1,
    "Trailing maximum over the past k years (window includes the current value). "
    "Compare the current level to TS_MAX for distance-from-high signals.",
    "RATIO(RATIO(ib, at), TS_MAX(RATIO(ib, at), 5)) → ROA relative to its 5-year high",
    params=("k",),
)

TS_MIN = Operator(
    "TS_MIN", "temporal", 1,
    "Trailing minimum over the past k years (window includes the current value). "
    "Compare the current level to TS_MIN for recovery-from-trough signals.",
    "SUB(RATIO(ib, at), TS_MIN(RATIO(ib, at), 5)) → ROA gain over its 5-year trough",
    params=("k",),
)

TS_RANK = Operator(
    "TS_RANK", "temporal", 1,
    "Percentile position of the current value within the firm's own trailing "
    "k-year window, in (0, 1]. Robust way to ask whether a ratio is at a "
    "multi-year high or low. NaN unless at least 2 window observations exist.",
    "TS_RANK(RATIO(ib, at), 5) → where current ROA sits in its own 5-year history",
    params=("k",),
)

TS_SUM = Operator(
    "TS_SUM", "temporal", 1,
    "Trailing sum over the past k years (window includes the current value). "
    "Useful for cumulative flows such as multi-year R&D or capex.",
    "RATIO(TS_SUM(xrd, 5), at) → 5-year cumulative R&D scaled by assets",
    params=("k",),
)

TS_COUNT = Operator(
    "TS_COUNT", "temporal", 1,
    "Count of non-missing observations in the firm's trailing k-year window. "
    "Use for persistence or availability checks, not as a standalone pricing signal.",
    "TS_COUNT(xrd, 5) → number of reported R&D observations in five years",
    params=("k",),
)

# ── Tier 3: Composition Operators ────────────────────────────────────────

ADD = Operator(
    "ADD", "composite", 2,
    "Sum of two variables: A + B.",
    "ADD(dltt, dlc) → total debt",
)

SUB = Operator(
    "SUB", "composite", 2,
    "Difference: A - B.",
    "SUB(revt, cogs) → gross profit",
)

MUL = Operator(
    "MUL", "composite", 2,
    "Product: A * B. Creates interaction terms.",
    "MUL(RATIO(ib, at), RATIO(sale, at)) → profitability × asset-turnover interaction",
)

SUMRANK = Operator(
    "SUMRANK", "composite", 2,
    "Ranks A and B independently within each year's cross-section "
    "(percentile rank in [0,1]) and returns rank(A) + rank(B) - 1, "
    "so the result lies in roughly (-1, 1], not [0, 1]. "
    "Pass raw expressions, not pre-ranked ones — do not wrap inputs in RANK(...). "
    "Use to combine two signals that you want to weight symmetrically (e.g., high profit AND low growth).",
    "SUMRANK(RATIO(SUB(revt, cogs), at), NEG(GROWTH(at, 1))) → cash-cow composite",
)

INDICATOR = Operator(
    "INDICATOR", "composite", 1,
    "Binary indicator: 1 if X > threshold, else 0. Threshold is a numeric "
    "constant, e.g. INDICATOR(x, threshold). "
    "Used for building composite scores (F-score style).",
    "INDICATOR(GROWTH(sale, 1), 0) → sales growth positive indicator",
    params=("threshold",),
)

COALESCE = Operator(
    "COALESCE", "composite", 2,
    "Missing-value fallback: A where A is present, otherwise B. "
    "B may be a constant or another expression. "
    "Useful when a variable is often unreported (e.g., xrd).",
    "COALESCE(xrd, 0) → R&D with missing values treated as zero",
)

# ── Tier 4: Mathematical ─────────────────────────────────────────────────

ABS_OP = Operator("ABS", "math", 1, "Absolute value.", "ABS(DELTA(ni, 1)) → magnitude of earnings change")
SIGN_OP = Operator("SIGN", "math", 1, "Sign function: +1, 0, or -1.", "SIGN(ib) → profit/loss indicator")
LOG_OP = Operator("LOG", "math", 1, "Natural logarithm (for positive variables).", "LOG(at) → log total assets (size)")
INV_OP = Operator("INV", "math", 1, "Reciprocal: 1/X. Flips a ratio.", "INV(RATIO(dltt, ceq)) → equity-to-debt from debt-to-equity")
NEG_OP = Operator("NEG", "math", 1, "Negate: -X. Flips the signal direction.", "NEG(GROWTH(at, 1)) → negated asset growth (high AG = low expected return)")
MIN_OP = Operator(
    "MIN", "math", 2,
    "Elementwise minimum of two expressions: min(A, B). "
    "Use for caps and conservative combinations. NaN in either input yields NaN.",
    "MIN(che, lct) → liquid assets capped at current liabilities",
)
MAX_OP = Operator(
    "MAX", "math", 2,
    "Elementwise maximum of two expressions: max(A, B). "
    "Use for floors, e.g. flooring a variable at zero. NaN in either input yields NaN.",
    "MAX(xrd, 0) → R&D expense floored at zero",
)

# ── Tier 5: Cross-Sectional ──────────────────────────────────────────────

ZSCORE = Operator(
    "ZSCORE", "cross_section", 1,
    "Within-year cross-sectional z-score: (X_i - mean_t) / std_t. "
    "Sensitive to outliers; prefer SUMRANK or IND_ADJ for robustness.",
    "ZSCORE(RATIO(ib, at)) → cross-sectionally z-scored ROA",
)

RANK = Operator(
    "RANK", "cross_section", 1,
    "Within-year cross-sectional percentile rank in (0, 1].",
    "RANK(RATIO(ceq, at)) → book-equity-to-assets percentile",
)

IND_ZSCORE = Operator(
    "IND_ZSCORE", "cross_section", 1,
    "Industry z-score: (X_i - industry_mean) / industry_std, grouped by "
    "year x 2-digit SIC. Stronger form of industry adjustment than IND_ADJ.",
    "IND_ZSCORE(RATIO(ib, at)) → industry-relative ROA z-score",
)

IND_RANK = Operator(
    "IND_RANK", "cross_section", 1,
    "Within-industry percentile rank in (0, 1] at each year "
    "(year × 2-digit SIC groups). Robust alternative to IND_ZSCORE "
    "for skewed ratios.",
    "IND_RANK(RATIO(ib, at)) → ROA percentile within industry",
)

WINSOR = Operator(
    "WINSOR", "cross_section", 1,
    "Within-year winsorize at the pth and (100-p)th cross-sectional percentiles.",
    "WINSOR(RATIO(ib, ceq), 1) → winsorized ROE",
    params=("p",),
)


# ── Operator Registry ────────────────────────────────────────────────────

OPERATOR_CATALOG: dict[str, Operator] = {
    # Tier 1 - Core
    "RATIO": RATIO, "GROWTH": GROWTH, "DELTA": DELTA,
    "IND_ADJ": IND_ADJ,
    # Tier 2 - Temporal
    "MA": MA, "TREND": TREND, "VOL": VOL, "ACCEL": ACCEL, "LAG": LAG, "TS_MIN": TS_MIN, "TS_MAX": TS_MAX, "TS_SUM": TS_SUM, "TS_COUNT": TS_COUNT, "TS_RANK": TS_RANK,
    # Tier 3 - Composition
    "ADD": ADD, "SUB": SUB, "MUL": MUL, "SUMRANK": SUMRANK, "INDICATOR": INDICATOR, "COALESCE": COALESCE,
    # Tier 4 - Math
    "ABS": ABS_OP, "SIGN": SIGN_OP, "LOG": LOG_OP, "INV": INV_OP, "NEG": NEG_OP, "MIN": MIN_OP, "MAX": MAX_OP,
    # Tier 5 - Cross-sectional
    "ZSCORE": ZSCORE, "IND_ZSCORE": IND_ZSCORE, "IND_RANK": IND_RANK, "WINSOR": WINSOR,
    "RANK": RANK,
}


def get_operators_by_category(category: str) -> list[Operator]:
    """Get all operators in a given category."""
    return [op for op in OPERATOR_CATALOG.values() if op.category == category]


def operator_summary() -> str:
    """Return a human-readable summary of the operator library."""
    lines = ["FUNDAMENTAL SIGNAL OPERATOR LIBRARY", "=" * 60]
    for cat in ["ratio", "growth", "delta", "cross_section",
                "temporal", "composite", "math"]:
        ops = get_operators_by_category(cat)
        if ops:
            lines.append(f"\n{cat.upper()} ({len(ops)} operators):")
            for op in ops:
                params = f" [{', '.join(op.params)}]" if op.params else ""
                lines.append(f"  {op.name:12s} arity={op.arity}{params}")
                lines.append(f"               Example: {op.example}")
    return "\n".join(lines)
