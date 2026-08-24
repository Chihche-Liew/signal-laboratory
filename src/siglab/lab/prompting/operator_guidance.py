"""Shared prompt guidance for valid signal operator syntax."""

from __future__ import annotations


def format_operator_arity_guard() -> str:
    return "\n".join([
        "OPERATOR ARITY GUARD",
        (
            "- Binary expression operators take exactly two expressions: "
            "RATIO(a, b), ADD(a, b), SUB(a, b), MUL(a, b), "
            "SUMRANK(a, b), MIN(a, b), MAX(a, b), COALESCE(a, b)."
        ),
        (
            "- Unary expression operators take exactly one expression, except "
            "INDICATOR(x, threshold): RANK(x), ZSCORE(x), IND_ADJ(x), "
            "IND_ZSCORE(x), IND_RANK(x), ACCEL(x), ABS(x), SIGN(x), "
            "LOG(x), INV(x), NEG(x)."
        ),
        (
            "- Temporal/window operators take one expression and at most one "
            "numeric parameter: GROWTH(x, 1), DELTA(x, 1), LAG(x, 1), "
            "MA(x, 3), TREND(x, 5), VOL(x, 5), TS_MIN(x, 5), TS_MAX(x, 5), "
            "TS_SUM(x, 5), TS_COUNT(x, 5), TS_RANK(x, 5)."
        ),
        (
            "- Temporal/window operators may wrap nested expressions, for "
            "example ACCEL(RATIO(xrd, sale)), "
            "GROWTH(RATIO(SUB(ib, oancf), at), 1), and "
            "MA(RATIO(ib, at), 3)."
        ),
        (
            "- WINSOR takes one expression and at most one symmetric tail "
            "percentile: WINSOR(x, 1). Do not write "
            "WINSOR(x, 0.01, 0.99)."
        ),
        (
            "- SUMRANK accepts exactly two expressions. To combine more than "
            "two ranked channels, nest binary calls: SUMRANK(a, "
            "SUMRANK(b, c))."
        ),
        (
            "- INDICATOR(x, threshold) returns 1 if x > threshold else 0, "
            "where threshold is a numeric constant: INDICATOR(x, 0)."
        ),
        (
            "- Do not write SUMRANK(RANK(a), RANK(b)) — SUMRANK already "
            "ranks its inputs; pass raw expressions."
        ),
    ])
