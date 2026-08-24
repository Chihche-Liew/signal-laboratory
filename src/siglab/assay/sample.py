"""Shared monthly sample construction for discovery and post-hoc tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from siglab.utils.panel import fill_var


DEFAULT_SAMPLE_START = "1963-07-01"
POSTHOC_RESULTS_SEMANTICS = "2026-07-monthly-universe-v2"


def normalize_monthly_panel(
    panel: pd.DataFrame,
    *,
    name: str = "panel",
) -> pd.DataFrame:
    """Return a copy indexed by canonical calendar month-end timestamps.

    CRSP rows use the final trading day while factor files commonly use the
    calendar month end. Normalizing before alignment makes those labels refer
    to the same economic month. Duplicate economic months are rejected rather
    than silently aggregated.
    """
    result = panel.copy(deep=False)
    try:
        if isinstance(result.index, pd.PeriodIndex):
            months = result.index.asfreq("M")
        else:
            months = pd.DatetimeIndex(result.index).to_period("M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have a datetime-like monthly index") from exc

    if months.has_duplicates:
        duplicates = months[months.duplicated()].unique().astype(str).tolist()
        raise ValueError(f"{name} has duplicate economic months: {duplicates[:5]}")

    result.index = months.to_timestamp("M")
    return result.sort_index()


def normalize_factor_panel(panel: pd.DataFrame, *, name: str) -> pd.DataFrame:
    """Normalize a factor file without using it to define the stock sample."""
    return normalize_monthly_panel(panel, name=name)


@dataclass(frozen=True)
class MonthlySample:
    """Canonical stock universe shared by discovery and robustness tests."""

    returns: pd.DataFrame
    market_equity: pd.DataFrame
    nyse: pd.DataFrame


@dataclass(frozen=True)
class PreparedSignal:
    """A signal and its formation-month inputs on a canonical sample."""

    signal: pd.DataFrame
    returns: pd.DataFrame
    market_equity: pd.DataFrame
    nyse: pd.DataFrame
    coverage: float


def build_monthly_sample(
    returns: pd.DataFrame,
    market_equity: pd.DataFrame,
    nyse: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp = DEFAULT_SAMPLE_START,
    end_date: str | pd.Timestamp | None = None,
) -> MonthlySample:
    """Build a contiguous monthly stock sample from CRSP panels."""
    ret = normalize_monthly_panel(returns, name="returns")
    me = normalize_monthly_panel(market_equity, name="market_equity")
    nyse_norm = normalize_monthly_panel(nyse, name="nyse")

    columns = ret.columns.intersection(me.columns)
    if len(columns) == 0:
        raise ValueError("returns and market_equity have no common securities")

    first = max(
        pd.Timestamp(start_date).to_period("M"),
        ret.index.min().to_period("M"),
        me.index.min().to_period("M"),
    )
    last = min(ret.index.max().to_period("M"), me.index.max().to_period("M"))
    if end_date is not None:
        last = min(last, pd.Timestamp(end_date).to_period("M"))
    if first > last:
        raise ValueError("monthly sample has no dates after applying bounds")

    dates = pd.period_range(first, last, freq="M").to_timestamp("M")
    return MonthlySample(
        returns=ret.reindex(index=dates, columns=columns),
        market_equity=me.reindex(index=dates, columns=columns),
        nyse=nyse_norm.reindex(index=dates, columns=columns).eq(True),
    )


def prepare_signal_panel(
    raw_signal: pd.DataFrame,
    sample: MonthlySample,
    *,
    financials: pd.DataFrame | None = None,
    exclude_financials: bool = True,
    exclude_microcap: bool = True,
) -> PreparedSignal:
    """Prepare one signal using formation-month sample eligibility.

    The signal and formation market equity are masked at month ``t``. The
    return panel remains untouched, so the downstream one-month lag naturally
    applies the month-t eligibility decision to the month-(t+1) return. This
    avoids looking at return-month size when deciding whether a stock belongs
    to the investable universe.
    """
    raw = normalize_monthly_panel(raw_signal, name="signal")
    aligned = raw.reindex(
        index=sample.returns.index,
        columns=sample.returns.columns,
    )
    coverage = float(aligned.notna().to_numpy().mean()) if aligned.size else 0.0
    signal = fill_var(aligned, sample.market_equity)
    # fill_var's median imputation is cross-sectional, not a license to
    # manufacture observations before a signal exists (or after it ends).
    signal = signal.where(aligned.notna().any(axis=1), axis=0)

    if exclude_financials and financials is not None:
        fin = normalize_monthly_panel(financials, name="financials").reindex(
            index=signal.index,
            columns=signal.columns,
        ).fillna(False)
        signal = signal.where(~fin.astype(bool))

    formation_me = sample.market_equity
    if exclude_microcap:
        nyse_me = sample.market_equity.where(sample.nyse)
        cutoff = nyse_me.quantile(0.1, axis=1)
        eligible = sample.market_equity.gt(cutoff, axis=0)
        signal = signal.where(eligible)
        formation_me = sample.market_equity.where(eligible)

    return PreparedSignal(
        signal=signal,
        returns=sample.returns,
        market_equity=formation_me,
        nyse=sample.nyse,
        coverage=coverage,
    )


def direction_matches(value: float | None, expected_sign: str) -> bool:
    """Whether a finite statistic has the declared economic direction."""
    try:
        statistic = float(value)
    except (TypeError, ValueError):
        return False
    if not np.isfinite(statistic):
        return False
    return statistic < 0 if expected_sign == "negative" else statistic > 0


def finite_or_none(value):
    """Convert scalar numeric values to JSON-safe finite Python scalars."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    return value
