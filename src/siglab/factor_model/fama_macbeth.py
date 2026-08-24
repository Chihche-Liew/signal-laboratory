"""Fama-MacBeth (1973) cross-sectional regression.

Mirrors runFamaMacBeth.m from AssayingAnomalies.
Two-stage procedure:
  Stage 1: Monthly cross-sectional regressions
  Stage 2: Time-series average of monthly coefficients with Newey-West SEs
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from siglab.utils.regression import nanols, nanwls
from siglab.utils.stats import newey_west_t, trim


@dataclass
class FMBResult:
    """Result of Fama-MacBeth regression."""
    beta: np.ndarray           # (k,) average regression coefficients
    tstat: np.ndarray          # (k,) Newey-West t-statistics
    ts_beta: np.ndarray        # (T, k) time-series of monthly coefficients
    r2_cs: np.ndarray          # (T,) cross-sectional R² by month
    char_names: list[str]      # characteristic names


def _validate_panel_labels(panel: pd.DataFrame, *, name: str) -> None:
    """Validate labels before any set operation can discard duplicates."""
    if panel.index.has_duplicates:
        raise ValueError(f"{name} dates must be unique")
    if panel.columns.has_duplicates:
        raise ValueError(f"{name} security labels must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError(
            f"{name} dates must be sorted ascending — positional lags would "
            "otherwise look ahead"
        )
    if isinstance(panel.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        months = (
            panel.index.asfreq("M")
            if isinstance(panel.index, pd.PeriodIndex)
            else panel.index.to_period("M")
        )
        if months.has_duplicates:
            raise ValueError(f"{name} must have at most one row per calendar month")


def _validate_contiguous_months(index: pd.Index) -> None:
    if not isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) or len(index) <= 1:
        return
    months = index.asfreq("M") if isinstance(index, pd.PeriodIndex) else index.to_period("M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    if not months.equals(expected):
        raise ValueError(
            "panel dates must be contiguous calendar months; reindex missing "
            "months with NaN rows before applying positional lags"
        )


def run_fama_macbeth(
    chars: dict[str, pd.DataFrame],
    rets: pd.DataFrame,
    *,
    n_lags: int = 1,
    newey_west: bool = True,
    nw_lags: int = 6,
    trim_pct: float = 0,
    weights: pd.DataFrame | None = None,
    time_period: tuple[str, str] | None = None,
) -> FMBResult:
    """Run Fama-MacBeth cross-sectional regressions.

    Parameters
    ----------
    chars : dict mapping characteristic names to (nMonths × nStocks) panels
    rets : (nMonths × nStocks) next-month stock returns
    n_lags : months to lag characteristics (default 1)
    newey_west : use Newey-West standard errors
    nw_lags : Newey-West lag length (default 6, matching Petersen 2009 / standard FMB practice)
    trim_pct : trim extreme values before regression (0 = no trimming)
    weights : optional weight panel for WLS; lagged by n_lags like the
        characteristics, so weights are known at formation (e.g. ME at the
        start of the return month), never the return month's own ME
    time_period : (start, end) date strings to restrict sample

    Returns
    -------
    FMBResult with average coefficients, t-stats, and time-series of monthly betas.
    """
    char_names = list(chars.keys())
    k = len(char_names)

    # Align all panels to common dates and stocks
    all_panels = list(chars.values()) + [rets]
    if weights is not None:
        all_panels.append(weights)

    for name, panel in chars.items():
        _validate_panel_labels(panel, name=f"characteristic {name!r}")
    _validate_panel_labels(rets, name="returns")
    if weights is not None:
        _validate_panel_labels(weights, name="weights")

    common_dates = all_panels[0].index
    common_cols = all_panels[0].columns
    for p in all_panels[1:]:
        common_dates = common_dates.intersection(p.index)
        common_cols = common_cols.intersection(p.columns)

    if common_dates.has_duplicates or common_cols.has_duplicates:
        raise ValueError("aligned panel labels must be unique")
    _validate_contiguous_months(common_dates)

    rets_a = rets.loc[common_dates, common_cols]
    chars_a = {name: chars[name].loc[common_dates, common_cols] for name in char_names}
    if weights is not None:
        weights_a = weights.loc[common_dates, common_cols]

    # Apply time period filter
    if time_period is not None:
        start, end = pd.Timestamp(time_period[0]), pd.Timestamp(time_period[1])
        mask = (rets_a.index >= start) & (rets_a.index <= end)
        rets_a = rets_a.loc[mask]
        chars_a = {name: df.loc[mask] for name, df in chars_a.items()}
        if weights is not None:
            weights_a = weights_a.loc[mask]

    _validate_panel_labels(rets_a, name="aligned returns")
    for name, panel in chars_a.items():
        _validate_panel_labels(panel, name=f"aligned characteristic {name!r}")
    if weights is not None:
        _validate_panel_labels(weights_a, name="aligned weights")
    _validate_contiguous_months(rets_a.index)

    T = len(rets_a)
    dates = rets_a.index

    # Pre-allocate
    ts_beta = np.full((T, k + 1), np.nan)  # +1 for intercept
    r2_cs = np.full(T, np.nan)

    # Stage 1: Monthly cross-sectional regressions
    for t in range(n_lags, T):
        # Dependent variable: returns at time t
        y = rets_a.iloc[t].values

        # Independent variables: characteristics lagged by n_lags
        X_parts = []
        for name in char_names:
            x_col = chars_a[name].iloc[t - n_lags].values
            X_parts.append(x_col)

        X_chars = np.column_stack(X_parts)  # (nStocks, k)

        # Trim if requested — trim each characteristic column independently
        if trim_pct > 0:
            for j in range(k):
                col = X_chars[:, j:j+1].T  # (1, nStocks)
                X_chars[:, j] = trim(col, trim_pct).ravel()

        # Add intercept
        n_stocks = len(y)
        X = np.column_stack([np.ones(n_stocks), X_chars])

        # Run regression
        if weights is not None:
            w = weights_a.iloc[t - n_lags].values
            res = nanwls(y, X, w)
        else:
            res = nanols(y, X)

        if res.nobs >= k + 2:
            ts_beta[t] = res.beta
            r2_cs[t] = res.rsq

    # Stage 2: Time-series average with Newey-West SEs
    if newey_west:
        mean_beta, tstat = newey_west_t(ts_beta, nlags=nw_lags)
    else:
        mean_beta = np.nanmean(ts_beta, axis=0)
        se = np.nanstd(ts_beta, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(ts_beta[:, 0])))
        tstat = np.where(se > 0, mean_beta / se, np.nan)

    return FMBResult(
        beta=mean_beta,
        tstat=tstat,
        ts_beta=ts_beta,
        r2_cs=r2_cs,
        char_names=["const"] + char_names,
    )
