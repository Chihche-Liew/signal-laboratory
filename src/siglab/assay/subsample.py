"""Expanded subsample tests and rolling decay analysis.

Part 6 of the rigorous testing framework:

6.1 Subsample Tests — evaluate signals across multiple time periods
6.2 Rolling Decay Analysis — 60-month rolling FMB t-statistics with trend test
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from siglab.factor_model.fama_macbeth import run_fama_macbeth
from siglab.assay.sample import direction_matches


# NBER recession dates (peak to trough, monthly)
NBER_RECESSIONS = [
    ("1960-04-01", "1961-02-01"),
    ("1969-12-01", "1970-11-01"),
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# Standard subsample periods
SUBSAMPLES = {
    "full":            (None, None),
    "pre_anomaly":     ("1963-07-01", "1990-12-31"),
    "post_publication": ("1991-01-01", None),
    "pre_2000":        ("1963-07-01", "2000-06-30"),
    "post_2000":       ("2000-07-01", None),
    "post_2010":       ("2011-01-01", None),
    "ex_recession":    "ex_recession",  # special handling
}


@dataclass
class SubsampleResult:
    """FMB results across all subsamples for one signal."""
    signal_name: str
    results: dict[str, float | None]   # subsample_label -> FMB t-stat
    n_robust: int = 0                  # compatibility alias for n_pass
    n_available: int = 0
    n_pass: int = 0
    pass_ratio: float | None = None
    statuses: dict[str, str] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    # subsample_label -> error message for labels whose evaluation RAISED.
    # results[label] is None for these too, but insufficient-data labels
    # stay OUT of this dict — crash and thin-sample are no longer conflated.
    errors: dict[str, str] = field(default_factory=dict)


@dataclass
class DecayResult:
    """Rolling decay analysis for one signal."""
    signal_name: str
    rolling_dates: list[str]      # center date of each window
    rolling_t: list[float]        # FMB t-stat for each window
    trend_slope: float            # OLS slope of |t| on time
    trend_tstat: float            # t-stat for the trend
    classification: str           # "stable", "strengthening", "decaying"


def _is_recession(date: pd.Timestamp) -> bool:
    """Check if a date falls within an NBER recession (peak..trough inclusive).

    Comparison is by calendar month, so month-END timestamps (e.g. the
    2009-06-30 row of a monthly panel) are correctly inside a recession
    whose trough month is 2009-06.
    """
    month = pd.Timestamp(date).to_period("M")
    for start, end in NBER_RECESSIONS:
        if pd.Timestamp(start).to_period("M") <= month <= pd.Timestamp(end).to_period("M"):
            return True
    return False


def run_subsample_tests(
    signal_name: str,
    signal_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    *,
    subsamples: dict[str, tuple | str] | None = None,
    expected_sign: str = "positive",
) -> SubsampleResult:
    """Evaluate a signal's FMB t-stat across multiple subsamples.

    Parameters
    ----------
    signal_panel : (dates x permnos) signal
    ret_panel : (dates x permnos) returns
    subsamples : dict of label -> (start, end) or "ex_recession"
    """
    if subsamples is None:
        subsamples = SUBSAMPLES

    results: dict[str, float | None] = {}
    errors: dict[str, str] = {}
    statuses: dict[str, str] = {}
    reasons: dict[str, str] = {}

    for label, period in subsamples.items():
        try:
            sig = signal_panel.copy()
            ret = ret_panel.copy()

            # Align
            common_dates = sig.index.intersection(ret.index)
            common_cols = sig.columns.intersection(ret.columns)
            sig = sig.loc[common_dates, common_cols]
            ret = ret.loc[common_dates, common_cols]

            active = np.ones(len(ret), dtype=bool)
            if period == "ex_recession":
                active = ~np.array([_is_recession(d) for d in ret.index])
                # Keep the FULL monthly index and blank only the returns.
                # Dropping rows would make run_fama_macbeth's positional
                # 1-month lag pair the last pre-recession signal with the
                # first post-recession return (e.g. the 2007-11 signal with
                # the 2009-07 return). With returns NaN, recession months
                # simply contribute no observations, the lag stays
                # calendar-true, and stage-2 Newey-West (which pairs
                # autocovariances by calendar position) skips the gap.
                ret.loc[~active] = np.nan
            elif period != (None, None):
                start, end = period
                if start is not None:
                    s = pd.Timestamp(start)
                    active &= ret.index >= s
                if end is not None:
                    e = pd.Timestamp(end)
                    active &= ret.index <= e
                ret.loc[~active] = np.nan

            if int(active.sum()) < 24:
                results[label] = None
                statuses[label] = "not_estimable"
                reasons[label] = "fewer_than_24_calendar_months"
                continue

            fmb = run_fama_macbeth({signal_name: sig}, ret, n_lags=1)
            t = float(fmb.tstat[1]) if fmb is not None and len(fmb.tstat) > 1 else None
            if t is None or not np.isfinite(t):
                results[label] = None
                statuses[label] = "not_estimable"
                reasons[label] = "non_finite_fmb_t"
            else:
                results[label] = t
                statuses[label] = "ok"

        except Exception as exc:
            results[label] = None
            errors[label] = f"{type(exc).__name__}: {exc}"
            statuses[label] = "error"
            reasons[label] = "subsample_exception"

    n_available = sum(value is not None for value in results.values())
    n_pass = sum(
        1 for v in results.values()
        if v is not None
        and abs(v) > 1.96
        and direction_matches(v, expected_sign)
    )
    pass_ratio = n_pass / n_available if n_available else None

    return SubsampleResult(
        signal_name=signal_name,
        results=results,
        n_robust=n_pass,
        n_available=n_available,
        n_pass=n_pass,
        pass_ratio=pass_ratio,
        statuses=statuses,
        reasons=reasons,
        errors=errors,
    )


def run_decay_analysis(
    signal_name: str,
    signal_panel: pd.DataFrame,
    ret_panel: pd.DataFrame,
    *,
    window: int = 60,
    step: int = 12,
) -> DecayResult:
    """Compute rolling FMB t-statistics and test for a time trend.

    Parameters
    ----------
    window : rolling window in months (default 60)
    step : step size in months between windows (default 12)
    """
    common_dates = signal_panel.index.intersection(ret_panel.index)
    common_cols = signal_panel.columns.intersection(ret_panel.columns)
    sig = signal_panel.loc[common_dates, common_cols]
    ret = ret_panel.loc[common_dates, common_cols]

    dates_list: list[str] = []
    t_list: list[float] = []

    n = len(sig)
    for start_idx in range(0, n - window + 1, step):
        end_idx = start_idx + window
        sig_w = sig.iloc[start_idx:end_idx]
        ret_w = ret.iloc[start_idx:end_idx]

        try:
            fmb = run_fama_macbeth({signal_name: sig_w}, ret_w, n_lags=1)
            t = float(fmb.tstat[1]) if fmb is not None and len(fmb.tstat) > 1 else np.nan
        except Exception:
            t = np.nan

        center = sig_w.index[window // 2]
        dates_list.append(str(center.date()))
        t_list.append(t)

    # Trend test: OLS of |t| on time index with HAC (Newey-West) SEs.
    # Adjacent windows share (window - step) months — 80% overlap at the
    # defaults — so the |t| series is strongly serially correlated and an
    # iid OLS SE mislabels ~25% of pure-noise signals as
    # strengthening/decaying at the nominal 5% level (P2-17).
    t_arr = np.array(t_list)
    valid = np.isfinite(t_arr)
    maxlags = max(1, math.ceil(window / step) - 1)
    # HAC needs ~3x(maxlags + 1) observations to be calibrated (Newey-West
    # rule of thumb); with fewer windows the HAC t is severely anti-conservative.
    min_windows = max(5, 3 * (maxlags + 1))
    if valid.sum() >= min_windows:
        import statsmodels.api as sm

        x = np.arange(len(t_arr))[valid].astype(float)
        y = np.abs(t_arr[valid])
        fit = sm.OLS(y, sm.add_constant(x)).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        slope = float(fit.params[1])
        trend_t = float(fit.tvalues[1])
        # Degeneracy guard (successor of the old iid `se > 0` check): on a
        # flat |t| series the residuals are pure float rounding, so the HAC
        # t is meaningless (NaN or an arbitrary finite value). A flat series
        # is by definition trendless -> force t to 0.0 so it reads "stable".
        resid = y - fit.fittedvalues
        degenerate = np.std(resid) < 1e-10 * max(1.0, float(np.max(np.abs(y))))
        if not np.isfinite(trend_t) or degenerate:
            trend_t = 0.0
    else:
        slope = np.nan
        trend_t = np.nan

    if np.isnan(slope):
        classification = "insufficient_data"
    elif trend_t > 1.96:
        classification = "strengthening"
    elif trend_t < -1.96:
        classification = "decaying"
    else:
        classification = "stable"

    return DecayResult(
        signal_name=signal_name,
        rolling_dates=dates_list,
        rolling_t=t_list,
        trend_slope=float(slope),
        trend_tstat=float(trend_t),
        classification=classification,
    )
