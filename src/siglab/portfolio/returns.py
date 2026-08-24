"""Portfolio return computation.

Mirrors calcPtfRets.m from AssayingAnomalies.
Computes value-weighted or equal-weighted portfolio returns from
stock-level returns and portfolio assignments.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioResult:
    """Result of portfolio return computation."""
    ptf_rets: pd.DataFrame   # (nMonths × nPtf) portfolio returns
    ptf_count: pd.DataFrame  # (nMonths × nPtf) stock counts per portfolio
    ptf_mve: pd.DataFrame    # (nMonths × nPtf) total market value


def calc_ptf_rets(
    ret: pd.DataFrame,
    ind: pd.DataFrame,
    me: pd.DataFrame,
    *,
    weighting: str = "value",
    holding_period: int = 1,
    min_stocks: int = 1,
    n_ptf: int | None = None,
) -> PortfolioResult:
    """Compute portfolio returns from stock returns and portfolio assignments.

    Uses the standard convention: portfolio assignments and market cap are
    LAGGED by 1 period. Stocks are assigned to portfolios at month t-1 and
    returns are computed at month t.

    Parameters
    ----------
    ret : (nMonths × nStocks) stock return panel
    ind : (nMonths × nStocks) portfolio assignments (1-based, NaN = unassigned)
    me : (nMonths × nStocks) market equity panel
    weighting : 'value' for value-weighted, 'equal' for equal-weighted
    holding_period : months to hold before rebalancing (1 = monthly)
    min_stocks : minimum stocks for a valid portfolio-month
    n_ptf : expected number of portfolio labels. When given, result frames
        always carry the FULL label set 1..n_ptf even if some label never
        appears in the sample, so a downstream long-short spread is by LABEL
        (n_ptf minus 1) — a sample-wide-missing extreme bucket yields an
        all-NaN column instead of a silently relabeled spread.

    Returns
    -------
    PortfolioResult with portfolio returns, counts, and market values.
    """
    # Align all panels
    common_dates = ret.index.intersection(ind.index).intersection(me.index)
    common_cols = ret.columns.intersection(ind.columns).intersection(me.columns)

    if not common_dates.is_monotonic_increasing:
        raise ValueError(
            "panel dates must be sorted ascending — positional lags would "
            "otherwise look ahead"
        )

    ret_a = ret.loc[common_dates, common_cols]
    ind_a = ind.loc[common_dates, common_cols]
    me_a = me.loc[common_dates, common_cols]

    # Lag ind and me by 1 period (formation-month convention)
    ind_lag = ind_a.shift(1)
    me_lag = me_a.shift(1)

    # Handle holding periods > 1 (carry forward assignments)
    if holding_period > 1:
        ind_lag = _apply_holding_period(ind_lag, holding_period)

    # Determine portfolio labels
    all_ptfs = sorted(ind_lag.stack().dropna().unique())
    if len(all_ptfs) == 0:
        empty = pd.DataFrame(index=common_dates)
        return PortfolioResult(ptf_rets=empty, ptf_count=empty, ptf_mve=empty)

    observed_labels = [int(p) for p in all_ptfs]
    if n_ptf is None:
        ptf_labels = observed_labels
    else:
        ptf_labels = list(range(1, int(n_ptf) + 1))
        unexpected = sorted(set(observed_labels) - set(ptf_labels))
        if unexpected:
            raise ValueError(
                f"portfolio assignments contain labels {unexpected} outside "
                f"the expected 1..{int(n_ptf)} range"
            )

    # Pre-allocate result arrays
    dates = common_dates
    ptf_rets = pd.DataFrame(np.nan, index=dates, columns=ptf_labels)
    ptf_count = pd.DataFrame(0, index=dates, columns=ptf_labels)
    ptf_mve = pd.DataFrame(0.0, index=dates, columns=ptf_labels)

    # Compute returns for each date
    for t, date in enumerate(dates):
        r_t = ret_a.iloc[t].values       # stock returns this month
        ind_t = ind_lag.iloc[t].values    # portfolio assignments (lagged)
        me_t = me_lag.iloc[t].values      # market cap (lagged)

        for ptf in ptf_labels:
            in_ptf = (ind_t == ptf) & np.isfinite(r_t)
            if weighting == "value":
                # VW needs a finite formation-month ME to weight by.
                mask = in_ptf & np.isfinite(me_t)
            else:
                # EW contract: bucket membership + a finite return suffice; a
                # stock with unknown formation ME still earns its equal weight.
                # (No EW caller exists today — documented contract, P2-8b.)
                mask = in_ptf
            n = mask.sum()

            if n < min_stocks:
                continue

            ptf_count.loc[dates[t], ptf] = n

            if weighting == "value":
                weights = me_t[mask]
                w_sum = weights.sum()
                if w_sum > 0:
                    ptf_rets.loc[dates[t], ptf] = np.sum(weights * r_t[mask]) / w_sum
                    ptf_mve.loc[dates[t], ptf] = w_sum
            else:  # equal-weighted
                ptf_rets.loc[dates[t], ptf] = np.mean(r_t[mask])
                ptf_mve.loc[dates[t], ptf] = np.nansum(me_t[mask])

    return PortfolioResult(ptf_rets=ptf_rets, ptf_count=ptf_count, ptf_mve=ptf_mve)


def _apply_holding_period(
    ind: pd.DataFrame, holding_period: int
) -> pd.DataFrame:
    """Carry forward portfolio assignments for multi-month holding periods.

    The rebalance grid is anchored at the FIRST row with any finite
    assignment — after calc_ptf_rets's 1-month lag, row 0 is always all-NaN,
    and anchoring there would silently drop the first holding_period-1
    months and shift every rebalance date (P2-8c). Rebalances happen every
    `holding_period` rows from the anchor; other rows carry the last
    assignment forward.
    """
    result = ind.copy()
    has_assignment = ind.notna().any(axis=1).to_numpy()
    if not has_assignment.any():
        return result
    first = int(np.argmax(has_assignment))

    last_valid = ind.iloc[first].copy()
    for i in range(first + 1, len(ind)):
        if (i - first) % holding_period == 0:
            # Rebalancing month: use new assignments
            row = ind.iloc[i]
            # Fill NaN with previous (for stocks that dropped from sort)
            last_valid = row.where(row.notna(), last_valid)
            result.iloc[i] = last_valid
        else:
            # Holding month: carry forward
            result.iloc[i] = last_valid

    return result


def add_long_short(ptf_rets: pd.DataFrame) -> pd.DataFrame:
    """Add a long-short portfolio (last minus first quintile).

    Parameters
    ----------
    ptf_rets : (nMonths × nPtf) portfolio returns

    Returns
    -------
    Same DataFrame with an additional 'LS' column.
    """
    cols = ptf_rets.columns
    ptf_rets = ptf_rets.copy()
    ptf_rets["LS"] = ptf_rets[cols[-1]] - ptf_rets[cols[0]]
    return ptf_rets
