"""Portfolio sorting breakpoints and assignment.

Mirrors makeUnivSortInd.m, makeBivSortInd.m, assignToPtf.m from AssayingAnomalies.

Convention: panels are pd.DataFrame with DatetimeIndex rows, permno (int) columns.
Portfolio indices are 1-based (1, 2, ..., n_ptf). 0 or NaN = unassigned.
"""

import numpy as np
import pandas as pd


def assign_to_ptf(var: pd.Series, breakpoints: np.ndarray) -> pd.Series:
    """Assign stocks to portfolios based on breakpoints.

    Parameters
    ----------
    var : Series of sorting variable values for one month
    breakpoints : (m,) array of breakpoint values (sorted ascending)

    Returns
    -------
    Series of portfolio assignments (1 to m+1). NaN where var is NaN.
    """
    result = pd.Series(np.nan, index=var.index)
    valid = var.notna()

    if not valid.any() or len(breakpoints) == 0:
        return result

    # np.searchsorted gives bin index (0-based); add 1 for 1-based portfolios
    vals = var[valid].values
    bins = np.searchsorted(breakpoints, vals, side="right") + 1
    result[valid] = bins.astype(float)

    return result


def make_univ_sort_ind(
    var: pd.DataFrame,
    n_ptf: int | list[float] = 5,
    *,
    nyse_indicator: pd.DataFrame | None = None,
    me: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create univariate portfolio sort assignments.

    Parameters
    ----------
    var : (nMonths × nStocks) sorting variable panel
    n_ptf : number of equal quantile portfolios (e.g., 5 = quintiles),
            OR a list of percentile breakpoints (e.g., [30, 70] for 3 groups)
    nyse_indicator : boolean panel — if provided, compute breakpoints using
                     NYSE stocks only (standard academic approach)
    me : market cap panel — if provided, use cap-weighted breakpoints

    Returns
    -------
    DataFrame of portfolio assignments (1-based). NaN for unassigned.
    """
    ind = pd.DataFrame(np.nan, index=var.index, columns=var.columns)

    for date in var.index:
        row = var.loc[date]
        valid = row.notna()

        if valid.sum() < 5:
            continue

        # Determine which stocks to use for computing breakpoints
        if nyse_indicator is not None and date in nyse_indicator.index:
            bp_mask = valid & nyse_indicator.loc[date].fillna(False)
        else:
            bp_mask = valid

        bp_vals = row[bp_mask].dropna()
        if len(bp_vals) < 3:
            continue

        # Compute breakpoints
        if isinstance(n_ptf, list):
            # Custom percentile breakpoints
            breakpoints = np.percentile(bp_vals.values, n_ptf)
        elif me is not None and date in me.index:
            # Cap-weighted breakpoints
            me_row = me.loc[date]
            mask = bp_mask & me_row.notna() & (me_row > 0)
            if mask.sum() < 3:
                continue
            breakpoints = _cap_weighted_breakpoints(
                row[mask].values, me_row[mask].values, n_ptf
            )
        else:
            # Equal quantile breakpoints
            pcts = np.linspace(0, 100, n_ptf + 1)[1:-1]
            breakpoints = np.percentile(bp_vals.values, pcts)

        ind.loc[date] = assign_to_ptf(row, breakpoints)

    return ind


def _cap_weighted_breakpoints(
    var_vals: np.ndarray, me_vals: np.ndarray, n_ptf: int
) -> np.ndarray:
    """Compute breakpoints such that each portfolio has equal market cap mass.

    Parameters
    ----------
    var_vals : sorting variable values
    me_vals : market cap values (positive)
    n_ptf : number of portfolios

    Returns
    -------
    Array of (n_ptf - 1) breakpoint values.
    """
    order = np.argsort(var_vals)
    sorted_var = var_vals[order]
    sorted_me = me_vals[order]

    cum_me = np.cumsum(sorted_me)
    total_me = cum_me[-1]

    breakpoints = []
    for i in range(1, n_ptf):
        target = total_me * i / n_ptf
        idx = np.searchsorted(cum_me, target)
        if idx >= len(sorted_var):
            idx = len(sorted_var) - 1
        breakpoints.append(sorted_var[idx])

    return np.array(breakpoints)


def make_biv_sort_ind(
    var1: pd.DataFrame,
    var2: pd.DataFrame,
    n_ptf1: int = 5,
    n_ptf2: int = 5,
    *,
    sort_type: str = "unconditional",
    nyse_indicator: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create bivariate (double) sort portfolio assignments.

    Parameters
    ----------
    var1, var2 : (nMonths × nStocks) panels of two sorting variables
    n_ptf1, n_ptf2 : number of portfolios for each variable
    sort_type : 'unconditional' (independent sorts) or 'conditional' (sort var2 within var1)
    nyse_indicator : boolean panel for NYSE breakpoints

    Returns
    -------
    DataFrame with combined portfolio assignments (1 to n_ptf1 * n_ptf2).
    For unconditional: portfolio = (i-1) * n_ptf2 + j
    For conditional: same numbering, but var2 sorted within each var1 bucket
    """
    if sort_type == "unconditional":
        return _biv_sort_unconditional(var1, var2, n_ptf1, n_ptf2, nyse_indicator)
    elif sort_type == "conditional":
        return _biv_sort_conditional(var1, var2, n_ptf1, n_ptf2, nyse_indicator)
    else:
        raise ValueError(f"sort_type must be 'unconditional' or 'conditional', got '{sort_type}'")


def _biv_sort_unconditional(
    var1: pd.DataFrame,
    var2: pd.DataFrame,
    n_ptf1: int,
    n_ptf2: int,
    nyse_indicator: pd.DataFrame | None,
) -> pd.DataFrame:
    """Independent double sort."""
    ind1 = make_univ_sort_ind(var1, n_ptf1, nyse_indicator=nyse_indicator)
    ind2 = make_univ_sort_ind(var2, n_ptf2, nyse_indicator=nyse_indicator)

    # Combined: (i-1) * n_ptf2 + j
    combined = (ind1 - 1) * n_ptf2 + ind2
    # NaN where either is NaN
    combined[ind1.isna() | ind2.isna()] = np.nan

    return combined


def _biv_sort_conditional(
    var1: pd.DataFrame,
    var2: pd.DataFrame,
    n_ptf1: int,
    n_ptf2: int,
    nyse_indicator: pd.DataFrame | None,
) -> pd.DataFrame:
    """Conditional sort: sort var2 within each var1 bucket."""
    ind1 = make_univ_sort_ind(var1, n_ptf1, nyse_indicator=nyse_indicator)
    combined = pd.DataFrame(np.nan, index=var1.index, columns=var1.columns)

    for date in var1.index:
        row1 = ind1.loc[date]
        row_var2 = var2.loc[date]

        for i in range(1, n_ptf1 + 1):
            in_bucket = row1 == i
            if in_bucket.sum() < 3:
                continue

            # Sort var2 within this var1 bucket
            bucket_var2 = row_var2.copy()
            bucket_var2[~in_bucket] = np.nan

            if nyse_indicator is not None and date in nyse_indicator.index:
                bucket_nyse = nyse_indicator.loc[date] & in_bucket
                bp_vals = bucket_var2[bucket_nyse].dropna()
            else:
                bp_vals = bucket_var2.dropna()

            if len(bp_vals) < 3:
                continue

            pcts = np.linspace(0, 100, n_ptf2 + 1)[1:-1]
            breakpoints = np.percentile(bp_vals.values, pcts)

            assigns = assign_to_ptf(bucket_var2, breakpoints)
            valid = assigns.notna()
            combined.loc[date, valid] = (i - 1) * n_ptf2 + assigns[valid]

    return combined
