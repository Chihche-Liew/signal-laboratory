"""Statistical utility functions.

Mirrors winsor.m, trim.m, invprctile.m from AssayingAnomalies,
plus Newey-West standard error computation.
"""

import numpy as np
from scipy import stats as sp_stats


def winsorize(x: np.ndarray, pcts: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    """Winsorize array by capping extreme values at given percentiles.

    Parameters
    ----------
    x : array_like — input data (any shape)
    pcts : (lower_pct, upper_pct) — percentile thresholds (0-100 scale)

    Returns
    -------
    Winsorized array (same shape as x). NaN values preserved.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if not finite.any():
        return x

    vals = x[finite]
    lo, hi = np.percentile(vals, pcts[0]), np.percentile(vals, pcts[1])
    x[finite & (x < lo)] = lo
    x[finite & (x > hi)] = hi
    return x


def winsorize_cross_section(
    data: np.ndarray, pcts: tuple[float, float] = (1.0, 99.0)
) -> np.ndarray:
    """Winsorize each row (cross-section) independently.

    Parameters
    ----------
    data : (nMonths, nStocks) array
    pcts : percentile thresholds

    Returns
    -------
    Row-wise winsorized array.
    """
    data = np.asarray(data, dtype=np.float64).copy()
    for i in range(data.shape[0]):
        row = data[i]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        vals = row[finite]
        lo, hi = np.percentile(vals, pcts[0]), np.percentile(vals, pcts[1])
        row[finite & (row < lo)] = lo
        row[finite & (row > hi)] = hi
        data[i] = row
    return data


def trim(data: np.ndarray, pct: float = 1.0) -> np.ndarray:
    """Set extreme values to NaN (row-wise trimming).

    Parameters
    ----------
    data : (nMonths, nStocks) array
    pct : trim percentage — removes top and bottom pct% each row

    Returns
    -------
    Array with extremes replaced by NaN.
    """
    data = np.asarray(data, dtype=np.float64).copy()
    for i in range(data.shape[0]):
        row = data[i]
        finite = np.isfinite(row)
        if finite.sum() < 3:
            continue
        vals = row[finite]
        lo = np.percentile(vals, pct)
        hi = np.percentile(vals, 100.0 - pct)
        mask = finite & ((row < lo) | (row > hi))
        row[mask] = np.nan
        data[i] = row
    return data


def invprctile(
    x: np.ndarray, xq: np.ndarray, method: str = "weibull"
) -> np.ndarray:
    """Compute nonexceedance probabilities (inverse percentile rank).

    Parameters
    ----------
    x : (n,) sample data
    xq : query values — points at which to evaluate percentile rank
    method : plotting position method — 'weibull', 'hazen', 'blom'

    Returns
    -------
    Array of percentile ranks (0-100 scale) for each query value.
    """
    x = np.asarray(x, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)

    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return np.full_like(xq, np.nan)

    sorted_x = np.sort(finite)
    n = len(sorted_x)

    # Plotting positions
    i = np.arange(1, n + 1, dtype=np.float64)
    if method == "weibull":
        p = i / (n + 1)
    elif method == "hazen":
        p = (i - 0.5) / n
    elif method == "blom":
        p = (i - 0.375) / (n + 0.25)
    else:
        p = i / (n + 1)  # default to weibull

    # Interpolate
    result = np.interp(xq, sorted_x, p * 100.0)
    result[xq < sorted_x[0]] = 0.0
    result[xq > sorted_x[-1]] = 100.0
    return result


def newey_west_se(
    X: np.ndarray, residuals: np.ndarray, nlags: int = 0,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Newey-West heteroskedasticity and autocorrelation consistent SEs.

    Parameters
    ----------
    X : (n, k) regressor matrix (should include constant if present)
    residuals : (n,) OLS residuals
    nlags : number of lags. 0 = White (heteroskedasticity-only).
    positions : (n,) integer array of original time-series positions for the
                n observations (e.g. ``np.where(valid_mask)[0]``).  When
                provided, the autocovariance at lag l only pairs observations
                whose positions differ by exactly l, preventing non-adjacent
                observations (created by dropping NaN rows) from being paired.
                If None, observations are assumed to be consecutive.

    Returns
    -------
    (k,) array of Newey-West standard errors for each coefficient.
    """
    X = np.asarray(X, dtype=np.float64)
    e = np.asarray(residuals, dtype=np.float64).ravel()

    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)

    Xe = X * e[:, None]  # (n, k)
    S = Xe.T @ Xe  # lag-0 component

    for lag in range(1, nlags + 1):
        w = 1.0 - lag / (nlags + 1)  # Bartlett kernel weight
        if positions is not None:
            # Only pair (j, i) where positions[j] - positions[i] == lag
            pos_to_idx = {int(p): idx for idx, p in enumerate(positions)}
            Gamma = np.zeros((k, k))
            for i in range(n):
                j_pos = int(positions[i]) + lag
                if j_pos in pos_to_idx:
                    j = pos_to_idx[j_pos]
                    Gamma += np.outer(Xe[j], Xe[i])
        else:
            Gamma = Xe[lag:].T @ Xe[:-lag]
        S += w * (Gamma + Gamma.T)

    # Sandwich: V = (X'X)^{-1} S (X'X)^{-1}
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    return se


def newey_west_t(
    ts_coefs: np.ndarray, nlags: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Newey-West t-statistics for a time-series of estimates.

    This is the version used in Fama-MacBeth: given T monthly coefficient
    estimates, compute the mean and its NW standard error.

    Parameters
    ----------
    ts_coefs : (T,) or (T, k) time-series of coefficient estimates
    nlags : number of lags for autocorrelation adjustment

    Returns
    -------
    (mean, tstat) — mean coefficients and their t-statistics.
    """
    ts_coefs = np.asarray(ts_coefs, dtype=np.float64)
    if ts_coefs.ndim == 1:
        ts_coefs = ts_coefs[:, None]

    T, k = ts_coefs.shape
    means = np.nanmean(ts_coefs, axis=0)

    # Demean
    demeaned = ts_coefs - means[None, :]

    # Retain original row indices so autocovariance pairs are calendar-adjacent
    valid = np.all(np.isfinite(demeaned), axis=1)
    valid_positions = np.where(valid)[0]   # original time positions (0-based)
    dm = demeaned[valid]
    T_eff = dm.shape[0]

    if T_eff < 2:
        return means, np.full(k, np.nan)

    var = np.sum(dm**2, axis=0) / T_eff  # gamma_0

    # Add autocovariance terms — only pair observations that are exactly
    # `lag` apart in calendar time (skips gaps caused by missing months).
    if nlags > 0:
        pos_to_local = {int(p): i for i, p in enumerate(valid_positions)}
        for lag in range(1, nlags + 1):
            w = 1.0 - lag / (nlags + 1)
            gamma = np.zeros(k)
            for local_i, orig_i in enumerate(valid_positions):
                orig_j = int(orig_i) + lag
                if orig_j in pos_to_local:
                    local_j = pos_to_local[orig_j]
                    gamma += dm[local_j] * dm[local_i]
            gamma /= T_eff
            var += 2.0 * w * gamma

    se = np.sqrt(np.maximum(var / T_eff, 0.0))
    tstat = np.where(se > 0, means / se, np.nan)
    return means.ravel(), tstat.ravel()
