"""OLS and WLS regression with NaN handling.

Mirrors the functionality of nanols.m and nanwls.m from AssayingAnomalies.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class OLSResult:
    """Result of an OLS or WLS regression."""

    beta: np.ndarray       # (k,) regression coefficients
    se: np.ndarray         # (k,) standard errors
    tstat: np.ndarray      # (k,) t-statistics
    rsq: float             # R-squared
    resid: np.ndarray      # (n,) residuals (full length, NaN where input was NaN)
    nobs: int              # number of non-NaN observations used
    yhat: np.ndarray       # (n,) fitted values (full length, NaN where input was NaN)


# Unit-diagonal Gram condition numbers above this indicate
# machine-precision collinearity (exact duplicates fail matrix_rank
# outright or land near 1e16+); honest near-duplicates (e.g.
# |corr| = 0.999 -> cond ~ 2e3) stay far below.
MAX_REGRESSION_CONDITION = 1e12


def _nan_result(k: int, n_full: int, nobs: int) -> OLSResult:
    """All-NaN result (rank-deficient design or too few observations)."""
    return OLSResult(
        beta=np.full(k, np.nan),
        se=np.full(k, np.nan),
        tstat=np.full(k, np.nan),
        rsq=np.nan,
        resid=np.full(n_full, np.nan),
        nobs=nobs,
        yhat=np.full(n_full, np.nan),
    )


def _rank_deficient(XtX: np.ndarray, k: int) -> bool:
    """True when X'X cannot support a unique OLS solution.

    The old pinv fallback silently split the coefficient across collinear
    columns and emitted absurd finite t-stats (duplicated regressors both
    read |t| ~ 77 in the verified repro; exact-fit cases reach ~1e16).
    All-NaN is the honest answer — FMB-style callers already skip NaN
    months (newey_west_t drops non-finite rows). Check the unit-diagonal
    Gram matrix so the result is invariant to regressor units.
    """
    diagonal = np.diag(XtX)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
        return True

    scale = np.sqrt(diagonal)
    normalized = XtX / scale[:, None] / scale[None, :]
    if np.any(~np.isfinite(normalized)):
        return True
    if np.linalg.matrix_rank(normalized) < k:
        return True
    return bool(np.linalg.cond(normalized) > MAX_REGRESSION_CONDITION)


def _nan_mask(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return boolean mask of rows where all values are finite."""
    combined = np.column_stack([y, X])
    return np.all(np.isfinite(combined), axis=1)


def nanols(y: np.ndarray, X: np.ndarray) -> OLSResult:
    """Ordinary least squares with automatic NaN removal.

    Parameters
    ----------
    y : (n,) array — dependent variable
    X : (n, k) array — independent variables (should include constant if desired)

    Returns
    -------
    OLSResult with coefficients, standard errors, t-stats, R², residuals.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 1 and X.shape[1] == len(y):
        X = X.T  # handle row vector input

    n_full = len(y)
    mask = _nan_mask(y, X)
    n = mask.sum()
    k = X.shape[1]

    # Degenerate cases
    if n < k + 1:
        return _nan_result(k, n_full, n)

    yc = y[mask]
    Xc = X[mask]

    # OLS: beta = (X'X)^{-1} X'y — refuse rank-deficient designs outright
    XtX = Xc.T @ Xc
    if _rank_deficient(XtX, k):
        return _nan_result(k, n_full, n)
    XtX_inv = np.linalg.inv(XtX)

    beta = XtX_inv @ (Xc.T @ yc)
    yhat_c = Xc @ beta
    resid_c = yc - yhat_c

    # Residual variance (unbiased)
    dof = n - k
    s2 = (resid_c @ resid_c) / max(dof, 1)

    # Standard errors
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    tstat = np.where(se > 0, beta / se, np.nan)

    # R-squared
    ss_res = resid_c @ resid_c
    ss_tot = np.sum((yc - yc.mean()) ** 2)
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Map back to full arrays
    resid_full = np.full(n_full, np.nan)
    yhat_full = np.full(n_full, np.nan)
    resid_full[mask] = resid_c
    yhat_full[mask] = yhat_c

    return OLSResult(
        beta=beta, se=se, tstat=tstat, rsq=rsq,
        resid=resid_full, nobs=n, yhat=yhat_full,
    )


def nanwls(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> OLSResult:
    """Weighted least squares with automatic NaN removal.

    Parameters
    ----------
    y : (n,) array — dependent variable
    X : (n, k) array — independent variables
    w : (n,) array — observation weights (positive)

    Returns
    -------
    OLSResult with coefficients, standard errors, t-stats, R².
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    w = np.asarray(w, dtype=np.float64).ravel()
    if X.shape[0] == 1 and X.shape[1] == len(y):
        X = X.T

    n_full = len(y)
    mask = _nan_mask(y, X) & np.isfinite(w) & (w > 0)
    n = mask.sum()
    k = X.shape[1]

    if n < k + 1:
        return _nan_result(k, n_full, n)

    yc = y[mask]
    Xc = X[mask]
    wc = w[mask]

    # WLS: transform by sqrt(w)
    sqw = np.sqrt(wc)
    Xw = Xc * sqw[:, None]
    yw = yc * sqw

    XtX = Xw.T @ Xw
    if _rank_deficient(XtX, k):
        return _nan_result(k, n_full, n)
    XtX_inv = np.linalg.inv(XtX)

    beta = XtX_inv @ (Xw.T @ yw)
    yhat_c = Xc @ beta
    resid_c = yc - yhat_c

    dof = n - k
    s2 = np.sum(wc * resid_c**2) / max(dof, 1)
    var_beta = s2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(se > 0, beta / se, np.nan)

    ss_res = np.sum(wc * resid_c**2)
    ymean = np.average(yc, weights=wc)
    ss_tot = np.sum(wc * (yc - ymean) ** 2)
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    resid_full = np.full(n_full, np.nan)
    yhat_full = np.full(n_full, np.nan)
    resid_full[mask] = resid_c
    yhat_full[mask] = yhat_c

    return OLSResult(
        beta=beta, se=se, tstat=tstat, rsq=rsq,
        resid=resid_full, nobs=n, yhat=yhat_full,
    )
