"""Gibbons-Ross-Shanken (1989) test for joint significance of alphas.

Mirrors GRStest_p.m from AssayingAnomalies.
Tests H0: all portfolio alphas are jointly zero.
"""

from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats


@dataclass
class GRSResult:
    """Result of the GRS test."""
    f_stat: float    # GRS F-statistic
    p_value: float   # p-value (from F-distribution)
    df1: int         # numerator degrees of freedom (= number of portfolios)
    df2: int         # denominator degrees of freedom


def grs_test(
    alphas: np.ndarray,
    residuals: np.ndarray | None,
    factor_rets: np.ndarray,
) -> GRSResult:
    """Gibbons-Ross-Shanken (1989) joint alpha significance test.

    Tests whether all portfolio alphas are jointly zero against a
    set of benchmark factors.

    Parameters
    ----------
    alphas : (N,) portfolio alpha estimates (from time-series regressions)
    residuals : (T, N) regression residuals. If None, returns NaN result.
    factor_rets : (T, K) factor returns used in the regression

    Returns
    -------
    GRSResult with F-statistic, p-value, and degrees of freedom.
    """
    alphas = np.asarray(alphas, dtype=np.float64).ravel()
    N = len(alphas)

    if residuals is None or N < 2:
        return GRSResult(f_stat=np.nan, p_value=np.nan, df1=N, df2=0)

    residuals = np.asarray(residuals, dtype=np.float64)
    factor_rets = np.asarray(factor_rets, dtype=np.float64)

    # Remove rows with any NaN
    mask = (
        np.all(np.isfinite(residuals), axis=1)
        & np.all(np.isfinite(factor_rets), axis=1)
    )
    resid = residuals[mask]
    F = factor_rets[mask]
    T, K = F.shape

    if T <= N + K:
        return GRSResult(f_stat=np.nan, p_value=np.nan, df1=N, df2=0)

    # Residual covariance matrix — OLS estimator divides by T - K - 1
    # (K regressors + 1 intercept per portfolio; GRS 1989 eq. 5)
    Sigma_e = (resid.T @ resid) / (T - K - 1)

    # Factor moments
    mu_f = F.mean(axis=0)
    Sigma_f = np.cov(F, rowvar=False)
    if Sigma_f.ndim == 0:
        Sigma_f = Sigma_f.reshape(1, 1)

    # GRS statistic
    # F = (T - N - K) / N * alpha' Sigma_e^{-1} alpha / (1 + mu_f' Sigma_f^{-1} mu_f)
    try:
        Sigma_e_inv = np.linalg.inv(Sigma_e)
        Sigma_f_inv = np.linalg.inv(Sigma_f)
    except np.linalg.LinAlgError:
        return GRSResult(f_stat=np.nan, p_value=np.nan, df1=N, df2=0)

    numerator = alphas @ Sigma_e_inv @ alphas
    denominator = 1.0 + mu_f @ Sigma_f_inv @ mu_f

    df1 = N
    df2 = T - N - K

    if df2 <= 0:
        return GRSResult(f_stat=np.nan, p_value=np.nan, df1=df1, df2=df2)

    f_stat = (df2 / df1) * numerator / denominator
    p_value = 1.0 - sp_stats.f.cdf(f_stat, df1, df2)

    return GRSResult(
        f_stat=float(f_stat),
        p_value=float(p_value),
        df1=df1,
        df2=df2,
    )
