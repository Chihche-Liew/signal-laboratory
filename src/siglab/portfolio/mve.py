"""Mean-variance efficient (tangency) portfolio computation.

Mirrors calcMve.m from AssayingAnomalies. (calcNetMve's port was deleted
2026-07-10: zero callers repo-wide, decorative rets_factors argument, and a
market-only combination it could never evaluate — C-3.)
"""

import numpy as np


def calc_mve(rets: np.ndarray) -> tuple[np.ndarray, float]:
    """Compute the maximum Sharpe ratio (tangency) portfolio.

    Parameters
    ----------
    rets : (T, N) excess return matrix for N assets over T periods

    Returns
    -------
    (weights, sharpe) — max-Sharpe portfolio weights and its (annualized,
    non-negative) Sharpe ratio. Weights may be negative (short positions)
    and are normalized so |sum(weights)| = 1, with the overall sign chosen
    so the portfolio mean — hence the Sharpe — is non-negative.
    """
    rets = np.asarray(rets, dtype=np.float64)
    finite = np.all(np.isfinite(rets), axis=1)
    rets_clean = rets[finite]

    T, N = rets_clean.shape
    if T < N + 1:
        return np.full(N, np.nan), np.nan

    if N == 1:
        mu = rets_clean.mean()
        sigma = rets_clean.std()
        if sigma == 0:
            return np.array([np.sign(mu)]), np.nan
        # Holding sign(mu) units of the asset earns |mu|: the returned
        # (weights, sharpe) pair must describe the SAME portfolio.
        w = np.array([np.sign(mu)])
        sharpe = abs(mu) / sigma * np.sqrt(12)  # annualized
        return w, float(sharpe)

    mu = rets_clean.mean(axis=0)
    Sigma = np.cov(rets_clean, rowvar=False)

    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)

    w_raw = Sigma_inv @ mu
    w_sum = np.sum(w_raw)
    if abs(w_sum) < 1e-12:
        return np.full(N, np.nan), np.nan

    w = w_raw / w_sum

    ptf_mu = w @ mu
    if ptf_mu < 0:
        # Sum-normalization by a negative w_sum flips the tangency direction;
        # flip back so the returned weights ATTAIN the reported (maximum)
        # Sharpe instead of its negative. Weights then sum to -1.
        w = -w
        ptf_mu = -ptf_mu
    ptf_var = w @ Sigma @ w
    sharpe = ptf_mu / np.sqrt(max(ptf_var, 1e-16)) * np.sqrt(12)

    return w, float(sharpe)
