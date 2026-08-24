"""Time-series factor model regressions.

Mirrors estFactorRegs.m from AssayingAnomalies.
Regresses portfolio excess returns on factor models (CAPM, FF3, FF4, FF5, FF6, q-factor).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from siglab.utils.regression import nanols
from siglab.utils.stats import newey_west_se


@dataclass
class FactorRegResult:
    """Result of factor model regressions across portfolios."""
    alpha: np.ndarray       # (nPtf,) intercepts (monthly)
    talpha: np.ndarray      # (nPtf,) t-stats for alphas
    beta: np.ndarray        # (nPtf, nFactors) factor loadings
    tbeta: np.ndarray       # (nPtf, nFactors) t-stats for betas
    rsq: np.ndarray         # (nPtf,) R-squared values
    sharpe: np.ndarray      # (nPtf,) annualized Sharpe ratios
    ir: np.ndarray          # (nPtf,) information ratios (annualized)
    mean_ret: np.ndarray    # (nPtf,) mean excess returns (annualized)
    t_mean_ret: np.ndarray  # (nPtf,) t-stats for mean excess returns
    resid: np.ndarray | None = None  # (T, nPtf) residuals (for GRS test)


def est_factor_regs(
    ptf_rets: pd.DataFrame,
    ff_factors: pd.DataFrame,
    *,
    model: int | pd.DataFrame = 4,
    newey_west_lags: int = 0,
    q_factors: pd.DataFrame | None = None,
) -> FactorRegResult:
    """Estimate factor model regressions for portfolio returns.

    Parameters
    ----------
    ptf_rets : (nMonths × nPtf) portfolio return DataFrame
    ff_factors : Fama-French factor DataFrame with columns Mkt-RF, SMB, HML, etc.
    model : factor model number (1=CAPM, 3=FF3, 4=FF4, 5=FF5, 6=FF6,
            7=q-factor) OR a (dates x K) DataFrame of custom factor returns
            with a DatetimeIndex (aligned by label on the common dates, like
            the int-model path). Bare ndarrays are rejected: positional
            pairing against internally computed dates silently misaligns.
    newey_west_lags : lags for Newey-West SEs (0 = OLS standard errors)
    q_factors : Hou-Xue-Zhang q-factor DataFrame (required for model 7).
                Columns: R_MKT, R_ME, R_IA, R_ROE, R_F

    Returns
    -------
    FactorRegResult with alphas, betas, t-stats, R², Sharpe, IR for each portfolio.
    """
    from siglab.data.factors import get_factor_columns

    # Choose the right factor source for q-models
    is_q_model = isinstance(model, (int, float)) and int(model) == 7
    if is_q_model:
        if q_factors is None:
            raise ValueError("q_factors DataFrame required for model 7")
        factor_src = q_factors
    else:
        factor_src = ff_factors

    # Align dates
    common_dates = ptf_rets.index.intersection(factor_src.index)
    if isinstance(model, pd.DataFrame):
        if not isinstance(model.index, pd.DatetimeIndex):
            raise TypeError(
                "custom factor DataFrame must have a DatetimeIndex to align "
                "with portfolio return dates"
            )
        common_dates = common_dates.intersection(model.index)
    ptf = ptf_rets.loc[common_dates].values  # (T, nPtf)
    T, nPtf = ptf.shape

    # Track which columns are zero-investment (already excess returns)
    is_zero_inv = np.zeros(nPtf, dtype=bool)
    if hasattr(ptf_rets, "columns"):
        for j, col in enumerate(ptf_rets.columns):
            if str(col) == "LS":
                is_zero_inv[j] = True

    # Get factor matrix
    if isinstance(model, (int, float)):
        factor_cols = get_factor_columns(int(model))
        F = factor_src.loc[common_dates, factor_cols].values  # (T, K)
    elif isinstance(model, pd.DataFrame):
        F = model.loc[common_dates].to_numpy(dtype=float)     # label-aligned
    else:
        raise TypeError(
            "custom factor model must be an int model id or a pandas "
            "DataFrame with a DatetimeIndex; got "
            f"{type(model).__name__} — bare arrays pair positionally and "
            "silently misalign (P2-10)"
        )

    K = F.shape[1]

    # Get risk-free rate for excess returns
    if is_q_model:
        rf = factor_src.loc[common_dates, "R_F"].values if "R_F" in factor_src.columns else np.zeros(T)
    elif "RF" in ff_factors.columns:
        rf = ff_factors.loc[common_dates, "RF"].values
    else:
        rf = np.zeros(T)

    # Convert portfolio returns to excess returns
    # Zero-investment (L-S) portfolios are already excess returns; don't subtract RF
    ptf_excess = ptf.copy()
    for j in range(nPtf):
        if not is_zero_inv[j]:
            ptf_excess[:, j] = ptf[:, j] - rf

    # Pre-allocate results
    alpha = np.full(nPtf, np.nan)
    talpha = np.full(nPtf, np.nan)
    beta = np.full((nPtf, K), np.nan)
    tbeta = np.full((nPtf, K), np.nan)
    rsq = np.full(nPtf, np.nan)
    resid_full = np.full((T, nPtf), np.nan)

    # Regressor matrix: [constant, factors]
    X = np.column_stack([np.ones(T), F])

    for j in range(nPtf):
        y = ptf_excess[:, j]
        res = nanols(y, X)

        if res.nobs < K + 2:
            continue

        alpha[j] = res.beta[0]
        beta[j] = res.beta[1:]
        rsq[j] = res.rsq

        if newey_west_lags > 0:
            # Newey-West SEs — pass original positions so non-adjacent
            # observations (after NaN removal) are not incorrectly paired.
            mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            if mask.sum() > K + 1:
                positions = np.where(mask)[0]
                se_nw = newey_west_se(X[mask], res.resid[mask], newey_west_lags,
                                      positions=positions)
                talpha[j] = res.beta[0] / se_nw[0] if se_nw[0] > 0 else np.nan
                tbeta[j] = np.where(se_nw[1:] > 0, res.beta[1:] / se_nw[1:], np.nan)
        else:
            talpha[j] = res.tstat[0]
            tbeta[j] = res.tstat[1:]

        resid_full[:, j] = res.resid

    # Compute Sharpe ratios and information ratios
    mean_ret = np.nanmean(ptf_excess, axis=0) * 12  # annualized
    std_ret = np.nanstd(ptf_excess, axis=0, ddof=1) * np.sqrt(12)
    sharpe = np.where(std_ret > 0, mean_ret / std_ret, np.nan)

    # t-stat for mean excess returns
    nobs_per_ptf = np.sum(np.isfinite(ptf_excess), axis=0)
    se_mean = np.nanstd(ptf_excess, axis=0, ddof=1) / np.sqrt(np.maximum(nobs_per_ptf, 1))
    t_mean_ret = np.where(se_mean > 0, np.nanmean(ptf_excess, axis=0) / se_mean, np.nan)

    # Information ratio: alpha / std(residuals), annualized
    resid_std = np.nanstd(resid_full, axis=0, ddof=1) * np.sqrt(12)
    ir = np.where(resid_std > 0, alpha * 12 / resid_std, np.nan)

    return FactorRegResult(
        alpha=alpha,
        talpha=talpha,
        beta=beta,
        tbeta=tbeta,
        rsq=rsq,
        sharpe=sharpe,
        ir=ir,
        mean_ret=mean_ret,
        t_mean_ret=t_mean_ret,
        resid=resid_full,
    )


def est_grs_inputs(
    excess_rets: np.ndarray,
    factor_rets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Re-estimate portfolio regressions on the joint complete sample for GRS.

    GRS (1989) requires the alphas and the residual covariance to come
    from the SAME sample. ``est_factor_regs`` estimates each portfolio on
    its own valid months (correct for the reported per-portfolio
    alphas/t-stats), but ``grs_test`` builds Sigma_e only from rows where
    EVERY portfolio's residual and every factor are finite. Feeding
    per-portfolio-sample alphas into that joint-sample covariance mixes
    samples and mis-sizes the test whenever any portfolio has missing
    months (P1-4).

    Parameters
    ----------
    excess_rets : (T, N) portfolio EXCESS returns (may contain NaN)
    factor_rets : (T, K) factor returns

    Returns
    -------
    (alphas, resid) : (N,) intercepts and (T, N) residuals, both estimated
    on rows where all N portfolios and all K factors are finite. Residuals
    are NaN outside those rows, so grs_test's own finite mask selects
    exactly the estimation sample. Both are all-NaN when the joint sample
    has fewer than K + 2 rows (grs_test then reports NaN, matching its
    existing degenerate behavior).
    """
    R = np.asarray(excess_rets, dtype=np.float64)
    F = np.asarray(factor_rets, dtype=np.float64)
    T, N = R.shape
    K = F.shape[1]

    joint = np.all(np.isfinite(R), axis=1) & np.all(np.isfinite(F), axis=1)
    alphas = np.full(N, np.nan)
    resid = np.full((T, N), np.nan)

    if int(joint.sum()) < K + 2:
        return alphas, resid

    X = np.column_stack([np.ones(int(joint.sum())), F[joint]])
    # One shared design matrix: lstsq solves every portfolio's OLS at once
    # (identical to per-portfolio OLS on the joint sample, which has no
    # NaNs by construction).
    # Unguarded (no rank-deficiency check) is fine here: lstsq's SVD gives
    # the min-norm solution even if X were collinear, and the rank policy
    # in regression.py (P1-5) guards the normal-equations path used
    # elsewhere; X's factor columns are fixed FF/q-factor sets that are
    # never collinear with the intercept.
    coef, _, _, _ = np.linalg.lstsq(X, R[joint], rcond=None)
    alphas = coef[0].copy()
    resid[joint] = R[joint] - X @ coef
    return alphas, resid
