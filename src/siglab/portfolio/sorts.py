"""High-level portfolio sort analysis.

Mirrors runUnivSort.m and runBivSort.m from AssayingAnomalies.
These are the main entry points that orchestrate breakpoints → returns → factor regressions.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class UnivSortResult:
    """Result of a univariate portfolio sort analysis."""
    # Portfolio excess returns and t-stats
    xret: np.ndarray          # (nPtf,) mean excess returns (annualized)
    txret: np.ndarray         # (nPtf,) t-stats for excess returns
    # Factor model results
    alpha: np.ndarray         # (nPtf,) factor model alphas (annualized)
    talpha: np.ndarray        # (nPtf,) t-stats for alphas
    beta: np.ndarray          # (nPtf, nFactors) factor loadings
    tbeta: np.ndarray         # (nPtf, nFactors) t-stats for betas
    rsq: np.ndarray           # (nPtf,) R-squared values
    sharpe: np.ndarray        # (nPtf,) annualized Sharpe ratios
    ir: np.ndarray            # (nPtf,) information ratios
    # GRS test
    grs_pval: float           # p-value from GRS joint alpha test
    # Portfolio diagnostics
    ptf_count: pd.DataFrame   # (nMonths × nPtf) stock counts
    ptf_mve: pd.DataFrame     # (nMonths × nPtf) market values
    ptf_rets: pd.DataFrame    # (nMonths × nPtf) raw portfolio returns
    # Metadata
    n_ptf: int = 0
    weighting: str = ""
    factor_model: int = 0


@dataclass
class BivSortResult:
    """Result of a bivariate portfolio sort analysis."""
    main: UnivSortResult                  # Full grid results
    cond_var1: UnivSortResult | None = None  # Long-short on var1
    cond_var2: UnivSortResult | None = None  # Long-short on var2


def run_univ_sort(
    ret: pd.DataFrame,
    var: pd.DataFrame,
    me: pd.DataFrame,
    ff_factors: pd.DataFrame,
    *,
    n_ptf: int | list[float] = 5,
    weighting: str = "value",
    holding_period: int = 1,
    factor_model: int = 4,
    nyse_indicator: pd.DataFrame | None = None,
    add_long_short: bool = True,
    time_period: tuple[str, str] | None = None,
    newey_west_lags: int = 6,
    q_factors: pd.DataFrame | None = None,
) -> UnivSortResult:
    """Run a complete univariate portfolio sort analysis.

    Parameters
    ----------
    ret : (nMonths × nStocks) stock return panel
    var : (nMonths × nStocks) sorting variable panel
    me : (nMonths × nStocks) market equity panel
    ff_factors : Fama-French factor DataFrame (from data.factors)
    n_ptf : number of portfolios or list of percentile breakpoints
    weighting : 'value' or 'equal'
    holding_period : rebalancing frequency in months
    factor_model : 1=CAPM, 3=FF3, 4=FF4, 5=FF5, 6=FF6, 7=q-factor
    nyse_indicator : boolean panel for NYSE breakpoints
    add_long_short : if True, add high-minus-low portfolio
    time_period : (start, end) date strings to restrict sample
    q_factors : Hou-Xue-Zhang q-factor DataFrame (required for model 7)

    Returns
    -------
    UnivSortResult with all portfolio statistics.
    """
    from siglab.portfolio.breakpoints import make_univ_sort_ind
    from siglab.portfolio.returns import calc_ptf_rets, add_long_short as _add_ls
    from siglab.factor_model.time_series import est_factor_regs, est_grs_inputs
    from siglab.factor_model.grs import grs_test

    # Restrict time period if specified
    if time_period is not None:
        start, end = pd.Timestamp(time_period[0]), pd.Timestamp(time_period[1])
        mask = (ret.index >= start) & (ret.index <= end)
        ret = ret.loc[mask]
        var = var.reindex(ret.index)
        me = me.reindex(ret.index)
        if nyse_indicator is not None:
            nyse_indicator = nyse_indicator.reindex(ret.index)

    # Step 1: Create portfolio assignments
    ind = make_univ_sort_ind(
        var, n_ptf, nyse_indicator=nyse_indicator
    )

    n_ptf_expected = n_ptf if isinstance(n_ptf, int) else len(n_ptf) + 1

    # Step 2: Compute portfolio returns
    ptf_result = calc_ptf_rets(
        ret, ind, me, weighting=weighting, holding_period=holding_period,
        n_ptf=n_ptf_expected,
    )
    ptf_rets = ptf_result.ptf_rets

    if ptf_rets.empty or ptf_rets.shape[1] == 0:
        return _empty_result(n_ptf_expected)

    # Step 3: Add long-short spread
    if add_long_short:
        ptf_rets = _add_ls(ptf_rets)

    # Step 4: Run factor regressions
    reg_result = est_factor_regs(ptf_rets, ff_factors, model=factor_model,
                                  newey_west_lags=newey_west_lags,
                                  q_factors=q_factors)

    # Step 5: GRS test (exclude long-short portfolio)
    is_q = isinstance(factor_model, int) and factor_model in (7, 8)
    factor_src = q_factors if is_q else ff_factors
    n_actual = ptf_rets.shape[1] - (1 if add_long_short else 0)
    if n_actual >= 2 and factor_src is not None:
        from siglab.data.factors import get_factor_columns
        factor_cols = get_factor_columns(factor_model)

        # Align dates
        common = ptf_rets.index.intersection(factor_src.index)
        ptf_aligned = ptf_rets.loc[common].iloc[:, :n_actual]
        factors_aligned = factor_src.loc[common, factor_cols]

        rf_col = "R_F" if is_q else "RF"
        if rf_col in factor_src.columns:
            rf = factor_src.loc[common, rf_col]
            excess_rets = ptf_aligned.subtract(rf, axis=0)
        else:
            excess_rets = ptf_aligned

        # GRS inputs re-estimated on the joint complete sample: grs_test
        # computes Sigma_e on rows where every portfolio is observed, so
        # the alphas fed to it must come from those same rows (P1-4). The
        # reported per-portfolio alpha/talpha keep their own samples.
        grs_alphas, grs_resid = est_grs_inputs(
            excess_rets.values, factors_aligned.values
        )
        grs_result = grs_test(
            grs_alphas,
            grs_resid,
            factors_aligned.values,
        )
        grs_pval = grs_result.p_value
    else:
        grs_pval = np.nan

    n_ptf_actual = n_ptf_expected

    return UnivSortResult(
        xret=reg_result.mean_ret,
        txret=reg_result.t_mean_ret,
        alpha=reg_result.alpha,
        talpha=reg_result.talpha,
        beta=reg_result.beta,
        tbeta=reg_result.tbeta,
        rsq=reg_result.rsq,
        sharpe=reg_result.sharpe,
        ir=reg_result.ir,
        grs_pval=grs_pval,
        ptf_count=ptf_result.ptf_count,
        ptf_mve=ptf_result.ptf_mve,
        ptf_rets=ptf_rets,
        n_ptf=n_ptf_actual,
        weighting=weighting,
        factor_model=factor_model,
    )


def run_biv_sort(
    ret: pd.DataFrame,
    var1: pd.DataFrame,
    var2: pd.DataFrame,
    me: pd.DataFrame,
    ff_factors: pd.DataFrame,
    *,
    n_ptf1: int = 5,
    n_ptf2: int = 5,
    sort_type: str = "unconditional",
    weighting: str = "value",
    factor_model: int = 4,
    nyse_indicator: pd.DataFrame | None = None,
    time_period: tuple[str, str] | None = None,
    q_factors: pd.DataFrame | None = None,
) -> BivSortResult:
    """Run a complete bivariate portfolio sort analysis.

    Parameters
    ----------
    ret, var1, var2, me, ff_factors : data panels
    n_ptf1, n_ptf2 : portfolios per dimension
    sort_type : 'unconditional' or 'conditional'
    weighting, factor_model, nyse_indicator, time_period : same as run_univ_sort
    q_factors : Hou-Xue-Zhang q-factor DataFrame (required for model 7)

    Returns
    -------
    BivSortResult with main grid and conditional long-short results.
    """
    from siglab.portfolio.breakpoints import make_biv_sort_ind
    from siglab.portfolio.returns import calc_ptf_rets
    from siglab.factor_model.time_series import est_factor_regs

    if time_period is not None:
        start, end = pd.Timestamp(time_period[0]), pd.Timestamp(time_period[1])
        mask = (ret.index >= start) & (ret.index <= end)
        ret = ret.loc[mask]
        var1 = var1.reindex(ret.index)
        var2 = var2.reindex(ret.index)
        me = me.reindex(ret.index)
        if nyse_indicator is not None:
            nyse_indicator = nyse_indicator.reindex(ret.index)

    # Create combined portfolio assignments
    ind = make_biv_sort_ind(
        var1, var2, n_ptf1, n_ptf2,
        sort_type=sort_type, nyse_indicator=nyse_indicator,
    )

    # Compute returns for all grid portfolios
    ptf_result = calc_ptf_rets(ret, ind, me, weighting=weighting)

    # Factor regressions on the full grid
    reg_result = est_factor_regs(ptf_result.ptf_rets, ff_factors, model=factor_model,
                                  q_factors=q_factors)

    main_result = UnivSortResult(
        xret=reg_result.mean_ret,
        txret=reg_result.t_mean_ret,
        alpha=reg_result.alpha,
        talpha=reg_result.talpha,
        beta=reg_result.beta,
        tbeta=reg_result.tbeta,
        rsq=reg_result.rsq,
        sharpe=reg_result.sharpe,
        ir=reg_result.ir,
        grs_pval=np.nan,
        ptf_count=ptf_result.ptf_count,
        ptf_mve=ptf_result.ptf_mve,
        ptf_rets=ptf_result.ptf_rets,
        n_ptf=n_ptf1 * n_ptf2,
        weighting=weighting,
        factor_model=factor_model,
    )

    return BivSortResult(main=main_result)


def _empty_result(n_ptf: int) -> UnivSortResult:
    """Return an empty result when no valid portfolios are formed."""
    return UnivSortResult(
        xret=np.full(n_ptf, np.nan),
        txret=np.full(n_ptf, np.nan),
        alpha=np.full(n_ptf, np.nan),
        talpha=np.full(n_ptf, np.nan),
        beta=np.full((n_ptf, 1), np.nan),
        tbeta=np.full((n_ptf, 1), np.nan),
        rsq=np.full(n_ptf, np.nan),
        sharpe=np.full(n_ptf, np.nan),
        ir=np.full(n_ptf, np.nan),
        grs_pval=np.nan,
        ptf_count=pd.DataFrame(),
        ptf_mve=pd.DataFrame(),
        ptf_rets=pd.DataFrame(),
        n_ptf=n_ptf,
    )
