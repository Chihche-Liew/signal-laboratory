"""Evaluation-only signal agent.

This module evaluates already-proposed signal expressions. Candidate generation
and theme search live in the lab proposer components, keeping the public agent
surface focused on execution and statistical evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from siglab.agent.executor import SignalEngine
from siglab.factor_model.fama_macbeth import run_fama_macbeth
from siglab.portfolio.sorts import run_univ_sort
from siglab.assay.sample import (
    build_monthly_sample,
    normalize_factor_panel,
    prepare_signal_panel,
)


@dataclass
class SignalResult:
    """Result of evaluating a candidate signal."""

    name: str
    expression: str
    hypothesis: str
    theme: str
    # Portfolio sort results
    ls_alpha: float = np.nan     # L-S FF5 alpha (annualized)
    ls_talpha: float = np.nan    # t-stat for L-S alpha
    ls_sharpe: float = np.nan    # L-S Sharpe ratio
    # FMB results
    fmb_coef: float = np.nan     # FMB regression coefficient
    fmb_tstat: float = np.nan    # FMB t-stat
    # Metadata
    coverage: float = np.nan     # Signal coverage (fraction non-NaN)
    elapsed_sec: float = np.nan  # Time to compute
    error: str | None = None     # Error message if failed
    quintile_rets: list[float] = field(default_factory=list)  # Annualized quintile returns


class SignalEvaluator:
    """Fundamental signal evaluator.

    Parameters
    ----------
    engine : SignalEngine with loaded Compustat-CCM and CRSP data
    ret_panel : (dates x permnos) return panel
    me_panel : (dates x permnos) market equity panel
    ff_factors : Fama-French factor DataFrame
    nyse_panel : (dates x permnos) NYSE indicator panel
    fin_panel : Optional (dates x permnos) financial firm indicator panel
    start_date : Sample start date (default: 1963-07-01)
    """

    def __init__(
        self,
        engine: SignalEngine,
        ret_panel: pd.DataFrame,
        me_panel: pd.DataFrame,
        ff_factors: pd.DataFrame,
        nyse_panel: pd.DataFrame,
        fin_panel: pd.DataFrame | None = None,
        start_date: str = "1963-07-01",
    ):
        self.engine = engine
        self.ret_panel = ret_panel
        self.me_panel = me_panel
        self.ff_factors = ff_factors
        self.nyse_panel = nyse_panel
        self.fin_panel = fin_panel
        self.start_date = start_date

    def evaluate_signal(
        self,
        expression: str,
        name: str = "signal",
        hypothesis: str = "",
        theme: str = "",
        exclude_financials: bool = True,
        exclude_microcap: bool = True,
        alpha_factor_model: int = 5,
    ) -> SignalResult:
        """Execute and evaluate a single signal expression.

        Parameters
        ----------
        exclude_microcap : if True, drop stocks below the 10th NYSE ME
            percentile (following Hou, Xue & Zhang 2020). Default True
            per the paper's main specification.
        alpha_factor_model : factor model used to estimate long-short alpha
            and its t-statistic. Default 5 (Fama-French five-factor model).

        Returns a SignalResult with portfolio sort and FMB statistics.
        """
        t0 = time.time()
        result = SignalResult(
            name=name, expression=expression,
            hypothesis=hypothesis, theme=theme,
        )

        try:
            # Execute expression to get monthly panel
            signal = self.engine.execute(expression)

            sample = build_monthly_sample(
                self.ret_panel,
                self.me_panel,
                self.nyse_panel,
                start_date=self.start_date,
            )
            prepared = prepare_signal_panel(
                signal,
                sample,
                financials=self.fin_panel,
                exclude_financials=exclude_financials,
                exclude_microcap=exclude_microcap,
            )
            sig = prepared.signal
            ret = prepared.returns
            me = prepared.market_equity
            nyse = prepared.nyse
            ff_sub = normalize_factor_panel(
                self.ff_factors,
                name="ff_factors",
            ).reindex(sample.returns.index)

            # Coverage check (on raw signal, before fill/exclusion)
            result.coverage = prepared.coverage
            if result.coverage < 0.01:
                result.error = f"Insufficient coverage: {result.coverage:.1%}"
                result.elapsed_sec = time.time() - t0
                return result

            # Portfolio sort
            sort_result = run_univ_sort(
                ret, sig, me, ff_sub,
                n_ptf=5, weighting="value", factor_model=alpha_factor_model,
                nyse_indicator=nyse, add_long_short=True,
            )

            result.ls_alpha = sort_result.alpha[-1] * 12 if sort_result.alpha is not None else np.nan
            result.ls_talpha = sort_result.talpha[-1] if sort_result.talpha is not None else np.nan
            result.ls_sharpe = sort_result.sharpe[-1] if sort_result.sharpe is not None else np.nan

            if sort_result.ptf_rets is not None:
                result.quintile_rets = (sort_result.ptf_rets.mean() * 12 * 100).tolist()[:5]

            # FMB regression
            fmb_result = run_fama_macbeth({name: sig}, ret, n_lags=1)
            if fmb_result is not None and len(fmb_result.tstat) > 1:
                result.fmb_coef = fmb_result.beta[1]
                result.fmb_tstat = fmb_result.tstat[1]

        except Exception as e:
            result.error = str(e)

        result.elapsed_sec = time.time() - t0
        return result


__all__ = ["SignalEvaluator", "SignalResult"]
