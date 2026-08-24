"""Spanning tests: does a new signal carry incremental information beyond known anomalies?

Implements cross-sectional spanning regressions following the framework in
Harvey, Liu & Zhu (2016) and Hou, Xue & Zhang (2020). For each LLM-discovered
signal, we ask: does it remain significant after controlling for the most
correlated existing anomalies from Chen & Zimmermann (2022)?

Key functions
-------------
- pairwise_correlations: rank-correlate a new signal against all CZ anomalies
- spanning_fmb: multivariate FMB with new signal + top-K CZ controls
- run_spanning_test: full pipeline for one signal
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from siglab.factor_model.fama_macbeth import run_fama_macbeth
from siglab.assay.sample import direction_matches
from siglab.utils.regression import MAX_REGRESSION_CONDITION


SKIP_ZERO_VARIANCE = "zero_variance"
SKIP_RANK_DEFICIENCY = "rank_deficiency"
SKIP_POOR_CONDITIONING = "poor_conditioning"
SKIP_INSUFFICIENT_COVERAGE = "insufficient_coverage"


@dataclass
class SpanningResult:
    """Result of a spanning test for one signal."""
    signal_name: str

    # Univariate (no controls)
    fmb_t_univariate: float

    # Conditional (after controlling for top-K CZ anomalies)
    fmb_t_conditional: float
    n_controls: int
    control_names: list[str]
    control_corrs: list[float]       # Spearman corr with new signal

    # Full regression coefficients for controls
    control_t_stats: dict[str, float] = field(default_factory=dict)
    n_identified_months: int = 0  # characteristics-only selection design
    n_valid_months: int = 0  # return-complete, full-rank, well-conditioned
    skipped_controls: dict[str, str] = field(default_factory=dict)

    # Whether the signal survives
    survives_196: bool = False       # correct-direction |t| > 1.96
    survives_300: bool = False       # correct-direction |t| > 3.00
    absolute_survives_196: bool = False
    absolute_survives_300: bool = False
    direction_consistent: bool = False

    status: str = "ok"
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class MonthlyDesignDiagnostics:
    """Counts of monthly designs surviving each identification check."""

    coverage_months: int = 0
    nonzero_variance_months: int = 0
    full_rank_months: int = 0
    well_conditioned_months: int = 0


def _rankdata_2d(arr: np.ndarray) -> np.ndarray:
    """Row-wise rank (NaN-aware, scipy-free, vectorised per row)."""
    from scipy.stats import rankdata as _rd
    out = np.full_like(arr, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        mask = np.isfinite(row)
        if mask.sum() > 1:
            out[i, mask] = _rd(row[mask])
    return out


def pairwise_correlations(
    new_signal: pd.DataFrame,
    cz_panels: dict[str, pd.DataFrame],
    *,
    sample_every: int = 12,
) -> pd.Series:
    """Compute average cross-sectional Spearman correlation between a new
    signal and each CZ anomaly.

    Uses vectorised numpy operations: pre-ranks the new signal once,
    then computes Pearson correlation on ranks for each CZ panel in bulk.

    Parameters
    ----------
    new_signal : (dates × permnos) panel for the new signal
    cz_panels : dict mapping CZ anomaly name → (dates × permnos) panel
    sample_every : compute correlation every N months for speed

    Returns
    -------
    Series indexed by CZ anomaly name, values are average Spearman rho.
    Sorted by absolute value descending.
    """
    sampled_dates = new_signal.index[::sample_every]
    if len(sampled_dates) == 0:
        return pd.Series(dtype=float)

    # Pre-rank the new signal at sampled dates (once, reused for all CZ)
    new_sampled = new_signal.loc[sampled_dates]
    new_ranks = _rankdata_2d(new_sampled.values)  # (n_dates, n_cols)
    new_cols = new_sampled.columns

    corrs: dict[str, float] = {}

    for cz_name, cz_panel in cz_panels.items():
        shared_dates = sampled_dates.intersection(cz_panel.index)
        if len(shared_dates) < 6:
            continue

        shared_cols = new_cols.intersection(cz_panel.columns)
        if len(shared_cols) < 50:
            continue

        # Align to shared dates and columns
        date_idx = np.array([sampled_dates.get_loc(d) for d in shared_dates])
        col_idx_new = np.array([new_cols.get_loc(c) for c in shared_cols])

        nr = new_ranks[np.ix_(date_idx, col_idx_new)]  # (n_shared_dates, n_shared_cols)
        cz_vals = cz_panel.loc[shared_dates, shared_cols].values
        cr = _rankdata_2d(cz_vals)

        # Pearson correlation of ranks per row, averaged
        rhos = []
        for i in range(nr.shape[0]):
            a, b = nr[i], cr[i]
            mask = np.isfinite(a) & np.isfinite(b)
            n = mask.sum()
            if n < 30:
                continue
            a_m, b_m = a[mask], b[mask]
            a_dm = a_m - a_m.mean()
            b_dm = b_m - b_m.mean()
            denom = np.sqrt((a_dm ** 2).sum() * (b_dm ** 2).sum())
            if denom > 0:
                rhos.append(float((a_dm * b_dm).sum() / denom))

        if rhos:
            corrs[cz_name] = float(np.mean(rhos))

    result = pd.Series(corrs).sort_values(key=lambda x: x.abs(), ascending=False)
    return result


def spanning_fmb(
    new_signal: pd.DataFrame,
    ret_panel: pd.DataFrame,
    control_panels: dict[str, pd.DataFrame],
    *,
    n_lags: int = 1,
    nw_lags: int = 6,
) -> tuple[float, float, dict[str, float]]:
    """Multivariate Fama-MacBeth: new signal + controls.

    Returns
    -------
    (t_univariate, t_conditional, control_t_stats)
    """
    # Univariate FMB
    fmb_uni = run_fama_macbeth(
        {"new": new_signal}, ret_panel, n_lags=n_lags, nw_lags=nw_lags,
    )
    t_uni = float(fmb_uni.tstat[1]) if fmb_uni is not None else np.nan

    # Multivariate FMB with controls
    chars = {"new": new_signal}
    chars.update(control_panels)
    fmb_mv = run_fama_macbeth(chars, ret_panel, n_lags=n_lags, nw_lags=nw_lags)

    if fmb_mv is None or len(fmb_mv.tstat) < 2:
        return t_uni, np.nan, {}

    t_cond = float(fmb_mv.tstat[1])  # new signal is first char
    control_ts = {}
    for i, name in enumerate(fmb_mv.char_names[2:], start=2):  # skip const and new
        control_ts[name] = float(fmb_mv.tstat[i])

    return t_uni, t_cond, control_ts


def _monthly_design_diagnostics(
    new_signal: pd.DataFrame,
    control_panels: dict[str, pd.DataFrame],
    *,
    n_lags: int = 1,
    return_panel: pd.DataFrame | None = None,
) -> MonthlyDesignDiagnostics:
    """Count monthly designs through nested coverage/identification gates.

    Each count is a nested gate: enough joint characteristic observations,
    non-zero cross-sectional variance, full column rank (including the
    intercept), and finally an acceptable unit-diagonal Gram condition number.
    When returns are supplied, their finite-observation mask is applied before
    testing the design. The last gate matches the scale-invariant guard used by
    ``nanols``.
    """
    if n_lags < 0:
        raise ValueError("n_lags must be non-negative")

    panels = [new_signal, *control_panels.values()]
    alignment_panels = [*panels]
    if return_panel is not None:
        alignment_panels.append(return_panel)
    common_dates = new_signal.index
    common_cols = new_signal.columns
    for panel in alignment_panels[1:]:
        common_dates = common_dates.intersection(panel.index)
        common_cols = common_cols.intersection(panel.columns)

    if len(common_dates) <= n_lags or len(common_cols) == 0:
        return MonthlyDesignDiagnostics()

    aligned = [
        np.asarray(panel.loc[common_dates, common_cols], dtype=float)
        for panel in panels
    ]
    aligned_returns = (
        np.asarray(return_panel.loc[common_dates, common_cols], dtype=float)
        if return_panel is not None
        else None
    )
    n_chars = len(aligned)
    required_observations = n_chars + 2
    coverage_months = 0
    nonzero_variance_months = 0
    full_rank_months = 0
    well_conditioned_months = 0

    # Formation row t can only feed a return row t + n_lags, so the final
    # n_lags characteristic rows are not potential monthly regressions.
    for t in range(len(common_dates) - n_lags):
        chars = np.column_stack([panel[t] for panel in aligned])
        mask = np.all(np.isfinite(chars), axis=1)
        if aligned_returns is not None:
            mask &= np.isfinite(aligned_returns[t + n_lags])
        if int(mask.sum()) < required_observations:
            continue
        coverage_months += 1

        complete_chars = chars[mask]
        if np.any(np.ptp(complete_chars, axis=0) == 0.0):
            continue
        nonzero_variance_months += 1

        design = np.column_stack(
            [np.ones(len(complete_chars), dtype=float), complete_chars]
        )
        gram = design.T @ design
        diagonal = np.diag(gram)
        if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0):
            continue
        scale = np.sqrt(diagonal)
        normalized = gram / scale[:, None] / scale[None, :]
        if np.any(~np.isfinite(normalized)):
            continue

        singular_values = np.linalg.svd(normalized, compute_uv=False)
        tolerance = (
            singular_values[0]
            * max(normalized.shape)
            * np.finfo(singular_values.dtype).eps
        )
        if int(np.sum(singular_values > tolerance)) < normalized.shape[0]:
            continue
        full_rank_months += 1

        condition = singular_values[0] / singular_values[-1]
        if not np.isfinite(condition) or condition > MAX_REGRESSION_CONDITION:
            continue
        well_conditioned_months += 1

    return MonthlyDesignDiagnostics(
        coverage_months=coverage_months,
        nonzero_variance_months=nonzero_variance_months,
        full_rank_months=full_rank_months,
        well_conditioned_months=well_conditioned_months,
    )


def monthly_design_diagnostics(
    new_signal: pd.DataFrame,
    control_panels: dict[str, pd.DataFrame],
    *,
    n_lags: int = 1,
) -> MonthlyDesignDiagnostics:
    """Count identified monthly regressor designs without using returns."""
    return _monthly_design_diagnostics(
        new_signal,
        control_panels,
        n_lags=n_lags,
    )


def fmb_design_diagnostics(
    new_signal: pd.DataFrame,
    ret_panel: pd.DataFrame,
    control_panels: dict[str, pd.DataFrame],
    *,
    n_lags: int = 1,
) -> MonthlyDesignDiagnostics:
    """Count actual return-complete, identified monthly FMB designs."""
    return _monthly_design_diagnostics(
        new_signal,
        control_panels,
        n_lags=n_lags,
        return_panel=ret_panel,
    )


def _design_skip_reason(
    diagnostics: MonthlyDesignDiagnostics,
    *,
    min_valid_months: int,
) -> str | None:
    """Return the first monthly-design gate that misses the requirement."""
    if diagnostics.coverage_months < min_valid_months:
        return SKIP_INSUFFICIENT_COVERAGE
    if diagnostics.nonzero_variance_months < min_valid_months:
        return SKIP_ZERO_VARIANCE
    if diagnostics.full_rank_months < min_valid_months:
        return SKIP_RANK_DEFICIENCY
    if diagnostics.well_conditioned_months < min_valid_months:
        return SKIP_POOR_CONDITIONING
    return None


def select_rank_aware_controls(
    new_signal: pd.DataFrame,
    cz_panels: dict[str, pd.DataFrame],
    correlations: pd.Series,
    *,
    top_k: int,
    n_lags: int = 1,
    min_valid_months: int = 24,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], int]:
    """Greedily select identified CZ controls in correlation-ranked order.

    Selection depends only on the supplied characteristic panels and their
    precomputed correlation order. Returns, regression coefficients, and
    t-statistics are deliberately absent from this interface.
    """
    selected: dict[str, pd.DataFrame] = {}
    skipped: dict[str, str] = {}
    n_design_months = 0

    for control_name in correlations.index:
        if len(selected) >= top_k:
            break

        control = cz_panels[control_name]
        aligned_control = (
            control
            if control.index.equals(new_signal.index)
            else control.reindex(index=new_signal.index)
        )
        candidate_controls = {**selected, control_name: aligned_control}
        diagnostics = monthly_design_diagnostics(
            new_signal,
            candidate_controls,
            n_lags=n_lags,
        )
        reason = _design_skip_reason(
            diagnostics,
            min_valid_months=min_valid_months,
        )
        if reason is not None:
            skipped[control_name] = reason
            continue

        selected[control_name] = aligned_control
        n_design_months = diagnostics.well_conditioned_months

    return selected, skipped, n_design_months


def count_valid_fmb_months(
    new_signal: pd.DataFrame,
    ret_panel: pd.DataFrame,
    control_panels: dict[str, pd.DataFrame],
    *,
    n_lags: int = 1,
) -> int:
    """Count return-complete monthly regressions that are safely identified."""
    return fmb_design_diagnostics(
        new_signal,
        ret_panel,
        control_panels,
        n_lags=n_lags,
    ).well_conditioned_months


def run_spanning_test(
    signal_name: str,
    new_signal: pd.DataFrame,
    ret_panel: pd.DataFrame,
    cz_panels: dict[str, pd.DataFrame],
    *,
    top_k: int = 10,
    n_lags: int = 1,
    nw_lags: int = 6,
    corr_sample_every: int = 12,
    expected_sign: str = "positive",
    min_valid_months: int = 24,
) -> SpanningResult:
    """Full spanning test pipeline for one signal.

    1. Rank-correlate the new signal against ALL CZ anomalies.
    2. Greedily pick the top-K identified, well-conditioned controls while
       preserving the absolute-correlation ranking.
    3. Run multivariate FMB: new signal + top-K controls.
    4. Report whether the new signal survives.

    Parameters
    ----------
    signal_name : identifier for the new signal
    new_signal : (dates × permnos) panel
    ret_panel : (dates × permnos) stock return panel
    cz_panels : dict of ALL CZ anomaly panels
    top_k : number of CZ controls to include
    """
    n_identified_months = 0
    try:
        # Step 1: Correlations
        corrs = pairwise_correlations(
            new_signal,
            cz_panels,
            sample_every=corr_sample_every,
        )

        if len(corrs) == 0:
            return SpanningResult(
                signal_name=signal_name,
                fmb_t_univariate=np.nan,
                fmb_t_conditional=np.nan,
                n_controls=0,
                control_names=[],
                control_corrs=[],
                status="not_estimable",
                reason="no_overlapping_controls",
            )

        # Step 2: Preserve the correlation ranking, but admit a candidate
        # only when at least min_valid_months characteristic-only monthly
        # designs are full-rank and well-conditioned. Outcome data never
        # enter this greedy selection.
        control_panels, skipped_controls, n_identified_months = (
            select_rank_aware_controls(
                new_signal,
                cz_panels,
                corrs,
                top_k=top_k,
                n_lags=n_lags,
                min_valid_months=min_valid_months,
            )
        )

        top_names = list(control_panels)
        top_corrs = [float(corrs[name]) for name in top_names]
        if not top_names:
            return SpanningResult(
                signal_name=signal_name,
                fmb_t_univariate=np.nan,
                fmb_t_conditional=np.nan,
                n_controls=0,
                control_names=[],
                control_corrs=[],
                n_identified_months=n_identified_months,
                skipped_controls=skipped_controls,
                status="not_estimable",
                reason="no_valid_controls",
            )

        # Step 2b: Audit the fixed control set on the actual return-complete
        # regression sample. This gates estimation and reporting, but never
        # feeds back into which controls are selected.
        n_valid_months = count_valid_fmb_months(
            new_signal,
            ret_panel,
            control_panels,
            n_lags=n_lags,
        )
        if n_valid_months < min_valid_months:
            return SpanningResult(
                signal_name=signal_name,
                fmb_t_univariate=np.nan,
                fmb_t_conditional=np.nan,
                n_controls=len(top_names),
                control_names=top_names,
                control_corrs=top_corrs,
                n_identified_months=n_identified_months,
                n_valid_months=n_valid_months,
                skipped_controls=skipped_controls,
                status="not_estimable",
                reason="insufficient_joint_coverage",
            )

        # Step 3: Spanning FMB
        t_uni, t_cond, ctrl_ts = spanning_fmb(
            new_signal, ret_panel, control_panels,
            n_lags=n_lags, nw_lags=nw_lags,
        )

        finite_conditional = np.isfinite(t_cond)
        absolute_196 = bool(finite_conditional and abs(t_cond) > 1.96)
        absolute_300 = bool(finite_conditional and abs(t_cond) > 3.00)
        sign_ok = direction_matches(t_cond, expected_sign)
        return SpanningResult(
            signal_name=signal_name,
            fmb_t_univariate=t_uni,
            fmb_t_conditional=t_cond,
            n_controls=len(top_names),
            control_names=top_names,
            control_corrs=top_corrs,
            control_t_stats=ctrl_ts,
            n_identified_months=n_identified_months,
            n_valid_months=n_valid_months,
            skipped_controls=skipped_controls,
            survives_196=absolute_196 and sign_ok,
            survives_300=absolute_300 and sign_ok,
            absolute_survives_196=absolute_196,
            absolute_survives_300=absolute_300,
            direction_consistent=sign_ok,
            status="ok" if finite_conditional else "not_estimable",
            reason=None if finite_conditional else "non_finite_conditional_t",
        )

    except Exception as e:
        return SpanningResult(
            signal_name=signal_name,
            fmb_t_univariate=np.nan,
            fmb_t_conditional=np.nan,
            n_controls=0,
            control_names=[],
            control_corrs=[],
            n_identified_months=n_identified_months,
            status="error",
            reason="spanning_exception",
            error=str(e),
        )
