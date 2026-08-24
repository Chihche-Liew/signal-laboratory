"""Horse race and convergence checks for signal discovery.

General-purpose tools for comparing competing signals and deciding when
to stop an evolutionary search. Works with any set of signal panels —
single-theme, cross-theme, or mixed.

Key functions:
- horse_race(): Multivariate spanning test — which signals are independent?
- check_convergence(): Should the evolutionary loop stop?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from siglab.factor_model.fama_macbeth import run_fama_macbeth
from siglab.utils.panel import cross_sectional_rank, fill_var


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class HorseRaceResult:
    """Result of a multivariate signal horse race."""
    # FMB horse race (all signals simultaneous)
    fmb_conditional_t: dict[str, float]   # signal -> conditional t-stat
    fmb_survivors: list[str]              # expected-sign t > threshold

    # Pairwise signal correlations
    corr_matrix: pd.DataFrame             # signal x signal avg cross-sectional Spearman

    # Stepwise elimination survivors (robust to multicollinearity)
    stepwise_survivors: list[str]
    stepwise_t: dict[str, float]          # final conditional t-stats for survivors

    # Summary
    n_input: int
    n_survivors: int                      # stepwise count (primary recommendation)
    absolute_fmb_survivors: list[str] = field(default_factory=list)
    dropped_reasons: dict[str, str] = field(default_factory=dict)
    status: str = "ok"
    reason: str | None = None

    def summary(self) -> str:
        """Human-readable summary of the horse race."""
        lines = [
            f"Horse Race: {self.n_input} signals -> {self.n_survivors} independent survivors",
            "",
            "Simultaneous FMB (all signals as controls):",
        ]
        survivor_set = set(self.fmb_survivors)
        # Sort by |t| for display; the marker still requires expected sign.
        sorted_t = sorted(
            self.fmb_conditional_t.items(),
            key=lambda kv: abs(kv[1]), reverse=True,
        )
        for name, t in sorted_t:
            marker = " ***" if name in survivor_set else ""
            lines.append(f"  {name:<30s} t={t:>7.2f}{marker}")

        lines.append("")
        lines.append(f"Stepwise survivors ({len(self.stepwise_survivors)}):")
        for name in self.stepwise_survivors:
            t = self.stepwise_t[name]
            lines.append(f"  {name:<30s} t={t:>7.2f} ***")

        # High correlations
        lines.append("")
        lines.append("High correlations (|r| > 0.50):")
        names = list(self.corr_matrix.columns)
        found = False
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if j > i:
                    r = self.corr_matrix.iloc[i, j]
                    if abs(r) > 0.50:
                        lines.append(f"  {a} x {b}: r={r:.2f}")
                        found = True
        if not found:
            lines.append("  (none)")

        return "\n".join(lines)


@dataclass
class ConvergenceResult:
    """Result of a convergence check for evolutionary signal discovery."""
    should_stop: bool
    reason: str                    # "continue" or explanation of why to stop
    gen_hit_rates: list[float]     # per-generation hit rates
    gen_best_fmb: list[float]      # best |FMB t| per generation


# ── Horse race ───────────────────────────────────────────────────────────

def horse_race(
    signals: dict[str, pd.DataFrame],
    ret_panel: pd.DataFrame,
    me_panel: pd.DataFrame | None = None,
    *,
    t_threshold: float = 1.96,
    expected_signs: dict[str, str] | None = None,
) -> HorseRaceResult:
    """Run a multivariate horse race to identify independent signals.

    Three stages:
    1. Simultaneous FMB regression with all signals as controls
    2. Pairwise cross-sectional Spearman correlations
    3. Direction-aware backward elimination until every survivor has an
       expected-sign conditional t-stat above the threshold

    Parameters
    ----------
    signals : dict mapping signal names to (dates x permnos) panels
    ret_panel : (dates x permnos) stock returns
    me_panel : optional market equity panel for fill_var()
    t_threshold : significance threshold (default 1.96)
    expected_signs : optional mapping from signal name to positive/negative.
        Missing entries default to positive.

    Returns
    -------
    HorseRaceResult with survivors, conditional t-stats, and correlations.
    """
    if len(signals) == 0:
        raise ValueError("signals dict must not be empty")

    names = list(signals.keys())
    expected_signs = {
        name: (expected_signs or {}).get(name, "positive")
        for name in names
    }
    invalid_signs = {
        name: sign
        for name, sign in expected_signs.items()
        if sign not in {"positive", "negative"}
    }
    if invalid_signs:
        raise ValueError(f"invalid expected signs: {invalid_signs}")

    # ── Prepare signal panels ────────────────────────────────────────
    ranked = {}
    for name in names:
        sig = signals[name]
        if me_panel is not None:
            sig = fill_var(sig, me_panel)
        ranked[name] = cross_sectional_rank(sig)

    # ── Stage 1: Pairwise correlations (compute first for dedup) ────
    corr_matrix = _compute_corr_matrix(ranked)

    # ── Stage 2: Simultaneous FMB ────────────────────────────────────
    fmb = run_fama_macbeth(ranked, ret_panel, n_lags=1)
    fmb_conditional_t = {
        name: float(fmb.tstat[i + 1])
        for i, name in enumerate(names)
    }
    absolute_fmb_survivors = [
        name
        for name, t_value in fmb_conditional_t.items()
        if np.isfinite(t_value) and abs(t_value) > t_threshold
    ]
    fmb_survivors = [
        name
        for name, t_value in fmb_conditional_t.items()
        if _oriented_t(t_value, expected_signs[name]) > t_threshold
    ]

    # ── Stage 3: Direction-aware deduplication and stepwise selection ─
    deduped, dedup_drops = _deduplicate(
        ranked,
        ret_panel,
        corr_matrix,
        expected_signs=expected_signs,
        threshold=0.95,
    )
    (
        stepwise_survivors,
        stepwise_t,
        stepwise_drops,
        stepwise_had_finite,
    ) = _stepwise_elimination(
        deduped,
        ret_panel,
        t_threshold,
        expected_signs=expected_signs,
    )
    dropped_reasons = {**dedup_drops, **stepwise_drops}
    any_finite = stepwise_had_finite or any(
        np.isfinite(value) for value in fmb_conditional_t.values()
    )

    return HorseRaceResult(
        fmb_conditional_t=fmb_conditional_t,
        fmb_survivors=fmb_survivors,
        corr_matrix=corr_matrix,
        stepwise_survivors=stepwise_survivors,
        stepwise_t=stepwise_t,
        n_input=len(names),
        n_survivors=len(stepwise_survivors),
        absolute_fmb_survivors=absolute_fmb_survivors,
        dropped_reasons=dropped_reasons,
        status="ok" if any_finite else "not_estimable",
        reason=None if any_finite else "non_finite_conditional_t",
    )


def _deduplicate(
    ranked_signals: dict[str, pd.DataFrame],
    ret_panel: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    expected_signs: dict[str, str],
    threshold: float = 0.95,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Remove near-duplicate signals (|corr| > threshold).

    When two signals have correlation > threshold, keep the one with
    higher expected-sign-oriented univariate FMB t-stat. This prevents a
    strong coefficient in the wrong economic direction from displacing a
    correct-direction near-duplicate.
    """
    names = list(ranked_signals.keys())
    if len(names) <= 1:
        return dict(ranked_signals), {}

    # Compute univariate t-stats for tie-breaking
    univ_score = {}
    for name in names:
        fmb = run_fama_macbeth({name: ranked_signals[name]}, ret_panel, n_lags=1)
        univ_score[name] = _oriented_t(
            float(fmb.tstat[1]),
            expected_signs[name],
        )

    # Find pairs to deduplicate
    to_drop = set()
    for i, a in enumerate(names):
        if a in to_drop:
            continue
        for j, b in enumerate(names):
            if j <= i or b in to_drop:
                continue
            r = corr_matrix.loc[a, b]
            if abs(r) > threshold:
                # Drop the one with weaker expected-sign evidence.
                if univ_score[a] >= univ_score[b]:
                    to_drop.add(b)
                else:
                    to_drop.add(a)
                    break

    return (
        {n: ranked_signals[n] for n in names if n not in to_drop},
        {n: "near_duplicate" for n in to_drop},
    )


def _stepwise_elimination(
    ranked_signals: dict[str, pd.DataFrame],
    ret_panel: pd.DataFrame,
    t_threshold: float,
    *,
    expected_signs: dict[str, str],
) -> tuple[list[str], dict[str, float], dict[str, str], bool]:
    """Direction-aware backward elimination with re-estimation.

    Non-finite and wrong-sign coefficients are removed before significance
    comparisons. The model is re-estimated after every removal round, so a
    discarded wrong-sign variable cannot determine which correct-sign signal
    is subsequently judged weakest.
    """
    remaining = dict(ranked_signals)
    dropped: dict[str, str] = {}
    had_finite = False

    while remaining:
        fmb = run_fama_macbeth(remaining, ret_panel, n_lags=1)
        names = list(remaining.keys())
        t_vals = {
            name: float(fmb.tstat[i + 1])
            for i, name in enumerate(names)
        }
        finite_names = [name for name, value in t_vals.items() if np.isfinite(value)]
        had_finite = had_finite or bool(finite_names)

        nonfinite = [name for name in names if name not in finite_names]
        if nonfinite:
            # Drop one at a time so a singular joint model gets a chance to
            # recover without discarding every temporarily unidentified term.
            name = nonfinite[-1]
            dropped[name] = "conditional_t_not_estimable"
            del remaining[name]
            continue

        wrong_sign = [
            name
            for name in names
            if _oriented_t(t_vals[name], expected_signs[name]) <= 0
        ]
        if wrong_sign:
            # Remove the strongest contradiction first, then re-estimate;
            # other coefficients may recover once that control is gone.
            name = min(
                wrong_sign,
                key=lambda candidate: _oriented_t(
                    t_vals[candidate],
                    expected_signs[candidate],
                ),
            )
            dropped[name] = "conditional_sign_mismatch"
            del remaining[name]
            continue

        weakest = min(
            names,
            key=lambda name: _oriented_t(t_vals[name], expected_signs[name]),
        )
        if _oriented_t(t_vals[weakest], expected_signs[weakest]) > t_threshold:
            return names, t_vals, dropped, had_finite

        dropped[weakest] = "conditional_t_below_threshold"
        del remaining[weakest]

    return [], {}, dropped, had_finite


def _oriented_t(t_value: float, expected_sign: str) -> float:
    """Orient a t-statistic so positive values support the stated hypothesis."""
    if not np.isfinite(t_value):
        return float("-inf")
    return -t_value if expected_sign == "negative" else t_value


def _compute_corr_matrix(
    ranked_signals: dict[str, pd.DataFrame],
    *,
    min_stocks: int = 31,
    min_dates: int = 3,
) -> pd.DataFrame:
    """Average per-month pairwise-complete correlations of rank signals.

    For every sampled month and every signal pair, means and variances are
    computed on the pair's jointly-valid stocks only (pairwise-complete
    Pearson on the rank panels — equivalent to Spearman up to within-pair
    re-ranking). Pair-months with fewer than ``min_stocks`` joint stocks
    are skipped; pairs with fewer than ``min_dates`` contributing months
    are NaN. NaN means "overlap too thin to establish duplication" — it
    must never be treated as zero correlation.
    """
    names = list(ranked_signals.keys())
    n = len(names)

    all_dates = ranked_signals[names[0]].index
    all_cols = ranked_signals[names[0]].columns
    for name in names[1:]:
        all_dates = all_dates.union(ranked_signals[name].index)
        all_cols = all_cols.union(ranked_signals[name].columns)
    all_dates = all_dates.sort_values()

    # Sample every 12th month for speed
    sample_dates = all_dates[::12] if len(all_dates) > 60 else all_dates

    corr_sum = np.zeros((n, n))
    corr_cnt = np.zeros((n, n), dtype=int)

    for date in sample_dates:
        rows = []
        for name in names:
            panel = ranked_signals[name]
            if date in panel.index:
                rows.append(panel.loc[date].reindex(all_cols).to_numpy(dtype=float))
            else:
                rows.append(np.full(len(all_cols), np.nan))
        X = np.vstack(rows)                      # (n_signals, N_stocks)
        F = np.isfinite(X)
        Z = np.where(F, X, 0.0)
        Ff = F.astype(float)

        n_ij = Ff @ Ff.T                         # jointly-valid counts per pair
        S = Z @ Z.T                              # sum x_i * x_j over joint sample
        A = Z @ Ff.T                             # A[i, j] = sum of x_i over joint(i, j)
        B = (Z * Z) @ Ff.T                       # B[i, j] = sum of x_i^2 over joint(i, j)

        with np.errstate(invalid="ignore", divide="ignore"):
            mean_ij = A / n_ij                   # mean of x_i on joint(i, j)
            cov = S / n_ij - mean_ij * mean_ij.T
            var_ij = B / n_ij - mean_ij ** 2     # var of x_i on joint(i, j)
            corr = cov / np.sqrt(var_ij * var_ij.T)

        ok = (n_ij >= min_stocks) & np.isfinite(corr)
        corr_sum[ok] += corr[ok]
        corr_cnt += ok

    with np.errstate(invalid="ignore"):
        corr_avg = np.where(
            corr_cnt >= min_dates,
            corr_sum / np.maximum(corr_cnt, 1),
            np.nan,
        )
    np.fill_diagonal(corr_avg, 1.0)
    return pd.DataFrame(corr_avg, index=names, columns=names)


# ── Convergence check ────────────────────────────────────────────────────

def check_convergence(
    gen_results: list[list],
) -> ConvergenceResult:
    """Check if the evolutionary signal discovery loop should stop.

    Parameters
    ----------
    gen_results : list of lists, one per generation. Each inner list contains
        objects or dicts with a `fmb_tstat` field and optionally `error`.
        Works with EvaluatedSignal, SignalResult, or plain dicts.

    Returns
    -------
    ConvergenceResult with should_stop flag and reason.
    """
    if len(gen_results) == 0:
        return ConvergenceResult(
            should_stop=False,
            reason="No results yet",
            gen_hit_rates=[],
            gen_best_fmb=[],
        )

    gen_hit_rates = []
    gen_best_fmb = []

    for gen_sigs in gen_results:
        if len(gen_sigs) == 0:
            gen_hit_rates.append(0.0)
            gen_best_fmb.append(0.0)
            continue

        n_valid = 0
        n_sig = 0
        best_t = 0.0

        for sig in gen_sigs:
            fmb_t = _get_fmb_tstat(sig)
            has_error = _get_error(sig)

            if has_error:
                continue

            if fmb_t is None:
                continue

            fmb_t = float(fmb_t)
            if not np.isfinite(fmb_t):
                # NaN/inf t-stats are failed estimations, not weak signals —
                # they must not inflate the hit-rate denominator (P2-18a).
                continue

            n_valid += 1
            abs_t = abs(fmb_t)
            if abs_t > 1.96:
                n_sig += 1
            if abs_t > best_t:
                best_t = abs_t

        hit_rate = n_sig / n_valid if n_valid > 0 else 0.0
        gen_hit_rates.append(hit_rate)
        gen_best_fmb.append(best_t)

    # Need at least 1 generation to check
    if len(gen_results) < 2:
        return ConvergenceResult(
            should_stop=False,
            reason="continue — need at least 2 generations to assess convergence",
            gen_hit_rates=gen_hit_rates,
            gen_best_fmb=gen_best_fmb,
        )

    latest_hit = gen_hit_rates[-1]
    latest_best = gen_best_fmb[-1]
    gen0_best = gen_best_fmb[0]
    prior_hit = gen_hit_rates[-2]
    prior_best = gen_best_fmb[-2]

    # Criterion 1: Hit rate collapse
    if latest_hit < 0.20:
        return ConvergenceResult(
            should_stop=True,
            reason=f"Hit rate collapse: gen {len(gen_results)-1} hit rate = "
                   f"{latest_hit:.0%} (< 20% threshold)",
            gen_hit_rates=gen_hit_rates,
            gen_best_fmb=gen_best_fmb,
        )

    # Criterion 2: Diminishing returns vs. gen 0
    if gen0_best > 0 and latest_best < 0.5 * gen0_best:
        return ConvergenceResult(
            should_stop=True,
            reason=f"Diminishing returns: gen {len(gen_results)-1} best |FMB t| = "
                   f"{latest_best:.2f} < 50% of gen 0 best ({gen0_best:.2f})",
            gen_hit_rates=gen_hit_rates,
            gen_best_fmb=gen_best_fmb,
        )

    # Criterion 3: No improvement + declining hit rate
    if latest_best < prior_best and latest_hit < prior_hit:
        return ConvergenceResult(
            should_stop=True,
            reason=f"No improvement: gen {len(gen_results)-1} best |FMB t| = "
                   f"{latest_best:.2f} < prior {prior_best:.2f} AND hit rate "
                   f"declining ({latest_hit:.0%} < {prior_hit:.0%})",
            gen_hit_rates=gen_hit_rates,
            gen_best_fmb=gen_best_fmb,
        )

    return ConvergenceResult(
        should_stop=False,
        reason="continue — latest generation still productive",
        gen_hit_rates=gen_hit_rates,
        gen_best_fmb=gen_best_fmb,
    )


def _get_fmb_tstat(sig) -> float | None:
    """Extract fmb_tstat from an object or dict."""
    if isinstance(sig, dict):
        return sig.get("fmb_tstat")
    return getattr(sig, "fmb_tstat", None)


def _get_error(sig) -> bool:
    """Check if a signal result has an error."""
    if isinstance(sig, dict):
        return sig.get("error") is not None
    err = getattr(sig, "error", None)
    return err is not None
