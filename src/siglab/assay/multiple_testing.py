"""Multiple testing adjustments and Bayesian inference for signal evaluation.

Implements three families of corrections from the asset pricing literature:

1. **Family-Wise Error Rate (FWER)**
   - Bonferroni (single-step)
   - Holm (sequential step-down)
   Both from Harvey, Liu & Zhu (2016, RFS).

2. **False Discovery Rate (FDR)**
   - Benjamini-Hochberg-Yekutieli (BHY, sequential step-up)
   From Harvey, Liu & Zhu (2016) and Hou, Xue & Zhang (2020, RFS).

3. **Bayesian Inference**
   - Minimum Bayes Factor (MBF)
   - Symmetric-Descending MBF (SD-MBF)
   - Bayesianized p-values under multiple priors
   From Harvey (2017, JF Presidential Address).

Usage
-----
    from siglab.assay.multiple_testing import (
        bonferroni, holm, bhy,
        multiple_testing_summary,
        minimum_bayes_factor, bayesianized_pvalue,
        bayesian_report,
    )

    # Given a list of (name, t_stat) tuples:
    results = multiple_testing_summary(t_stats, alpha=0.05)

    # Bayesian inference for one signal:
    mbf = minimum_bayes_factor(t_stat=3.5)
    bp  = bayesianized_pvalue(t_stat=3.5, prior_prob=0.20)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import e, exp, log

import numpy as np
from scipy import stats as sp_stats


# ═══════════════════════════════════════════════════════════════════════════
# Part 1: Multiple Testing Adjustments (HLZ 2016)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdjustedResult:
    """Result for one hypothesis after multiple testing adjustment."""
    name: str
    t_stat: float
    p_value: float
    rank: int                     # rank by p-value (1 = smallest)
    # Bonferroni
    bonf_adj_p: float             # min(M * p, 1)
    bonf_reject: bool
    # Holm
    holm_adj_p: float
    holm_reject: bool
    # BHY
    bhy_critical: float           # (rank / (M * c_M)) * alpha
    bhy_reject: bool


def _harmonic_number(m: int) -> float:
    """c(M) = sum_{k=1}^{M} 1/k."""
    return sum(1.0 / k for k in range(1, m + 1))


def _sanitize_t_stats(
    t_stats: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Remove entries with NaN/inf/None t-statistics.

    NaN values corrupt Python's sort (NaN comparisons violate total
    order), which breaks the step-wise Holm and BHY procedures.
    """
    clean = []
    for name, t in t_stats:
        if t is None:
            continue
        if not np.isfinite(t):
            continue
        clean.append((name, t))
    return clean


def bonferroni(
    t_stats: list[tuple[str, float]],
    alpha: float = 0.05,
) -> list[AdjustedResult]:
    """Bonferroni correction (FWER control, single-step).

    p_adj(i) = min(M * p_i, 1). Reject if p_adj <= alpha.
    """
    t_stats = _sanitize_t_stats(t_stats)
    M = len(t_stats)
    if M == 0:
        return []

    items = []
    for name, t in t_stats:
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t)))
        items.append({"name": name, "t_stat": t, "p_value": p})

    items.sort(key=lambda x: x["p_value"])

    results = []
    for rank, item in enumerate(items, start=1):
        adj_p = min(M * item["p_value"], 1.0)
        results.append(AdjustedResult(
            name=item["name"],
            t_stat=item["t_stat"],
            p_value=item["p_value"],
            rank=rank,
            bonf_adj_p=adj_p,
            bonf_reject=adj_p <= alpha,
            holm_adj_p=np.nan,
            holm_reject=False,
            bhy_critical=np.nan,
            bhy_reject=False,
        ))
    return results


def holm(
    t_stats: list[tuple[str, float]],
    alpha: float = 0.05,
) -> list[AdjustedResult]:
    """Holm correction (FWER control, sequential step-down).

    Order p-values p_(1) <= ... <= p_(M).
    Find k = min{b : p_(b) > alpha / (M+1-b)}.
    Reject H_(1) ... H_(k-1).
    """
    t_stats = _sanitize_t_stats(t_stats)
    M = len(t_stats)
    if M == 0:
        return []

    items = []
    for name, t in t_stats:
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t)))
        items.append({"name": name, "t_stat": t, "p_value": p})

    items.sort(key=lambda x: x["p_value"])

    # Compute adjusted p-values (step-down)
    adj_p = [0.0] * M
    running_max = 0.0
    for i in range(M):
        raw = (M - i) * items[i]["p_value"]
        running_max = max(running_max, raw)
        adj_p[i] = min(running_max, 1.0)

    results = []
    for rank, (item, ap) in enumerate(zip(items, adj_p), start=1):
        results.append(AdjustedResult(
            name=item["name"],
            t_stat=item["t_stat"],
            p_value=item["p_value"],
            rank=rank,
            bonf_adj_p=np.nan,
            bonf_reject=False,
            holm_adj_p=ap,
            holm_reject=ap <= alpha,
            bhy_critical=np.nan,
            bhy_reject=False,
        ))
    return results


def bhy(
    t_stats: list[tuple[str, float]],
    alpha: float = 0.05,
) -> list[AdjustedResult]:
    """Benjamini-Hochberg-Yekutieli correction (FDR control, step-up).

    c(M) = sum_{k=1}^{M} 1/k.
    Order p-values p_(1) <= ... <= p_(M).
    Find k = max{b : p_(b) <= (b / (M * c(M))) * alpha}.
    Reject H_(1) ... H_(k).
    """
    t_stats = _sanitize_t_stats(t_stats)
    M = len(t_stats)
    if M == 0:
        return []

    c_M = _harmonic_number(M)

    items = []
    for name, t in t_stats:
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t)))
        items.append({"name": name, "t_stat": t, "p_value": p})

    items.sort(key=lambda x: x["p_value"])

    # Compute BHY critical values and adjusted p-values (step-up)
    crits = [(i / (M * c_M)) * alpha for i in range(1, M + 1)]

    # Find the largest k such that p_(k) <= crit_(k)
    k_max = 0
    for i in range(M):
        if items[i]["p_value"] <= crits[i]:
            k_max = i + 1  # 1-indexed

    # Adjusted p-values (step-up): work backwards
    adj_p = [0.0] * M
    adj_p[-1] = min((M * c_M / M) * items[-1]["p_value"], 1.0)
    for i in range(M - 2, -1, -1):
        raw = (M * c_M / (i + 1)) * items[i]["p_value"]
        adj_p[i] = min(raw, adj_p[i + 1], 1.0)

    results = []
    for rank, (item, crit, ap) in enumerate(zip(items, crits, adj_p), start=1):
        results.append(AdjustedResult(
            name=item["name"],
            t_stat=item["t_stat"],
            p_value=item["p_value"],
            rank=rank,
            bonf_adj_p=np.nan,
            bonf_reject=False,
            holm_adj_p=np.nan,
            holm_reject=False,
            bhy_critical=crit,
            bhy_reject=rank <= k_max,
        ))
    return results


@dataclass
class MultipleTestingSummary:
    """Combined results from all three adjustment methods."""
    M: int                        # total number of tests
    alpha: float
    bonf_threshold_t: float       # |t| hurdle for Bonferroni
    holm_threshold_t: float       # approximate (first rejection)
    bhy_threshold_t: float        # |t| at the LAST actual BHY rejection; NaN if none
    n_reject_bonf: int
    n_reject_holm: int
    n_reject_bhy: int
    results: list[dict]           # per-signal combined results


def multiple_testing_summary(
    t_stats: list[tuple[str, float]],
    alpha: float = 0.05,
) -> MultipleTestingSummary:
    """Run all three methods and produce a unified summary.

    Parameters
    ----------
    t_stats : list of (name, t_statistic) tuples (NaN/None filtered automatically)
    alpha : significance level (default 0.05)
    """
    t_stats = _sanitize_t_stats(t_stats)
    M = len(t_stats)
    if M == 0:
        return MultipleTestingSummary(
            M=0, alpha=alpha,
            bonf_threshold_t=np.nan, holm_threshold_t=np.nan,
            bhy_threshold_t=np.nan,
            n_reject_bonf=0, n_reject_holm=0, n_reject_bhy=0,
            results=[],
        )

    indexed_t_stats = [
        (f"__row_{i}__:{name}", t)
        for i, (name, t) in enumerate(t_stats)
    ]

    bonf_results = bonferroni(indexed_t_stats, alpha)
    holm_results = holm(indexed_t_stats, alpha)
    bhy_results = bhy(indexed_t_stats, alpha)

    # Build row-label → result lookup so duplicate display names stay distinct.
    bonf_map = {r.name: r for r in bonf_results}
    holm_map = {r.name: r for r in holm_results}
    bhy_map = {r.name: r for r in bhy_results}

    combined = []
    for i, (name, t) in enumerate(t_stats):
        row_label = f"__row_{i}__:{name}"
        combined.append({
            "name": name,
            "t_stat": t,
            "p_value": bonf_map[row_label].p_value,
            "bonf_reject": bonf_map[row_label].bonf_reject,
            "bonf_adj_p": bonf_map[row_label].bonf_adj_p,
            "holm_reject": holm_map[row_label].holm_reject,
            "holm_adj_p": holm_map[row_label].holm_adj_p,
            "bhy_reject": bhy_map[row_label].bhy_reject,
        })

    # Compute approximate t-stat hurdles
    bonf_t = sp_stats.norm.ppf(1.0 - alpha / (2.0 * M))
    # Holm: first rejection hurdle ≈ same as Bonferroni for rank=1
    holm_t = bonf_t

    n_reject_bonf = sum(1 for r in bonf_results if r.bonf_reject)
    n_reject_holm = sum(1 for r in holm_results if r.holm_reject)
    n_reject_bhy = sum(1 for r in bhy_results if r.bhy_reject)

    # BHY: the step-up procedure rejects ranks 1..k*, so the OPERATIVE
    # hurdle is the critical p-value at the LAST actual rejection,
    # alpha * k* / (M * c_M). The rank-M bound alpha / c_M is only
    # attained when EVERY hypothesis is rejected; reporting it otherwise
    # overstates how permissive the test was (verified M=20 example:
    # reported |t| 2.46 vs operative 3.39). With zero rejections there is
    # no operative hurdle: report NaN.
    if n_reject_bhy > 0:
        c_M = _harmonic_number(M)
        bhy_p_last = alpha * n_reject_bhy / (M * c_M)
        bhy_t = sp_stats.norm.ppf(1.0 - bhy_p_last / 2.0)
    else:
        bhy_t = np.nan

    return MultipleTestingSummary(
        M=M,
        alpha=alpha,
        bonf_threshold_t=float(bonf_t),
        holm_threshold_t=float(holm_t),
        bhy_threshold_t=float(bhy_t),
        n_reject_bonf=n_reject_bonf,
        n_reject_holm=n_reject_holm,
        n_reject_bhy=n_reject_bhy,
        results=combined,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Part 2: Bayesian Inference (Harvey 2017)
# ═══════════════════════════════════════════════════════════════════════════

def minimum_bayes_factor(t_stat: float) -> float:
    """Minimum Bayes Factor: MBF = exp(-t^2 / 2).

    Represents the strongest possible evidence against the null
    (assuming the alternative prior concentrates at the MLE).

    Parameters
    ----------
    t_stat : test statistic (z-score)

    Returns
    -------
    MBF value. Smaller = stronger evidence against null.
    """
    return exp(-t_stat ** 2 / 2.0)


def sd_minimum_bayes_factor(t_stat: float) -> float:
    """Symmetric-Descending Minimum Bayes Factor.

    SD-MBF = -e * p * ln(p), where p = 2*(1 - Phi(|t|)).
    More conservative than MBF (assumes dispersed alternative prior).

    Parameters
    ----------
    t_stat : test statistic

    Returns
    -------
    SD-MBF value.
    """
    p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t_stat)))
    if p <= 0 or p >= 1:
        return 1.0 if p >= 1 else 0.0
    return -e * p * log(p)


def bayesianized_pvalue(
    t_stat: float,
    prior_prob: float = 0.20,
    method: str = "mbf",
) -> float:
    """Compute the Bayesianized p-value: Pr(H0 | data).

    Pr(H0 | data) = BF * odds_null / (1 + BF * odds_null)
    where odds_null = (1 - prior_prob) / prior_prob.

    Parameters
    ----------
    t_stat : test statistic
    prior_prob : prior probability that the effect is TRUE (pi_0)
    method : "mbf" or "sd_mbf"

    Returns
    -------
    Posterior probability that H0 is true (between 0 and 1).
    Higher = more likely null is true = less significant.
    """
    if method == "mbf":
        bf = minimum_bayes_factor(t_stat)
    elif method == "sd_mbf":
        bf = sd_minimum_bayes_factor(t_stat)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'mbf' or 'sd_mbf'.")

    if prior_prob <= 0 or prior_prob >= 1:
        raise ValueError(f"prior_prob must be in (0, 1), got {prior_prob}")

    odds_null = (1.0 - prior_prob) / prior_prob
    return bf * odds_null / (1.0 + bf * odds_null)


def required_t_for_bayesian_threshold(
    prior_prob: float = 0.20,
    threshold: float = 0.05,
    method: str = "mbf",
) -> float:
    """Find the |t|-statistic needed so that Pr(H0 | data) <= threshold.

    Uses bisection search.

    Parameters
    ----------
    prior_prob : prior probability that the effect is TRUE
    threshold : target posterior probability (default 5%)
    method : "mbf" or "sd_mbf"

    Returns
    -------
    The minimum |t|-statistic required.
    """
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        bp = bayesianized_pvalue(mid, prior_prob, method)
        if bp > threshold:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _prior_key(prior: float) -> str:
    """Stable field label for a prior probability.

    Integer percents keep the legacy zero-padded form so the default
    priors produce exactly the historical keys: 0.05 -> "005",
    0.20 -> "020", 0.50 -> "050". Non-integer percents fall back to the
    shortest float repr (0.125 -> "0.125").
    """
    pct = prior * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return f"{int(round(pct)):03d}"
    return f"{prior:g}"


@dataclass
class BayesianResult:
    """Bayesian inference results for one signal.

    ``bayes_p`` / ``bayes_p_sd`` are keyed by prior label so arbitrary
    prior lists stay correctly labeled (the old fixed fields silently
    mislabeled non-default priors and dropped the fourth onward). With
    the default priors the keys are exactly the legacy trio
    ``bayes_p_005`` / ``bayes_p_020`` / ``bayes_p_050`` (and
    ``bayes_p_sd_005`` / ... for the SD-MBF variant).
    """
    name: str
    t_stat: float
    p_value: float
    mbf: float
    sd_mbf: float
    bayes_p: dict[str, float]
    bayes_p_sd: dict[str, float]


def bayesian_report(
    t_stats: list[tuple[str, float]],
    priors: list[float] | None = None,
) -> list[BayesianResult]:
    """Compute Bayesian inference metrics for a list of signals.

    Parameters
    ----------
    t_stats : list of (name, t_statistic) tuples
    priors : list of prior probabilities (default [0.05, 0.20, 0.50]).
        Every prior produces one labeled entry per signal; an empty list
        is a caller error and raises ValueError.
    """
    if priors is None:
        priors = [0.05, 0.20, 0.50]
    if not priors:
        raise ValueError("priors must contain at least one prior probability")

    results = []
    for name, t in t_stats:
        p = 2.0 * (1.0 - sp_stats.norm.cdf(abs(t)))
        mbf = minimum_bayes_factor(t)
        sd = sd_minimum_bayes_factor(t)

        bayes_p = {
            f"bayes_p_{_prior_key(pi)}": bayesianized_pvalue(t, pi, "mbf")
            for pi in priors
        }
        bayes_p_sd = {
            f"bayes_p_sd_{_prior_key(pi)}": bayesianized_pvalue(t, pi, "sd_mbf")
            for pi in priors
        }

        results.append(BayesianResult(
            name=name,
            t_stat=t,
            p_value=p,
            mbf=mbf,
            sd_mbf=sd,
            bayes_p=bayes_p,
            bayes_p_sd=bayes_p_sd,
        ))

    return results
