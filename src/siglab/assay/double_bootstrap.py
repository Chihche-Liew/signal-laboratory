"""Double-bootstrap FDR calibration (Harvey & Liu 2020, JF).

Jointly calibrates Type I (false discovery) and Type II (missed discovery)
error rates using the actual correlation structure among signals.

Algorithm
---------
Stage 1 (Ranking Bootstrap): I=100 iterations
  1. Bootstrap time periods → compute FMB t-stats for all N signals
  2. Rank by configured t-stat rule; label top p₀*N as "true", rest as "null"
  3. Construct pseudo-dataset with KNOWN truth labels

Stage 2 (Evaluation Bootstrap): for each pseudo-dataset, J=1000 iterations
  4. Bootstrap time periods again
  5. Apply testing rule (t-threshold)
  6. Record TP, FP, FN, TN

Output
------
  TYPE1(t*) = E[FP / (FP + TP)]           # realized FDR
  TYPE2(t*) = E[FN / (FN + TN)]           # realized miss rate
  ORATIO(t*) = E[FP / FN]                 # miss-to-false ratio

Parallel execution uses ProcessPoolExecutor for n_workers > 1.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from siglab.utils.stats import newey_west_t


_DENSE_STACK_LIMIT_BYTES = 4 * 1024 ** 3
_VALID_MODES = {"signed", "two_sided"}
_WORKER_MATRIX: np.ndarray | None = None


@dataclass
class DoubleBootstrapResult:
    """Output of the double-bootstrap calibration."""
    p0: float                     # assumed fraction of true signals
    t_thresholds: list[float]     # grid of t-thresholds evaluated
    type1: list[float]            # realized FDR at each threshold
    type2: list[float]            # realized miss rate at each threshold
    oratio: list[float]           # FP/FN ratio at each threshold

    # Data-dependent hurdle at target FDR
    t_hurdle_005: float           # t for TYPE1 = 5%
    t_hurdle_010: float           # t for TYPE1 = 10%

    I: int                        # ranking bootstrap reps
    J: int                        # evaluation bootstrap reps
    N: int                        # number of signals
    n_true: int = 0                # number of columns treated as true
    ranking_mode: str = "signed"   # signed or two_sided ranking rule
    testing_mode: str = "signed"   # signed or two_sided rejection rule
    nw_lags: int = 0               # Newey-West lags for column t-stats

    # ORATIO-based hurdles (Harvey & Liu 2020 Section III.B)
    t_oratio_1_0: float = np.nan  # t where ORATIO <= 1.0 (equal cost)
    t_oratio_0_2: float = np.nan  # t where ORATIO <= 0.2 (FP costs 5x)
    t_oratio_0_1: float = np.nan  # t where ORATIO <= 0.1 (FP costs 10x)


def _orient_series_matrix(matrix: np.ndarray, signs: np.ndarray | None) -> np.ndarray:
    """Orient each strategy column so larger positive values support discovery."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("strategy matrix must be 2-D")
    if signs is None:
        return arr

    signs_arr = np.asarray(signs, dtype=float)
    if signs_arr.shape != (arr.shape[1],):
        raise ValueError("signs must have shape (n_columns,)")
    return arr * signs_arr[np.newaxis, :]


def _mean_t_stats(matrix: np.ndarray, *, nw_lags: int = 0) -> np.ndarray:
    """Compute one t-statistic per column of a monthly outcome matrix."""
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2-D")
    if nw_lags < 0:
        raise ValueError("nw_lags must be non-negative")

    if nw_lags > 0:
        out = np.zeros(arr.shape[1], dtype=float)
        for col in range(arr.shape[1]):
            series = arr[:, col]
            finite = np.isfinite(series)
            if finite.sum() < 2:
                continue
            cleaned = np.where(finite, series, np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                _, t_stat = newey_west_t(cleaned, nlags=nw_lags)
            value = float(np.ravel(t_stat)[0])
            out[col] = value if np.isfinite(value) else 0.0
        return out

    valid = np.isfinite(arr)
    counts = valid.sum(axis=0)
    sums = np.where(valid, arr, 0.0).sum(axis=0)
    means = np.divide(
        sums,
        counts,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 0,
    )
    centered = np.where(valid, arr - means[np.newaxis, :], 0.0)
    ss = np.square(centered).sum(axis=0)
    var = np.divide(
        ss,
        counts - 1,
        out=np.full(arr.shape[1], np.nan, dtype=float),
        where=counts > 1,
    )
    se = np.sqrt(var) / np.sqrt(np.maximum(counts, 1))
    t_stats = np.divide(
        means,
        se,
        out=np.zeros(arr.shape[1], dtype=float),
        where=(counts > 1) & np.isfinite(se) & (se > 0),
    )
    return np.where(np.isfinite(t_stats), t_stats, 0.0)


def _construct_pseudo_matrix(
    original: np.ndarray,
    bootstrapped: np.ndarray,
    is_true: np.ndarray,
) -> np.ndarray:
    """Construct a Harvey-Liu pseudo matrix with known truth labels."""
    x0 = np.asarray(original, dtype=float)
    xi = np.asarray(bootstrapped, dtype=float)
    if x0.ndim != 2 or xi.ndim != 2 or x0.shape != xi.shape:
        raise ValueError("original and bootstrapped matrices must have the same 2-D shape")

    truth = np.asarray(is_true)
    if truth.dtype != np.bool_ or truth.shape != (x0.shape[1],):
        raise ValueError("is_true must be a bool array with shape (n_columns,)")

    original_means = np.nanmean(x0, axis=0)
    bootstrap_means = np.nanmean(xi, axis=0)
    target_means = np.where(truth, bootstrap_means, 0.0)
    return x0 - original_means[np.newaxis, :] + target_means[np.newaxis, :]


def _batch_monthly_fmb_betas(sigs: np.ndarray, rets: np.ndarray) -> np.ndarray:
    """Compute monthly one-signal Fama-MacBeth slopes for a dense panel."""
    sigs_arr = np.asarray(sigs, dtype=float)
    rets_arr = np.asarray(rets, dtype=float)
    if sigs_arr.ndim != 3:
        raise ValueError("sigs must have shape (T, N_sig, N_stocks)")
    if rets_arr.ndim != 2:
        raise ValueError("rets must have shape (T, N_stocks)")

    T, N_sig, N_stocks = sigs_arr.shape
    if rets_arr.shape != (T, N_stocks):
        raise ValueError("rets shape must match sigs time and stock dimensions")

    betas = np.full((max(T - 1, 0), N_sig), np.nan, dtype=float)
    for t in range(1, T):
        y = rets_arr[t]
        y_finite = np.isfinite(y)
        for s in range(N_sig):
            x = sigs_arr[t - 1, s]
            valid = y_finite & np.isfinite(x)
            if valid.sum() < 2:
                continue
            x_valid = x[valid]
            x_dm = x_valid - x_valid.mean()
            denom = float(np.dot(x_dm, x_dm))
            if not np.isfinite(denom) or denom <= 1e-15:
                continue
            betas[t - 1, s] = float(np.dot(x_dm, y[valid]) / denom)
    return betas


def _batch_fmb_t(sigs: np.ndarray, rets: np.ndarray) -> np.ndarray:
    """Vectorized FMB: compute t-stats for ALL signals simultaneously.

    sigs: (T, N_sig, N_stocks) — all signals stacked (may contain NaN)
    rets: (T, N_stocks) — returns (may contain NaN)

    Returns: (N_sig,) array of t-statistics.

    Kept as the legacy dense-panel helper; production double-bootstrap now
    works on the monthly beta matrix returned by _batch_monthly_fmb_betas.
    """
    return _mean_t_stats(_batch_monthly_fmb_betas(sigs, rets), nw_lags=0)


def _simple_fmb_t(sig: np.ndarray, ret: np.ndarray) -> float:
    """Single-signal FMB (fallback). Use _batch_fmb_t for bulk."""
    t = _batch_fmb_t(sig[np.newaxis].transpose(1, 0, 2), ret)
    return float(t[0])


def _count_outcomes(
    t_eval: np.ndarray,
    is_true: np.ndarray,
    t_grid_arr: np.ndarray,
    *,
    testing_mode: str,
) -> dict[str, np.ndarray]:
    """Count outcomes and realized rates for one evaluation bootstrap."""
    if testing_mode == "signed":
        rejected = t_eval[:, None] > t_grid_arr[None, :]
    elif testing_mode == "two_sided":
        rejected = np.abs(t_eval)[:, None] > t_grid_arr[None, :]
    else:
        raise ValueError("testing_mode must be 'signed' or 'two_sided'")

    truth = is_true[:, None]
    tp = (rejected & truth).sum(axis=0).astype(float)
    fp = (rejected & ~truth).sum(axis=0).astype(float)
    fn = (~rejected & truth).sum(axis=0).astype(float)
    tn = (~rejected & ~truth).sum(axis=0).astype(float)

    type1 = np.divide(fp, fp + tp, out=np.zeros_like(fp), where=(fp + tp) > 0)
    type2 = np.divide(fn, fn + tn, out=np.zeros_like(fn), where=(fn + tn) > 0)
    oratio = np.divide(fp, fn, out=np.zeros_like(fp), where=fn > 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "type1": type1,
        "type2": type2,
        "oratio": oratio,
    }


def _init_matrix_worker(matrix: np.ndarray) -> None:
    """Initialize process-local matrix state for compact matrix bootstraps."""
    global _WORKER_MATRIX
    _WORKER_MATRIX = np.asarray(matrix, dtype=float)


def _ranking_and_eval_matrix_batch(args: tuple) -> dict[str, np.ndarray]:
    """Run one ranking bootstrap and its evaluation bootstraps on D x N data."""
    X0, p0, J, t_grid, seed, ranking_mode, testing_mode, nw_lags = args
    if X0 is None:
        X0 = _WORKER_MATRIX
    if X0 is None:
        raise RuntimeError("worker matrix is not initialized")
    rng = np.random.RandomState(seed)
    D, N = X0.shape
    t_grid_arr = np.asarray(t_grid, dtype=float)
    n_thresholds = len(t_grid_arr)
    n_true = min(max(int(np.floor(p0 * N)), 0), N)

    idx_rank = rng.choice(D, size=D, replace=True)
    Xi = X0[idx_rank]
    t_stats = _mean_t_stats(Xi, nw_lags=nw_lags)
    if ranking_mode == "signed":
        ranked = np.argsort(-t_stats)
    elif ranking_mode == "two_sided":
        ranked = np.argsort(-np.abs(t_stats))
    else:
        raise ValueError("ranking_mode must be 'signed' or 'two_sided'")

    is_true = np.zeros(N, dtype=bool)
    if n_true > 0:
        is_true[ranked[:n_true]] = True

    Yi = _construct_pseudo_matrix(X0, Xi, is_true)

    totals = {
        "tp": np.zeros(n_thresholds, dtype=float),
        "fp": np.zeros(n_thresholds, dtype=float),
        "fn": np.zeros(n_thresholds, dtype=float),
        "tn": np.zeros(n_thresholds, dtype=float),
        "type1": np.zeros(n_thresholds, dtype=float),
        "type2": np.zeros(n_thresholds, dtype=float),
        "oratio": np.zeros(n_thresholds, dtype=float),
    }
    for _ in range(J):
        idx_eval = rng.choice(D, size=D, replace=True)
        t_eval = _mean_t_stats(Yi[idx_eval], nw_lags=nw_lags)
        eval_result = _count_outcomes(t_eval, is_true, t_grid_arr, testing_mode=testing_mode)
        for key, values in eval_result.items():
            totals[key] += values

    return totals


def _result_from_rates(
    *,
    p0: float,
    t_grid: list[float],
    type1_sum: np.ndarray,
    type2_sum: np.ndarray,
    oratio_sum: np.ndarray,
    total_evals: int,
    I: int,
    J: int,
    N: int,
    n_true: int,
    ranking_mode: str,
    testing_mode: str,
    nw_lags: int,
) -> DoubleBootstrapResult:
    """Build a DoubleBootstrapResult from accumulated realized rates."""
    n_thresholds = len(t_grid)
    denom = float(total_evals)
    type1 = (type1_sum / denom).tolist()
    type2 = (type2_sum / denom).tolist()
    oratio = (oratio_sum / denom).tolist()

    def _find_hurdle(target_fdr: float) -> float:
        for k in range(n_thresholds):
            if type1[k] <= target_fdr:
                return float(t_grid[k])
        return float(t_grid[-1])

    def _find_oratio_hurdle(target: float) -> float:
        for k in range(n_thresholds):
            if oratio[k] <= target:
                return float(t_grid[k])
        return float(t_grid[-1])

    return DoubleBootstrapResult(
        p0=p0,
        t_thresholds=t_grid,
        type1=type1,
        type2=type2,
        oratio=oratio,
        t_hurdle_005=_find_hurdle(0.05),
        t_hurdle_010=_find_hurdle(0.10),
        I=I,
        J=J,
        N=N,
        n_true=n_true,
        ranking_mode=ranking_mode,
        testing_mode=testing_mode,
        nw_lags=nw_lags,
        t_oratio_1_0=_find_oratio_hurdle(1.0),
        t_oratio_0_2=_find_oratio_hurdle(0.2),
        t_oratio_0_1=_find_oratio_hurdle(0.1),
    )


def run_double_bootstrap_from_matrix(
    strategy_matrix: np.ndarray,
    *,
    p0: float = 0.10,
    I: int = 100,
    J: int = 200,
    t_grid: list[float] | None = None,
    n_workers: int = 12,
    ranking_mode: str = "signed",
    testing_mode: str = "signed",
    nw_lags: int = 0,
) -> DoubleBootstrapResult:
    """Run Harvey-Liu double-bootstrap on a D x N monthly outcome matrix."""
    X0 = np.asarray(strategy_matrix, dtype=float)
    if X0.ndim != 2:
        raise ValueError("strategy_matrix must be 2-D")
    if X0.shape[0] == 0:
        raise ValueError("strategy_matrix must have at least one row")
    if X0.shape[1] == 0:
        raise ValueError("strategy_matrix must have at least one column")
    if not 0.0 <= p0 <= 1.0:
        raise ValueError("p0 must be between 0 and 1")
    if ranking_mode not in _VALID_MODES:
        raise ValueError("ranking_mode must be 'signed' or 'two_sided'")
    if testing_mode not in _VALID_MODES:
        raise ValueError("testing_mode must be 'signed' or 'two_sided'")
    if I < 1 or J < 1:
        raise ValueError("I and J must be positive")
    if n_workers < 1:
        raise ValueError("n_workers must be positive")
    if nw_lags < 0:
        raise ValueError("nw_lags must be non-negative")

    if t_grid is None:
        t_grid = list(np.linspace(1.0, 5.0, 41))
    t_grid_arr = np.asarray(t_grid, dtype=float)
    if t_grid_arr.ndim != 1 or len(t_grid_arr) == 0:
        raise ValueError("t_grid must be a non-empty 1-D sequence")
    t_grid_list = t_grid_arr.tolist()

    N = X0.shape[1]
    n_true = min(max(int(np.floor(p0 * N)), 0), N)
    n_thresholds = len(t_grid_list)
    type1_sum = np.zeros(n_thresholds, dtype=float)
    type2_sum = np.zeros(n_thresholds, dtype=float)
    oratio_sum = np.zeros(n_thresholds, dtype=float)

    from tqdm import tqdm

    pool = None
    if n_workers == 1:
        tasks = (
            (X0, p0, J, t_grid_list, seed, ranking_mode, testing_mode, nw_lags)
            for seed in range(I)
        )
        iterator = map(_ranking_and_eval_matrix_batch, tasks)
    else:
        tasks = (
            (None, p0, J, t_grid_list, seed, ranking_mode, testing_mode, nw_lags)
            for seed in range(I)
        )
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_matrix_worker,
            initargs=(X0,),
        )
        iterator = pool.map(_ranking_and_eval_matrix_batch, tasks)

    try:
        for batch_result in tqdm(
            iterator,
            total=I,
            desc=f"  bootstrap(p0={p0:.0%})",
            unit="rank",
        ):
            type1_sum += batch_result["type1"]
            type2_sum += batch_result["type2"]
            oratio_sum += batch_result["oratio"]
    finally:
        if pool is not None:
            pool.shutdown()

    return _result_from_rates(
        p0=p0,
        t_grid=t_grid_list,
        type1_sum=type1_sum,
        type2_sum=type2_sum,
        oratio_sum=oratio_sum,
        total_evals=I * J,
        I=I,
        J=J,
        N=N,
        n_true=n_true,
        ranking_mode=ranking_mode,
        testing_mode=testing_mode,
        nw_lags=nw_lags,
    )


def run_double_bootstrap(
    signal_panels: list[np.ndarray],
    ret_panel: np.ndarray,
    *,
    p0: float = 0.10,
    I: int = 100,
    J: int = 200,
    t_grid: list[float] | None = None,
    n_workers: int = 12,
) -> DoubleBootstrapResult:
    """Run the Harvey & Liu (2020) double-bootstrap FDR calibration.

    Parameters
    ----------
    signal_panels : list of N arrays, each (T, N_stocks)
    ret_panel : (T, N_stocks) return array
    p0 : prior fraction of true signals (default 10%)
    I : number of ranking bootstraps (default 100)
    J : evaluation bootstraps per ranking (default 200)
    t_grid : t-stat thresholds to evaluate (default linspace 1.0–5.0)
    n_workers : ProcessPoolExecutor workers (default 12)

    Returns
    -------
    DoubleBootstrapResult with TYPE1, TYPE2, ORATIO profiles.
    """
    rets = np.asarray(ret_panel, dtype=float)
    if rets.ndim != 2:
        raise ValueError("ret_panel must have shape (T, N_stocks)")
    T, N_stocks = rets.shape
    N_sig = len(signal_panels)
    if N_sig == 0:
        raise ValueError("signal_panels must contain at least one signal")

    for idx, sig in enumerate(signal_panels):
        sig_arr = np.asarray(sig)
        if sig_arr.shape != (T, N_stocks):
            raise ValueError(
                f"signal_panels[{idx}] must have shape {(T, N_stocks)}, "
                f"got {sig_arr.shape}"
            )

    estimated_bytes = T * N_sig * N_stocks * np.dtype(float).itemsize
    if estimated_bytes > _DENSE_STACK_LIMIT_BYTES:
        raise MemoryError(
            "Dense signal stack would exceed 4 GiB; use "
            "run_double_bootstrap_from_matrix or the compact CLI path."
        )

    sigs_all = np.stack([np.asarray(sig, dtype=float) for sig in signal_panels], axis=1)
    monthly_betas = _batch_monthly_fmb_betas(sigs_all, rets)
    return run_double_bootstrap_from_matrix(
        monthly_betas,
        p0=p0,
        I=I,
        J=J,
        t_grid=t_grid,
        n_workers=n_workers,
        ranking_mode="two_sided",
        testing_mode="two_sided",
        nw_lags=0,
    )
