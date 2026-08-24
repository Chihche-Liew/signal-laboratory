"""Rolling / moving window operations.

Mirrors moving.m from AssayingAnomalies.
"""

from typing import Callable

import numpy as np


def moving(
    x: np.ndarray,
    window: int,
    func: Callable[[np.ndarray], float] = np.nanmean,
) -> np.ndarray:
    """Apply a function over a rolling window.

    Parameters
    ----------
    x : (n,) or (n, m) array — input data
    window : window size (number of observations)
    func : aggregation function applied to each window (default: nanmean)

    Returns
    -------
    Array of same shape as x with rolling results.
    Leading values where window is incomplete are computed with available data.
    """
    x = np.asarray(x, dtype=np.float64)

    if x.ndim == 1:
        n = len(x)
        out = np.full(n, np.nan)
        for i in range(n):
            start = max(0, i - window + 1)
            out[i] = func(x[start : i + 1])
        return out

    # 2D: apply per column
    n, m = x.shape
    out = np.full_like(x, np.nan)
    for j in range(m):
        for i in range(n):
            start = max(0, i - window + 1)
            out[i, j] = func(x[start : i + 1, j])
    return out


def expanding(
    x: np.ndarray,
    func: Callable[[np.ndarray], float] = np.nanmean,
    min_periods: int = 1,
) -> np.ndarray:
    """Apply a function over an expanding window.

    Parameters
    ----------
    x : (n,) array
    func : aggregation function
    min_periods : minimum observations before producing a value

    Returns
    -------
    Array with expanding window results.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(n):
        if i + 1 >= min_periods:
            out[i] = func(x[: i + 1])
    return out
