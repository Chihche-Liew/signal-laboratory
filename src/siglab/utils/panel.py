"""Panel data utilities for time-series cross-sectional data.

Mirrors FillMonths.m, fillVar.m, calcPanelCorrels.m from AssayingAnomalies.
Convention: panels are pd.DataFrame with DatetimeIndex rows, permno (int) columns.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def fill_months(data: pd.DataFrame, persist: int = 12) -> pd.DataFrame:
    """Forward-fill lower-frequency data to monthly frequency.

    The panel is first reindexed to the complete month-end calendar spanning
    its dates (original non-month-end stamps are kept via union), so
    `persist` counts CALENDAR months — ffill's `limit` alone counts rows,
    which overshoots on gappy indexes (P2-12).

    Parameters
    ----------
    data : DataFrame (dates × permnos) — sparse panel with gaps
    persist : max calendar months to carry a value forward

    Returns
    -------
    Forward-filled DataFrame at monthly frequency (one row per month-end).
    """
    if len(data.index) == 0:
        return data.copy()
    full_index = pd.date_range(data.index.min(), data.index.max(), freq="ME")
    full_index = full_index.union(data.index)
    return data.reindex(full_index).ffill(limit=persist)


def cross_sectional_rank(var: pd.DataFrame) -> pd.DataFrame:
    """Convert each cross-section to percentile ranks scaled to [-0.5, 0.5].

    Parameters
    ----------
    var : (nMonths × nStocks) panel of raw signal values

    Returns
    -------
    Panel of cross-sectional ranks, each row in [-0.5, 0.5].
    NaN values remain NaN.
    """
    # rank(pct=True) gives ranks in (0, 1]. Shift to [-0.5, 0.5].
    ranked = var.rank(axis=1, pct=True, na_option="keep")
    return ranked - 0.5


def fill_var(var: pd.DataFrame, me: pd.DataFrame) -> pd.DataFrame:
    """Standardize signal to cross-sectional ranks and impute missing values.

    Mirrors fillVar.m:
    1. Convert to cross-sectional ranks scaled to [-0.5, 0.5]
    2. Fill missing values with 0 (cross-sectional median) where me exists
    3. Set to NaN where me is NaN

    Parameters
    ----------
    var : (nMonths × nStocks) raw signal panel
    me : (nMonths × nStocks) market equity panel (defines universe)

    Returns
    -------
    Standardized, filled signal panel.
    """
    ranked = cross_sectional_rank(var)

    # Fill missing with 0 (median rank) where stock is in universe (me exists)
    has_me = me.notna()
    filled = ranked.where(ranked.notna(), other=0.0)
    filled = filled.where(has_me, other=np.nan)

    return filled


def warm_index_engines(*frames: pd.DataFrame) -> None:
    """Force-build the lazy index hash engines of DataFrames, single-threaded.

    pandas builds an Index's lookup engine on first use, and that build is
    not thread-safe: concurrent first lookups on a shared panel can corrupt
    the engine and raise spurious ``KeyError: [...] not in index``. Call this
    on every panel shared across ThreadPoolExecutor workers before the pool
    starts.
    """
    for df in frames:
        for idx in (df.index, df.columns):
            if len(idx):
                idx.get_indexer(idx[:1])


def calc_panel_correls(
    var1: pd.DataFrame, var2: pd.DataFrame
) -> tuple[float, float]:
    """Compute Pearson and Spearman correlations for panel data.

    Pools all (date, stock) observations and computes correlations.

    Parameters
    ----------
    var1, var2 : (nMonths × nStocks) panels (must share index/columns)

    Returns
    -------
    (pearson_corr, spearman_corr) tuple of floats.
    """
    if not (var1.index.equals(var2.index) and var1.columns.equals(var2.columns)):
        raise ValueError(
            "calc_panel_correls requires panels with identical index and "
            "columns: .values.ravel() pairs cells positionally, so label "
            "misalignment silently correlates the wrong observations (P2-12)"
        )

    # Align and flatten
    v1 = var1.values.ravel()
    v2 = var2.values.ravel()

    # Remove NaN pairs
    mask = np.isfinite(v1) & np.isfinite(v2)
    v1c = v1[mask]
    v2c = v2[mask]

    if len(v1c) < 3:
        return np.nan, np.nan

    pearson = np.corrcoef(v1c, v2c)[0, 1]
    spearman = sp_stats.spearmanr(v1c, v2c).statistic

    return float(pearson), float(spearman)
