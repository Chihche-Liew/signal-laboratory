"""Chen-Zimmermann anomaly data loader.

Downloads and caches firm-level characteristics from the Open Source
Asset Pricing project (Chen & Zimmermann 2022, Critical Finance Review).

Data source: https://www.openassetpricing.com
File: signed_predictors_dl_wide.zip — all ~209 predictors in wide format.
Each row is a (permno, yyyymm) observation with signal values as columns.

Usage
-----
    from siglab.data.cz_anomalies import load_cz_predictors, load_cz_signal_doc

    # Full wide DataFrame (permno × yyyymm × 209 signals)
    cz = load_cz_predictors()

    # Signal documentation
    doc = load_cz_signal_doc()

    # Convert a single CZ predictor into a (dates × permnos) panel
    bm_panel = cz_to_panel(cz, "BM")
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from siglab.data import cache

CZ_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "cz_raw"
CZ_ZIP = CZ_RAW_DIR / "signed_predictors_dl_wide.zip"
SIGNALDOC_PATH = (
    Path(__file__).resolve().parents[3] / "reference" / "CrossSection" / "SignalDoc.csv"
)


def load_cz_predictors(use_cache: bool = True) -> pd.DataFrame:
    """Load the full CZ wide-format predictor dataset.

    Returns
    -------
    DataFrame with columns: permno, yyyymm, + ~209 signal columns.
    The yyyymm column is integer (e.g. 196307).
    """
    cache_key = "cz_predictors_wide"
    if use_cache and cache.panel_exists(cache_key):
        return cache.load_panel(cache_key)

    if not CZ_ZIP.exists():
        raise FileNotFoundError(
            f"CZ predictor data not found at {CZ_ZIP}.\n"
            "Download signed_predictors_dl_wide.zip from:\n"
            "  https://drive.google.com/file/d/1avFIMjz_7LoF3p3nO26eqLW5KdRTOdhW/view\n"
            f"and place it at: {CZ_ZIP}"
        )

    # Read the CSV inside the zip
    with zipfile.ZipFile(CZ_ZIP, "r") as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise ValueError(f"No CSV found inside {CZ_ZIP}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, low_memory=False)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    if use_cache:
        cache.save_panel(cache_key, df)

    return df


def load_cz_signal_doc() -> pd.DataFrame:
    """Load SignalDoc.csv — metadata for all CZ anomalies.

    Returns
    -------
    DataFrame with columns: Acronym, Cat.Economic, Sign, Authors, Year, etc.
    """
    if not SIGNALDOC_PATH.exists():
        raise FileNotFoundError(
            f"SignalDoc.csv not found at {SIGNALDOC_PATH}.\n"
            "Make sure reference/CrossSection is cloned."
        )
    return pd.read_csv(SIGNALDOC_PATH, encoding="utf-8")


def list_cz_predictors(cz: pd.DataFrame | None = None) -> list[str]:
    """Return the list of signal column names in the CZ dataset.

    Parameters
    ----------
    cz : CZ wide DataFrame. If None, loads from cache/disk.
    """
    if cz is None:
        cz = load_cz_predictors()
    skip = {"permno", "yyyymm", "date"}
    return [c for c in cz.columns if c not in skip]


def cz_to_panel(
    cz: pd.DataFrame,
    signal_name: str,
) -> pd.DataFrame:
    """Convert a single CZ predictor to a (dates × permnos) panel.

    Parameters
    ----------
    cz : CZ wide-format DataFrame (from load_cz_predictors)
    signal_name : column name of the signal (e.g. "BM", "AssetGrowth")

    Returns
    -------
    Panel DataFrame indexed by month-end dates with permno columns.
    """
    if signal_name not in cz.columns:
        raise KeyError(
            f"Signal '{signal_name}' not found in CZ data. "
            f"Available: {list_cz_predictors(cz)[:10]}..."
        )

    sub = cz[["permno", "yyyymm", signal_name]].copy()

    # Convert yyyymm → month-end date
    sub["date"] = pd.to_datetime(
        sub["yyyymm"].astype(str), format="%Y%m"
    ) + pd.offsets.MonthEnd(0)

    # Pivot to panel
    panel = sub.pivot_table(
        values=signal_name,
        index="date",
        columns="permno",
        aggfunc="first",
    )
    panel.index.name = None
    return panel


def cz_to_all_panels(
    cz: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Convert ALL CZ predictors to (dates × permnos) panels efficiently.

    Builds a (date, permno) → (row, col) index map ONCE, then scatters
    each signal's values into a pre-allocated numpy array. O(N log N) per
    signal (np.unique's sort, for first-valid-wins dedup) with no
    pivot/groupby/unstack overhead. Still ~100x faster than per-signal
    pivot_table.

    Parameters
    ----------
    cz : CZ wide DataFrame (pandas). If None, loads from cache/disk.

    Returns
    -------
    Dict mapping signal name → pandas panel DataFrame (dates × permnos).
    """
    import numpy as np

    if cz is None:
        cz = load_cz_predictors()

    signals = list_cz_predictors(cz)

    # Build date from yyyymm
    yyyymm = cz["yyyymm"].values
    permno = cz["permno"].values

    dates_raw = pd.to_datetime(
        pd.Index(yyyymm).astype(str), format="%Y%m"
    ) + pd.offsets.MonthEnd(0)
    dates_arr = dates_raw.values  # numpy datetime64

    # Sorted unique dates and permnos
    unique_dates = np.sort(np.unique(dates_arr))
    unique_permnos = np.sort(np.unique(permno))
    n_dates = len(unique_dates)
    n_permnos = len(unique_permnos)

    # Build index maps: value → integer index
    date_to_row = {d: i for i, d in enumerate(unique_dates)}
    permno_to_col = {int(p): i for i, p in enumerate(unique_permnos)}

    # Pre-compute row and col indices for every row in the DataFrame
    row_idx = np.array([date_to_row[d] for d in dates_arr], dtype=np.intp)
    col_idx = np.array([permno_to_col[int(p)] for p in permno], dtype=np.intp)

    # Date index and permno columns for the output DataFrames
    date_index = pd.DatetimeIndex(unique_dates)
    permno_columns = unique_permnos.astype(int)

    # Scatter each signal into a (n_dates, n_permnos) array.
    #
    # Duplicate (date, permno) keys are resolved PER COLUMN with
    # first-valid-wins semantics, matching cz_to_panel's
    # pivot_table(aggfunc="first"), which skips NaN per column. The old
    # row-level dedup kept the FIRST duplicate row wholesale, discarding
    # later rows' non-null cells for columns where the first row was NaN
    # (P2-13a). np.unique(return_index=True) picks the first occurrence
    # of each key deterministically (fancy-assignment order with
    # duplicate indices is NOT guaranteed by numpy).
    panels: dict[str, pd.DataFrame] = {}
    for sig in signals:
        if sig not in cz.columns:
            continue
        vals = cz[sig].values.astype(np.float64)
        arr = np.full((n_dates, n_permnos), np.nan, dtype=np.float64)
        finite = np.isfinite(vals)
        fr = row_idx[finite]
        fc = col_idx[finite]
        fv = vals[finite]
        flat = fr * n_permnos + fc
        _, first_pos = np.unique(flat, return_index=True)
        arr[fr[first_pos], fc[first_pos]] = fv[first_pos]
        panels[sig] = pd.DataFrame(arr, index=date_index, columns=permno_columns)

    return panels
