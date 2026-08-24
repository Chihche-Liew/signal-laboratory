"""Fama-French factor data download via pandas_datareader.

Downloads Mkt-RF, SMB, HML, RMW, CMA, Mom, RF from Ken French's data library.
Uses pandas_datareader.famafrench which handles the varying CSV formats robustly.
"""

import pandas as pd

from siglab.data import cache


def download_ff_factors(use_cache: bool = True) -> pd.DataFrame:
    """Download Fama-French 6-factor data (Mkt-RF, SMB, HML, RMW, CMA, Mom, RF).

    Returns
    -------
    DataFrame with monthly factor returns (in decimal, not percent).
    Index: DatetimeIndex (month-end dates).
    Columns: Mkt-RF, SMB, HML, RMW, CMA, Mom, RF
    """
    cache_key = "ff_factors"
    if use_cache and cache.panel_exists(cache_key):
        return cache.load_panel(cache_key)

    import pandas_datareader.data as web

    # FF5 (includes Mkt-RF, SMB, HML, RMW, CMA, RF)
    ff5_raw = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3", "famafrench", start="1960-01-01"
    )[0]

    # Momentum
    mom_raw = web.DataReader(
        "F-F_Momentum_Factor", "famafrench", start="1960-01-01"
    )[0]
    mom_raw = mom_raw.rename(columns={"Mom   ": "Mom", "WML": "Mom"}).filter(
        regex="Mom|WML"
    )
    mom_raw.columns = ["Mom"]

    # Merge and convert percent → decimal
    factors = ff5_raw.join(mom_raw, how="inner") / 100.0

    # Standardise column names (strip whitespace)
    factors.columns = [c.strip() for c in factors.columns]

    # Ensure month-end DatetimeIndex
    factors.index = factors.index.to_timestamp("M")

    # Keep standard columns in a fixed order
    keep = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "RF"]
    factors = factors[[c for c in keep if c in factors.columns]]

    if use_cache:
        cache.save_panel(cache_key, factors)

    return factors


def download_q_factors(use_cache: bool = True) -> pd.DataFrame:
    """Download Hou-Xue-Zhang q-factor data from global-q.org.

    Returns
    -------
    DataFrame with monthly factor returns (in decimal, not percent).
    Index: DatetimeIndex (month-end dates).
    Columns: R_MKT, R_ME, R_IA, R_ROE, R_F
    (The source file also includes R_EG, but we do not use it.)
    """
    cache_key = "q_factors"
    if use_cache and cache.panel_exists(cache_key):
        return cache.load_panel(cache_key)

    url = (
        "https://global-q.org/uploads/1/2/2/6/122679606/"
        "q5_factors_monthly_2024.csv"
    )
    raw = pd.read_csv(url)

    # Build month-end DatetimeIndex from year/month columns
    raw["date"] = pd.to_datetime(
        raw["year"].astype(str) + "-" + raw["month"].astype(str) + "-01"
    ) + pd.offsets.MonthEnd(0)
    raw = raw.set_index("date").drop(columns=["year", "month"])

    # Convert percent → decimal
    raw = raw / 100.0

    if use_cache:
        cache.save_panel(cache_key, raw)

    return raw


def get_factor_columns(model: int) -> list[str]:
    """Get factor column names for a given model number.

    Parameters
    ----------
    model : 1=CAPM, 3=FF3, 4=FF4(Carhart), 5=FF5, 6=FF6,
            7=q-factor (HXZ 2015)

    For model 7 the columns refer to the q-factor DataFrame
    returned by ``download_q_factors()``.
    """
    if model == 1:
        return ["Mkt-RF"]
    elif model == 3:
        return ["Mkt-RF", "SMB", "HML"]
    elif model == 4:
        return ["Mkt-RF", "SMB", "HML", "Mom"]
    elif model == 5:
        return ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif model == 6:
        return ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
    elif model == 7:
        return ["R_MKT", "R_ME", "R_IA", "R_ROE"]
    else:
        raise ValueError(
            f"Unknown model number: {model}. Use 1, 3, 4, 5, 6, or 7."
        )
