"""CCM link-table download.

Only the linking-table helper lives here. The former in-module CRSP/Compustat
merge helpers were removed 2026-07-10 (P2-7): they had zero callers anywhere
in the repo and their availability window started one month early versus the
standard July(t+1)-June(t+2) convention — a 1-month look-ahead. Production
merging goes through build_comp_ccm in scripts/data/refresh_compustat_cache.py.
"""

import pandas as pd

from siglab.data import wrds_conn


def download_ccm_link(
    start: str = "1960-01-01", end: str = "2024-12-31"
) -> pd.DataFrame:
    """Download CCM linking table from WRDS.

    Returns
    -------
    DataFrame with columns: gvkey, permno, linkdt, linkenddt, linktype, linkprim.
    """
    sql = f"""
    SELECT gvkey, lpermno AS permno, linkdt, linkenddt, linktype, linkprim
    FROM crsp_a_ccm.ccmxpf_lnkhist
    WHERE linktype IN ('LC', 'LU')
        AND linkprim IN ('P', 'C')
        AND linkdt <= '{end}'
        AND (linkenddt >= '{start}' OR linkenddt IS NULL)
    ORDER BY gvkey, linkdt
    """
    df = wrds_conn.raw_sql(sql)
    df["linkdt"] = pd.to_datetime(df["linkdt"])
    df["linkenddt"] = pd.to_datetime(df["linkenddt"])
    # Fill missing end dates with far future
    df["linkenddt"] = df["linkenddt"].fillna(pd.Timestamp("2099-12-31"))
    df["permno"] = df["permno"].astype(int)
    return df
