"""CRSP data download and panel construction.

Downloads monthly stock returns, prices, shares outstanding, and exchange
codes from CRSP via WRDS. Constructs panel matrices (dates × permnos).
"""

import numpy as np
import pandas as pd

from siglab.data import cache, wrds_conn

# The four panels that make up one cached CRSP bundle. A bundle is only
# usable if ALL of them exist — a crash between saves must read as a
# cache miss, not wedge every later run with FileNotFoundError (P2-15).
CRSP_PANEL_KEYS = ("ret", "me", "nyse", "exchcd")


def download_crsp_monthly(
    start: str = "1960-01-01", end: str = "2024-12-31",
    shrcd_filter: tuple[int, ...] = (10, 11),
) -> pd.DataFrame:
    """Download CRSP monthly stock file from WRDS with delisting return adjustment.

    Parameters
    ----------
    start, end : date strings for sample period
    shrcd_filter : share codes to include (10, 11 = common stock)

    Returns
    -------
    DataFrame with columns: permno, date, ret, retx, prc, shrout,
    shrcd, exchcd, siccd.  ret is adjusted for delisting returns
    (Shumway 1997; Beaver-McNichols-Price 2007).
    """
    shrcd_str = ", ".join(str(s) for s in shrcd_filter)

    # Main monthly file — LEFT JOIN delisting table to pick up dlret/dlstcd
    # for the month a stock delists (matching on permno and year-month).
    sql = f"""
    SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.shrout,
           a.vol, b.shrcd, b.exchcd, b.siccd,
           c.dlret, c.dlstcd
    FROM crsp_a_stock.msf AS a
    INNER JOIN crsp_a_stock.msenames AS b
        ON a.permno = b.permno
        AND a.date >= b.namedt
        AND a.date <= b.nameendt
    LEFT JOIN crsp_a_stock.msedelist AS c
        ON a.permno = c.permno
        AND DATE_TRUNC('month', a.date) = DATE_TRUNC('month', c.dlstdt)
    WHERE a.date BETWEEN '{start}' AND '{end}'
        AND b.shrcd IN ({shrcd_str})
    ORDER BY a.permno, a.date
    """
    df = wrds_conn.raw_sql(sql)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype(int)

    # Adjust last-month return for delisting (Shumway 1997).
    # When dlret is available, compound it with the monthly return.
    # When dlret is missing and delisting code indicates a bad exit
    # (codes 200–399 = performance/dropped), impute -30% (Beaver-McNichols-Price 2007).
    if "dlret" in df.columns:
        bad_exit = df["dlstcd"].between(200, 399).fillna(False)
        is_delisting = df["dlret"].notna() | bad_exit

        # Effective delisting multiplier: dlret if available, -30% for bad exits
        dlret_fill = df["dlret"].copy()
        dlret_fill.loc[dlret_fill.isna() & bad_exit] = -0.30
        dlret_fill = dlret_fill.fillna(0.0)

        # Compound only on delisting months; leave other months unchanged
        ret_adj = (1.0 + df["ret"].fillna(0.0)) * (1.0 + dlret_fill) - 1.0
        df["ret"] = np.where(is_delisting, ret_adj, df["ret"])

        df = df.drop(columns=["dlret", "dlstcd"], errors="ignore")

    # Market equity = |price| * shares outstanding (in thousands)
    # CRSP shrout is in thousands, prc can be negative (bid-ask midpoint)
    df["me"] = np.abs(df["prc"]) * df["shrout"]

    return df


def make_return_panel(crsp: pd.DataFrame) -> pd.DataFrame:
    """Pivot CRSP returns into (dates × permnos) panel.

    Parameters
    ----------
    crsp : raw CRSP DataFrame with permno, date, ret columns

    Returns
    -------
    DataFrame: index=monthly dates, columns=permnos, values=returns.
    """
    panel = crsp.pivot_table(
        values="ret", index="date", columns="permno", aggfunc="first"
    )
    panel.index = pd.to_datetime(panel.index)
    return panel


def make_me_panel(crsp: pd.DataFrame) -> pd.DataFrame:
    """Pivot market equity into (dates × permnos) panel."""
    panel = crsp.pivot_table(
        values="me", index="date", columns="permno", aggfunc="first"
    )
    panel.index = pd.to_datetime(panel.index)
    return panel


def make_nyse_indicator(crsp: pd.DataFrame) -> pd.DataFrame:
    """Create panel indicating NYSE membership (exchcd == 1).

    Returns
    -------
    Boolean DataFrame: True where stock is listed on NYSE.
    """
    crsp = crsp.copy()
    crsp["nyse"] = (crsp["exchcd"] == 1).astype(float)
    panel = crsp.pivot_table(
        values="nyse", index="date", columns="permno", aggfunc="first"
    )
    panel.index = pd.to_datetime(panel.index)
    # A pivot cell is NaN when the stock did not exist that month;
    # NaN.astype(bool) is True, which stamped absent stock-months as NYSE
    # members (P2-11). Absent must be False.
    return panel.fillna(0.0).astype(bool)


def make_exchange_panel(crsp: pd.DataFrame) -> pd.DataFrame:
    """Pivot exchange codes into (dates × permnos) panel."""
    panel = crsp.pivot_table(
        values="exchcd", index="date", columns="permno", aggfunc="first"
    )
    panel.index = pd.to_datetime(panel.index)
    return panel


def crsp_cache_key(start: str, end: str) -> str:
    """Cache-key prefix for the CRSP monthly panel bundle (see CRSP_PANEL_KEYS)."""
    return f"crsp_{start}_{end}"


def get_crsp_panels(
    start: str = "1960-01-01", end: str = "2024-12-31",
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Download and cache all CRSP panels.

    Returns
    -------
    Dict with keys: 'ret', 'me', 'nyse', 'exchcd' — each a (dates × permnos) panel.
    """
    cache_key = crsp_cache_key(start, end)

    if use_cache and all(
        cache.panel_exists(f"{cache_key}_{key}") for key in CRSP_PANEL_KEYS
    ):
        return {
            key: cache.load_panel(f"{cache_key}_{key}")
            for key in CRSP_PANEL_KEYS
        }

    crsp = download_crsp_monthly(start, end)

    panels = {
        "ret": make_return_panel(crsp),
        "me": make_me_panel(crsp),
        "nyse": make_nyse_indicator(crsp),
        "exchcd": make_exchange_panel(crsp),
    }

    if use_cache:
        # Overwrites any partial leftovers from an interrupted earlier save.
        for key, panel in panels.items():
            cache.save_panel(f"{cache_key}_{key}", panel)

    return panels
