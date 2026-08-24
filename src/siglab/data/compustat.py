"""Compustat data download for fundamental accounting variables.

Downloads annual and quarterly financial statement data from WRDS.
"""

import pandas as pd

from siglab.data import wrds_conn


# Common annual variables for asset pricing research
ANNUAL_VARS = [
    "gvkey", "datadate", "fyear", "fyr",
    "at",     # total assets
    "ceq",    # common equity
    "ib",     # income before extraordinary items
    "sale",   # net sales
    "revt",   # revenue total
    "cogs",   # cost of goods sold
    "xsga",   # selling, general & admin expense
    "xrd",    # R&D expense
    "capx",   # capital expenditures
    "aqc",    # acquisitions (cash flow statement)
    "dp",     # depreciation
    "ppegt",  # gross property, plant & equipment
    "ppent",  # net property, plant & equipment
    "invt",   # inventories
    "act",    # current assets
    "lct",    # current liabilities
    "dltt",   # long-term debt
    "dlc",    # debt in current liabilities
    "txdi",   # deferred taxes
    "itcb",   # investment tax credit (balance sheet)
    "pstkrv", # preferred stock — redemption value
    "pstkl",  # preferred stock — liquidating value
    "pstk",   # preferred stock — carrying value
    "txditc", # deferred taxes and investment tax credit
    "seq",    # stockholders' equity
    "lt",     # total liabilities
    "ni",     # net income
    "oibdp",  # operating income before depreciation
    "oiadp",  # operating income after depreciation
    "che",    # cash and short-term investments
    "rect",   # receivables total
    "aco",    # current assets — other
    "ap",     # accounts payable
    "xacc",   # accrued expenses
    "intan",  # intangible assets
    "ao",     # assets — other
    "lo",     # liabilities — other
    "spi",    # special items
    "gdwl",   # goodwill
    "emp",    # employees
    "txp",    # income taxes payable
    "xint",   # interest expense
    "csho",   # common shares outstanding
    "ajex",   # adjustment factor
    "prcc_f", # price close — fiscal year end
    # Cash flow statement
    "oancf",  # operating activities — net cash flow
    "dltis",  # long-term debt issuance
    "dltr",   # long-term debt retirement
    "sstk",   # sale of common stock
    "prstkc", # purchase of common/preferred stock
    "dvc",    # dividends on common stock
    "dvt",    # total dividends
    "dpc",    # depreciation & amortisation (cash flow)
    "recch",  # receivables decrease/increase (cash flow statement)
    "invch",  # inventory decrease/increase (cash flow statement)
    "apalch", # accounts payable and accrued liabilities change
    "txach",  # taxes payable change
    "aoloch", # other assets/liabilities change
    "fopt",   # funds from operations total
    # Income / balance sheet additions
    "pi",     # pretax income
    "txt",    # income taxes total
    "ebitda", # EBITDA (Compustat item)
    "xad",    # advertising expense
    "re",     # retained earnings
    "wcap",   # working capital
    "mib",    # minority interest (balance sheet)
    "sich",   # historical SIC code (needed for IND_ADJ / IND_ZSCORE)
]

QUARTERLY_VARS = [
    "gvkey", "datadate", "fyearq", "fqtr", "rdq",
    "atq", "ceqq", "ibq", "saleq", "revtq",
    "cogsq", "xsgaq", "actq", "lctq",
    "cheq", "rectq", "invtq", "apq",
    "dlttq", "dlcq", "ppentq", "txtq",
    "niq", "oibdpq", "cshoq", "prccq",
]


def download_compustat_annual(
    start: str = "1960-01-01", end: str = "2024-12-31"
) -> pd.DataFrame:
    """Download Compustat annual fundamentals from WRDS.

    Parameters
    ----------
    start, end : date range for datadate

    Returns
    -------
    DataFrame with annual accounting variables.
    """
    vars_str = ", ".join(ANNUAL_VARS)
    sql = f"""
    SELECT {vars_str}
    FROM comp.funda
    WHERE datadate BETWEEN '{start}' AND '{end}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
    ORDER BY gvkey, datadate
    """
    df = wrds_conn.raw_sql(sql)
    df["datadate"] = pd.to_datetime(df["datadate"])
    # Expose historical SIC under the name expected by SignalEngine
    if "sich" in df.columns:
        df["sic"] = df["sich"]
    return df


def download_compustat_quarterly(
    start: str = "1960-01-01", end: str = "2024-12-31"
) -> pd.DataFrame:
    """Download Compustat quarterly fundamentals from WRDS.

    Parameters
    ----------
    start, end : date range for datadate

    Returns
    -------
    DataFrame with quarterly accounting variables.
    """
    vars_str = ", ".join(QUARTERLY_VARS)
    sql = f"""
    SELECT {vars_str}
    FROM comp.fundq
    WHERE datadate BETWEEN '{start}' AND '{end}'
        AND indfmt = 'INDL'
        AND datafmt = 'STD'
        AND popsrc = 'D'
        AND consol = 'C'
    ORDER BY gvkey, datadate
    """
    df = wrds_conn.raw_sql(sql)
    df["datadate"] = pd.to_datetime(df["datadate"])
    return df


def compute_book_equity(comp: pd.DataFrame) -> pd.DataFrame:
    """Compute book equity following Fama-French convention.

    BE = SEQ (or CEQ + PSTK, or AT - LT)
         + TXDITC (or TXDI + ITCB, or 0)
         - PS (PSTKRV or PSTKL or PSTK, or 0)

    Parameters
    ----------
    comp : Compustat annual DataFrame

    Returns
    -------
    DataFrame with added 'be' column.
    """
    comp = comp.copy()

    # Stockholders' equity
    comp["se"] = comp["seq"].fillna(
        comp["ceq"] + comp["pstk"].fillna(0)
    ).fillna(comp["at"] - comp["lt"])

    # Deferred taxes
    comp["dt"] = comp["txditc"].fillna(
        comp["txdi"].fillna(0) + comp["itcb"].fillna(0)
    )

    # Preferred stock
    comp["ps"] = comp["pstkrv"].fillna(
        comp["pstkl"].fillna(comp["pstk"].fillna(0))
    )

    # Book equity
    comp["be"] = comp["se"] + comp["dt"] - comp["ps"]

    # Set negative book equity to NaN
    comp.loc[comp["be"] <= 0, "be"] = float("nan")

    return comp
