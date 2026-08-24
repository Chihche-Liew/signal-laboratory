"""Variable catalog for fundamental signal research.

Structured knowledge base mapping Compustat mnemonics to metadata.
The LLM agent uses this to understand what data is available and how to use it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variable:
    """Metadata for a single accounting variable."""
    mnemonic: str       # Compustat field name (e.g., "revt")
    name: str           # Human-readable name
    statement: str      # BS (balance sheet), IS (income statement), CF (cash flow)
    category: str       # asset, liability, equity, revenue, expense, cash_flow, other
    sign: str           # "positive" = higher is typically better, "negative" = higher is worse, "neutral"
    description: str    # One-line description for the LLM
    common_scalers: tuple[str, ...] = ()  # Typical denominators when used as numerator


# ── Balance Sheet Variables ──────────────────────────────────────────────

_BS_VARS = [
    Variable("at", "Total Assets", "BS", "asset", "neutral",
             "Total assets; the most common denominator for scaling",
             ("at",)),
    Variable("ceq", "Common Equity", "BS", "equity", "positive",
             "Book value of common equity; denominator for ROE and equity-scaled signals",
             ("at", "me")),
    Variable("seq", "Stockholders' Equity", "BS", "equity", "positive",
             "Total stockholders' equity including preferred; used in book equity computation",
             ("at",)),
    Variable("lt", "Total Liabilities", "BS", "liability", "negative",
             "Total liabilities; leverage = lt/at",
             ("at", "ceq")),
    Variable("act", "Current Assets", "BS", "asset", "neutral",
             "Current assets; used in working capital and accruals computations",
             ("at", "lct")),
    Variable("lct", "Current Liabilities", "BS", "liability", "neutral",
             "Current liabilities; current ratio = act/lct",
             ("at", "act")),
    Variable("che", "Cash and Short-Term Investments", "BS", "asset", "neutral",
             "Cash and equivalents; high cash can signal precautionary savings or low investment",
             ("at", "lct")),
    Variable("invt", "Inventories", "BS", "asset", "neutral",
             "Total inventories; inventory growth predicts low returns",
             ("at", "sale")),
    Variable("rect", "Receivables Total", "BS", "asset", "neutral",
             "Accounts receivable; used in accruals decomposition and turnover ratios",
             ("at", "sale")),
    Variable("ppent", "Net Property, Plant & Equipment", "BS", "asset", "neutral",
             "Net PP&E; tangibility = ppent/at, fixed asset turnover = sale/ppent",
             ("at",)),
    Variable("ppegt", "Gross Property, Plant & Equipment", "BS", "asset", "neutral",
             "Gross PP&E before depreciation; capital intensity measure",
             ("at",)),
    Variable("intan", "Intangible Assets", "BS", "asset", "neutral",
             "Reported intangible assets; intangibility = intan/at",
             ("at",)),
    Variable("gdwl", "Goodwill", "BS", "asset", "neutral",
             "Goodwill from acquisitions; high goodwill may indicate overpriced acquisitions",
             ("at",)),
    Variable("dltt", "Long-Term Debt", "BS", "liability", "negative",
             "Long-term debt; key leverage component",
             ("at", "ceq", "me")),
    Variable("dlc", "Debt in Current Liabilities", "BS", "liability", "negative",
             "Short-term debt; total debt = dltt + dlc",
             ("at",)),
    Variable("ap", "Accounts Payable", "BS", "liability", "neutral",
             "Accounts payable; trade credit, used in accruals",
             ("at",)),
    Variable("ao", "Assets Other", "BS", "asset", "neutral",
             "Other assets; captures non-standard items",
             ("at",)),
    Variable("lo", "Liabilities Other", "BS", "liability", "neutral",
             "Other liabilities; captures off-balance-sheet type items",
             ("at",)),
    Variable("aco", "Current Assets Other", "BS", "asset", "neutral",
             "Other current assets",
             ("at",)),
    Variable("xacc", "Accrued Expenses", "BS", "liability", "neutral",
             "Accrued expenses; accrued liabilities",
             ("at",)),
    Variable("re", "Retained Earnings", "BS", "equity", "positive",
             "Cumulative retained earnings; Altman Z-score component, earned/contributed capital",
             ("at",)),
    Variable("mib", "Minority Interest", "BS", "equity", "neutral",
             "Minority interest on balance sheet",
             ("at",)),
    # Preferred stock variants
    Variable("pstkrv", "Preferred Stock - Redemption", "BS", "equity", "neutral",
             "Preferred stock redemption value; used in book equity computation",
             ()),
    Variable("pstkl", "Preferred Stock - Liquidating", "BS", "equity", "neutral",
             "Preferred stock liquidating value; fallback for pstkrv",
             ()),
    Variable("pstk", "Preferred Stock - Carrying", "BS", "equity", "neutral",
             "Preferred stock carrying value; last fallback",
             ()),
    # Tax items on BS
    Variable("txditc", "Deferred Taxes and ITC", "BS", "asset", "neutral",
             "Deferred taxes and investment tax credit; book equity component",
             ()),
    Variable("txdi", "Deferred Taxes", "BS", "asset", "neutral",
             "Deferred income taxes; fallback component of txditc",
             ()),
    Variable("itcb", "Investment Tax Credit", "BS", "asset", "neutral",
             "Investment tax credit balance sheet; fallback component of txditc",
             ()),
    Variable("txp", "Income Taxes Payable", "BS", "liability", "neutral",
             "Current income taxes payable; used in balance-sheet accruals",
             ()),
]

# ── Income Statement Variables ───────────────────────────────────────────

_IS_VARS = [
    Variable("revt", "Revenue Total", "IS", "revenue", "positive",
             "Total revenue; top-line growth and profitability numerator",
             ("at", "me", "ppent")),
    Variable("sale", "Net Sales", "IS", "revenue", "positive",
             "Net sales/turnover; slightly different from revt for some firms",
             ("at", "me", "emp")),
    Variable("cogs", "Cost of Goods Sold", "IS", "expense", "negative",
             "Direct production costs; gross profit = revt - cogs",
             ("sale", "at")),
    Variable("xsga", "SG&A Expense", "IS", "expense", "negative",
             "Selling, general & admin; operating profit = GP - xsga",
             ("sale", "at")),
    Variable("xrd", "R&D Expense", "IS", "expense", "positive",
             "R&D spending; R&D intensity is positively priced (intangibles anomaly)",
             ("sale", "at", "me")),
    Variable("xint", "Interest Expense", "IS", "expense", "negative",
             "Interest on debt; used in operating profitability (FF 2015)",
             ("at", "dltt")),
    Variable("xad", "Advertising Expense", "IS", "expense", "positive",
             "Advertising spending; brand capital anomaly, often missing",
             ("sale", "at", "me")),
    Variable("dp", "Depreciation and Amortization", "IS", "expense", "neutral",
             "D&A from income statement; used in accruals computation",
             ("at", "ppent")),
    Variable("ib", "Income Before Extraordinary", "IS", "revenue", "positive",
             "Clean earnings measure; used in ROA, accruals, earnings quality",
             ("at", "ceq", "sale", "me")),
    Variable("ni", "Net Income", "IS", "revenue", "positive",
             "Bottom-line earnings; used in ROE, payout ratios",
             ("at", "ceq", "sale", "me")),
    Variable("pi", "Pre-Tax Income", "IS", "revenue", "positive",
             "Pre-tax income; effective tax rate = txt/pi",
             ("at", "sale")),
    Variable("txt", "Total Income Taxes", "IS", "expense", "neutral",
             "Income tax expense; tax burden signals",
             ("pi", "at")),
    Variable("oibdp", "Operating Income Before D&A", "IS", "revenue", "positive",
             "EBITDA proxy; used in enterprise multiples",
             ("at", "sale", "me")),
    Variable("oiadp", "Operating Income After D&A", "IS", "revenue", "positive",
             "EBIT proxy; operating earnings after depreciation",
             ("at", "sale", "me")),
    Variable("ebitda", "EBITDA", "IS", "revenue", "positive",
             "Earnings before interest, taxes, D&A; enterprise multiple = (ME+DLTT+DLC-CHE)/EBITDA",
             ("at", "sale", "me")),
    Variable("spi", "Special Items", "IS", "expense", "neutral",
             "Non-recurring items; earnings quality indicator",
             ("at", "sale")),
]

# ── Cash Flow Statement Variables ────────────────────────────────────────

_CF_VARS = [
    Variable("oancf", "Operating Cash Flow", "CF", "cash_flow", "positive",
             "Net cash from operations; cash ROA = oancf/at, accruals = ib - oancf",
             ("at", "ceq", "me", "sale")),
    Variable("capx", "Capital Expenditures", "CF", "cash_flow", "neutral",
             "Capital spending; investment intensity, free cash flow = oancf - capx",
             ("at", "ppent", "sale")),
    Variable("dpc", "Depreciation (CF Statement)", "CF", "cash_flow", "neutral",
             "D&A from cash flow statement; may differ from income statement dp",
             ("at",)),
    Variable("recch", "Change in Receivables", "CF", "cash_flow", "neutral",
             "Decrease (increase) in receivables; accruals decomposition component",
             ("at",)),
    Variable("invch", "Change in Inventories", "CF", "cash_flow", "neutral",
             "Decrease (increase) in inventories; accruals decomposition component",
             ("at",)),
    Variable("apalch", "Change in Accounts Payable", "CF", "cash_flow", "neutral",
             "Increase (decrease) in AP; accruals decomposition component",
             ("at",)),
    Variable("txach", "Change in Taxes Payable", "CF", "cash_flow", "neutral",
             "Increase (decrease) in taxes payable; accruals decomposition",
             ("at",)),
    Variable("aoloch", "Change in Other Assets/Liabilities", "CF", "cash_flow", "neutral",
             "Net change in other operating items; accruals decomposition",
             ("at",)),
    Variable("wcap", "Working Capital", "CF", "cash_flow", "neutral",
             "Working capital from balance sheet; alternative accruals base",
             ("at",)),
    Variable("fopt", "Funds from Operations", "CF", "cash_flow", "positive",
             "Funds from operations; fallback for oancf in pre-1988 data",
             ("at",)),
    Variable("prstkc", "Purchase of Stock", "CF", "cash_flow", "positive",
             "Share repurchases; net payout = dvc + prstkc - sstk",
             ("me", "at")),
    Variable("sstk", "Sale of Stock", "CF", "cash_flow", "negative",
             "Equity issuance; net issuance predicts low returns",
             ("me", "at")),
    Variable("dltis", "LT Debt Issuance", "CF", "cash_flow", "negative",
             "Long-term debt issuance; net debt issuance anomaly",
             ("at",)),
    Variable("dltr", "LT Debt Retirement", "CF", "cash_flow", "positive",
             "Long-term debt retirement; net debt change = dltis - dltr",
             ("at",)),
    Variable("dvc", "Common Dividends", "CF", "cash_flow", "positive",
             "Cash dividends on common stock; dividend yield, payout ratio",
             ("ni", "me", "ib")),
    Variable("dvt", "Total Dividends", "CF", "cash_flow", "positive",
             "Total dividends including preferred; total payout measure",
             ("ni", "me")),
    Variable("aqc", "Acquisitions", "CF", "cash_flow", "neutral",
             "Cash spent on acquisitions; M&A-driven vs organic growth",
             ("at", "me")),
]

# ── Other Variables ──────────────────────────────────────────────────────

_OTHER_VARS = [
    Variable("emp", "Employees", "BS", "other", "neutral",
             "Number of employees (thousands); hiring rate, productivity",
             ("at", "sale")),
    Variable("csho", "Common Shares Outstanding", "BS", "other", "neutral",
             "Shares outstanding; per-share computations",
             ()),
    Variable("ajex", "Adjustment Factor", "BS", "other", "neutral",
             "Cumulative adjustment factor for stock splits",
             ()),
    Variable("prcc_f", "Price Close (Fiscal Year End)", "BS", "other", "neutral",
             "Stock price at fiscal year end; Compustat-based market cap = prcc_f * csho",
             ()),
]


# ── Complete Catalog ─────────────────────────────────────────────────────

VARIABLE_CATALOG: dict[str, Variable] = {}
for _var_list in [_BS_VARS, _IS_VARS, _CF_VARS, _OTHER_VARS]:
    for v in _var_list:
        VARIABLE_CATALOG[v.mnemonic] = v


def get_variables_by_category(category: str) -> list[Variable]:
    """Get all variables in a given category."""
    return [v for v in VARIABLE_CATALOG.values() if v.category == category]


def get_variables_by_statement(statement: str) -> list[Variable]:
    """Get all variables from a given financial statement (BS, IS, CF)."""
    return [v for v in VARIABLE_CATALOG.values() if v.statement == statement]


def catalog_summary() -> str:
    """Return a human-readable summary of the variable catalog."""
    lines = ["FUNDAMENTAL VARIABLE CATALOG", "=" * 60]
    for stmt in ["BS", "IS", "CF"]:
        vars_in = get_variables_by_statement(stmt)
        stmt_name = {"BS": "Balance Sheet", "IS": "Income Statement", "CF": "Cash Flow"}[stmt]
        lines.append(f"\n{stmt_name} ({len(vars_in)} variables):")
        for v in vars_in:
            lines.append(f"  {v.mnemonic:10s} {v.name:40s} [{v.sign}]")
    return "\n".join(lines)
