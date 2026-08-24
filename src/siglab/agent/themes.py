"""Economic theme definitions for directed signal search.

Each theme defines:
- A curated subset of relevant variables
- Exemplar signals the agent can study
- Preferred operators for the theme
- Economic intuition the agent should reason about
"""

from dataclasses import dataclass, field


@dataclass
class ExemplarSignal:
    """A known signal the agent can study as a template."""
    name: str
    formula: str          # Human-readable formula
    reference: str        # Paper citation
    expected_sign: str    # "+" or "-" (higher signal → higher/lower expected return)
    typical_tstat: str    # Approximate t-stat range from literature


@dataclass
class Theme:
    """An economic theme directing the agent's signal search."""
    name: str
    description: str
    intuition: str        # Why signals in this theme predict returns
    variables: list[str]  # Compustat mnemonics relevant to this theme
    preferred_ops: list[str]  # Operator names most useful for this theme
    exemplars: list[ExemplarSignal] = field(default_factory=list)
    exclude_financials: bool = True  # Whether to exclude SIC 6000-6999


THEMES: dict[str, Theme] = {}

# ── Theme 1: Profitability ───────────────────────────────────────────────

THEMES["profitability"] = Theme(
    name="Profitability",
    description="Signals measuring a firm's ability to generate earnings from its assets or equity.",
    intuition=(
        "More profitable firms have higher expected returns (Novy-Marx 2013). "
        "This contradicts the risk-based explanation (profitable firms are less risky) "
        "and may reflect mispricing of growth expectations or earnings persistence."
    ),
    variables=["revt", "cogs", "xsga", "xint", "xrd", "ib", "ni", "oibdp",
               "oiadp", "oancf", "ebitda", "pi", "at", "ceq", "sale", "dp"],
    preferred_ops=["RATIO", "IND_ADJ", "RANK", "MA", "TREND"],
    exemplars=[
        ExemplarSignal("GP/AT", "(revt - cogs) / at", "Novy-Marx (2013)", "+", "t ≈ 4-5"),
        ExemplarSignal("ROA", "ib / at", "Haugen-Baker (1996)", "+", "t ≈ 2-3"),
        ExemplarSignal("CashROA", "oancf / at", "Ball et al. (2016)", "+", "t ≈ 4-5"),
        ExemplarSignal("OperProf", "(revt - cogs - xsga - xint) / ceq", "Fama-French (2015)", "+", "t ≈ 3-4"),
    ],
)

# ── Theme 2: Valuation ──────────────────────────────────────────────────

THEMES["valuation"] = Theme(
    name="Valuation",
    description="Signals measuring how cheap a stock is relative to its fundamental value.",
    intuition=(
        "Cheap stocks outperform expensive stocks (Fama-French 1993). "
        "This reflects either higher risk (distress, leverage) or behavioral overreaction "
        "to past poor performance. The value premium is one of the oldest anomalies."
    ),
    variables=["ceq", "ib", "ni", "sale", "revt", "oancf", "ebitda", "dvc",
               "at", "dltt", "dlc", "che", "prcc_f", "csho"],
    preferred_ops=["RATIO", "INV", "RANK", "LOG", "MA"],
    exemplars=[
        ExemplarSignal("BM", "ceq / ME", "Fama-French (1993)", "+", "t ≈ 3-4"),
        ExemplarSignal("EP", "ib / ME", "Basu (1977)", "+", "t ≈ 3-4"),
        ExemplarSignal("CFP", "oancf / ME", "Lakonishok-Shleifer-Vishny (1994)", "+", "t ≈ 3-4"),
        ExemplarSignal("SP", "sale / ME", "Barbee-Mukherji-Raines (1996)", "+", "t ≈ 2-3"),
    ],
    exclude_financials=False,
)

# ── Theme 3: Investment / Asset Growth ───────────────────────────────────

THEMES["investment"] = Theme(
    name="Investment",
    description="Signals measuring firm investment and asset expansion activity.",
    intuition=(
        "Firms that invest aggressively (high asset growth, capex) have lower expected returns "
        "(Cooper-Gulen-Schill 2008). This may reflect managers investing when discount rates "
        "are low (q-theory) or empire-building that destroys value."
    ),
    variables=["at", "capx", "ppent", "ppegt", "invt", "emp", "aqc",
               "sstk", "dltt", "dlc", "che", "intan", "gdwl"],
    preferred_ops=["GROWTH", "DELTA", "RATIO", "NEG", "RANK"],
    exemplars=[
        ExemplarSignal("AssetGrowth", "-(at_t - at_{t-1}) / at_{t-1}", "Cooper-Gulen-Schill (2008)", "+", "t ≈ 5-7"),
        ExemplarSignal("InvGrowth", "-(capx_t - capx_{t-1}) / capx_{t-1}", "Xing (2008)", "+", "t ≈ 2-3"),
        ExemplarSignal("HiringRate", "-(emp_t - emp_{t-1}) / emp_{t-1}", "Belo-Lin-Bazdresch (2014)", "+", "t ≈ 2-3"),
    ],
)

# ── Theme 4: Accruals ───────────────────────────────────────────────────

THEMES["accruals"] = Theme(
    name="Accruals",
    description="Signals measuring the divergence between accrual earnings and cash earnings.",
    intuition=(
        "High accruals predict low returns (Sloan 1996). Investors overweight reported earnings "
        "without distinguishing the cash vs. accrual components. The cash component of earnings "
        "is more persistent, so high-accrual firms experience negative earnings surprises."
    ),
    variables=["ib", "oancf", "at", "act", "lct", "che", "dlc", "txp", "dp",
               "recch", "invch", "apalch", "txach", "aoloch", "dpc", "wcap"],
    preferred_ops=["DELTA", "SUB", "RATIO", "NEG", "RANK"],
    exemplars=[
        ExemplarSignal("BS_Accruals", "-((Δact-Δche) - (Δlct-Δdlc-Δtxp) - dp) / avg(at)", "Sloan (1996)", "+", "t ≈ 3-5"),
        ExemplarSignal("CF_Accruals", "-(ib - oancf) / at", "Hribar-Collins (2002)", "+", "t ≈ 4-5"),
        ExemplarSignal("PctAccruals", "-(ib - oancf) / |ib|", "Hafzalla-Lundholm-Van Winkle (2011)", "+", "t ≈ 2-3"),
    ],
)

# ── Theme 5: Quality ────────────────────────────────────────────────────

THEMES["quality"] = Theme(
    name="Quality",
    description="Composite signals measuring earnings quality, stability, and financial health.",
    intuition=(
        "High-quality firms (stable earnings, low leverage, positive cash flows) outperform "
        "low-quality firms. Quality may proxy for a low-risk anomaly or behavioral mispricing "
        "of boring, stable companies."
    ),
    variables=["ib", "oancf", "at", "ceq", "ni", "sale", "revt", "cogs",
               "dltt", "dlc", "lt", "act", "lct", "dp", "re"],
    preferred_ops=["VOL", "TREND", "MA", "INDICATOR", "SUMRANK", "RATIO"],
    exemplars=[
        ExemplarSignal("EarningsVol", "-VOL(ib/at, k=5)", "Various", "+", "t ≈ 2-3"),
        ExemplarSignal("CashEarnings", "oancf / |ib|", "Sloan (1996)", "+", "t ≈ 2-3"),
        ExemplarSignal("EarningsPersist", "AR(1) coef of ib/at over 10 years", "Francis et al. (2004)", "+", "t ≈ 2-3"),
    ],
)

# ── Theme 6: Financing / Issuance ────────────────────────────────────────

THEMES["financing"] = Theme(
    name="Financing",
    description="Signals measuring net equity/debt issuance and payout activity.",
    intuition=(
        "Firms that issue equity tend to do so when overvalued (Baker-Wurgler 2002). "
        "Net issuers underperform, net repurchasers outperform. This is one of the most "
        "robust anomalies, potentially reflecting management information advantage."
    ),
    variables=["prstkc", "sstk", "dltis", "dltr", "dvc", "dvt",
               "csho", "dltt", "dlc", "at", "ceq", "lt"],
    preferred_ops=["DELTA", "GROWTH", "RATIO", "SUB", "NEG", "RANK"],
    exemplars=[
        ExemplarSignal("NetEquityIss", "-(log(csho_t * ajex_t) - log(csho_{t-1} * ajex_{t-1}))", "Daniel-Titman (2006)", "+", "t ≈ 3-5"),
        ExemplarSignal("NetPayoutYield", "(dvc + prstkc - sstk) / ME", "Boudoukh et al. (2007)", "+", "t ≈ 2-3"),
        ExemplarSignal("NetDebtIss", "-(dltis - dltr) / at", "Bradshaw-Richardson-Sloan (2006)", "+", "t ≈ 2-3"),
    ],
    exclude_financials=False,
)

# ── Theme 7: Intangibles / R&D ──────────────────────────────────────────

THEMES["intangibles"] = Theme(
    name="Intangibles",
    description="Signals related to R&D, advertising, and intangible capital.",
    intuition=(
        "R&D-intensive firms are undervalued because accounting treats R&D as an expense "
        "rather than an investment (Chan-Lakonishok-Sougiannis 2001). Firms building "
        "intangible capital have higher future earnings not reflected in current book value."
    ),
    variables=["xrd", "xad", "intan", "gdwl", "sale", "at", "ceq",
               "revt", "cogs"],
    preferred_ops=["RATIO", "GROWTH", "MA", "RANK"],
    exemplars=[
        ExemplarSignal("RD_ME", "xrd / ME", "Chan-Lakonishok-Sougiannis (2001)", "+", "t ≈ 2-3"),
        ExemplarSignal("RD_Sale", "xrd / sale", "Lev-Sougiannis (1996)", "+", "t ≈ 2-3"),
        ExemplarSignal("AdGrowth", "-GROWTH(xad, 1)", "Lou (2014)", "+", "t ≈ 2-3"),
    ],
)

# ── Theme 8: Distress / Leverage ─────────────────────────────────────────

THEMES["distress"] = Theme(
    name="Distress",
    description="Signals measuring financial distress risk and leverage.",
    intuition=(
        "The relationship between distress and returns is nuanced. Campbell-Hilscher-Szilagyi (2008) "
        "show that distressed stocks have LOW returns — the 'distress anomaly'. "
        "This may reflect limits to arbitrage (distressed stocks are hard to short) "
        "or investors' lottery preferences for low-priced distressed stocks."
    ),
    variables=["at", "lt", "ceq", "dltt", "dlc", "act", "lct", "ib", "ni",
               "sale", "re", "pi", "txt", "che", "oancf", "wcap"],
    preferred_ops=["RATIO", "ADD", "SUB", "INDICATOR", "SUMRANK", "RANK"],
    exemplars=[
        ExemplarSignal("Leverage", "-(dltt + dlc) / at", "Bhandari (1988)", "+", "t ≈ 1-2"),
        ExemplarSignal("Zscore", "1.2*wcap/at + 1.4*re/at + 3.3*pi/at + 0.6*ME/lt + sale/at",
                       "Altman (1968)", "+", "t ≈ 2-3"),
        ExemplarSignal("CurrentRatio", "act / lct", "Various", "+", "t ≈ 1-2"),
    ],
)


def get_theme(name: str) -> Theme:
    """Get a theme by name (case-insensitive)."""
    key = name.lower()
    if key not in THEMES:
        raise ValueError(f"Unknown theme '{name}'. Available: {list(THEMES.keys())}")
    return THEMES[key]


def theme_summary() -> str:
    """Return a human-readable summary of all themes."""
    lines = ["ECONOMIC THEMES FOR SIGNAL DISCOVERY", "=" * 60]
    for key, theme in THEMES.items():
        lines.append(f"\n{theme.name} ({len(theme.variables)} variables, {len(theme.exemplars)} exemplars)")
        lines.append(f"  {theme.description}")
        lines.append(f"  Variables: {', '.join(theme.variables[:8])}...")
        lines.append(f"  Operators: {', '.join(theme.preferred_ops)}")
        if theme.exemplars:
            lines.append("  Exemplars:")
            for ex in theme.exemplars:
                lines.append(f"    {ex.name}: {ex.formula} ({ex.reference}, {ex.expected_sign})")
    return "\n".join(lines)
