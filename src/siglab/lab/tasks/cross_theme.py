from __future__ import annotations

from dataclasses import dataclass

from siglab.lab.runspec import ModeSpec
from siglab.lab.task import (
    DiscoveryTask,
    EvaluationProtocol,
    NoveltyPolicy,
    OutputSchema,
    SuccessCriteria,
    TaskConstraint,
    TaskObjective,
)


@dataclass(frozen=True)
class CrossThemeTaskSpec:
    theme_a: str
    theme_b: str
    name: str
    interaction_label: str
    economic_story: str
    example_interactions: tuple[str, ...]
    exclude_financials: bool = True


CROSS_THEME_TASKS: tuple[CrossThemeTaskSpec, ...] = (
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="valuation",
        name="profit_value",
        interaction_label="Profitability x Valuation",
        economic_story=(
            "Highly profitable firms that are also cheap combine strong "
            "fundamentals with low market expectations. The task is to identify "
            "signals where profitability strengthens value rather than merely "
            "relabeling either effect."
        ),
        example_interactions=(
            "Gross profit to price",
            "Profitable value composite",
            "Earnings yield adjusted for profitability persistence",
            "Cash-flow-to-price with operating efficiency",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="investment",
        name="profit_invest",
        interaction_label="Profitability x Investment",
        economic_story=(
            "Profitable firms that do not invest aggressively can be underpriced "
            "cash generators, while profitable firms that invest aggressively can "
            "be overpriced because the market credits them for both profitability "
            "and growth. The task is to find signals at the intersection of "
            "profitability and investment discipline."
        ),
        example_interactions=(
            "Profit per dollar invested",
            "Cash cow composite: high profitability with low asset growth",
            "Profitable non-grower",
            "Return on new investment",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="accruals",
        name="profit_accrual",
        interaction_label="Profitability x Accruals",
        economic_story=(
            "Profitable firms are more credible when profits are backed by cash "
            "rather than accrual accounting. The task is to find signals where "
            "accrual quality separates durable profitability from earnings that "
            "are likely to reverse."
        ),
        example_interactions=(
            "Cash-backed profitability",
            "Accrual-adjusted ROA",
            "Profits net of working-capital accrual pressure",
            "Operating profitability with low accrual intensity",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="quality",
        name="profit_quality",
        interaction_label="Profitability x Quality",
        economic_story=(
            "High profitability is more valuable when it is stable, persistent, "
            "and financially healthy. The task is to distinguish durable "
            "operating strength from transient or levered profitability."
        ),
        example_interactions=(
            "Stable high ROA",
            "Cash-flow-supported profitability",
            "Low-volatility operating profitability",
            "Financially healthy profit spread",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="financing",
        name="profit_finance",
        interaction_label="Profitability x Financing",
        economic_story=(
            "Profitable firms that return capital may reveal underpriced cash "
            "generation, while profitable firms issuing capital may reveal "
            "overvaluation or funding needs. The task is to combine operating "
            "strength with financing behavior."
        ),
        example_interactions=(
            "Profitable repurchaser",
            "Self-funded payout quality",
            "Issuance-adjusted profitability",
            "Operating profit net of external financing",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="intangibles",
        name="profit_intangible",
        interaction_label="Profitability x Intangibles",
        economic_story=(
            "Intangible-intensive firms with current profitability may be "
            "underpriced because accounting understates their productive asset "
            "base. The task is to find signals for validated intangible "
            "investment productivity."
        ),
        example_interactions=(
            "Research productivity",
            "Profitable innovator composite",
            "Intangible-adjusted profitability",
            "Research efficiency trend",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="profitability",
        theme_b="distress",
        name="profit_distress",
        interaction_label="Profitability x Distress",
        economic_story=(
            "Profitability may be mispriced differently for firms near "
            "financial distress. The task is to find signals where operating "
            "profitability offsets, confirms, or sharpens distress risk."
        ),
        example_interactions=(
            "Profitable low-distress composite",
            "Operating strength net of leverage",
            "Cash profitability under debt pressure",
            "Turnaround profitability screen",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="investment",
        name="value_invest",
        interaction_label="Valuation x Investment",
        economic_story=(
            "Cheap firms that also avoid aggressive investment can be "
            "underappreciated disciplined capital allocators, while expensive "
            "aggressive investors may be overvalued growth stories. The task "
            "is to combine price expectations with investment discipline."
        ),
        example_interactions=(
            "Cheap disciplined investor",
            "Value with low asset growth",
            "Investment-adjusted value",
            "Expensive empire-builder short signal",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="accruals",
        name="value_accrual",
        interaction_label="Valuation x Accruals",
        economic_story=(
            "Cheap firms can be value traps when reported earnings rely on "
            "accruals rather than cash. The task is to identify value signals "
            "whose cheapness is supported by high-quality accounting."
        ),
        example_interactions=(
            "Cash-backed value",
            "Accrual-adjusted earnings yield",
            "Book-to-market net of accrual pressure",
            "Value trap filter",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="quality",
        name="value_quality",
        interaction_label="Valuation x Quality",
        economic_story=(
            "Cheap firms with high earnings quality are more likely to be "
            "mispriced than cheap firms whose low valuation reflects distress. "
            "The task is to separate quality value from value traps."
        ),
        example_interactions=(
            "Quality value composite",
            "Cash-backed value",
            "Stable earnings yield",
            "Low-leverage value",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="financing",
        name="value_finance",
        interaction_label="Valuation x Financing",
        economic_story=(
            "Cheap firms where management repurchases shares may reveal stronger "
            "undervaluation, while expensive firms issuing equity may reveal "
            "overvaluation. The task is to combine market valuation with "
            "financing behavior."
        ),
        example_interactions=(
            "Conviction value",
            "Buyback yield for value stocks",
            "Management-confirmed cheapness",
            "Issuance-adjusted valuation",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="intangibles",
        name="value_intangible",
        interaction_label="Valuation x Intangibles",
        economic_story=(
            "Accounting can understate book value for intangible-intensive "
            "firms, making standard valuation ratios noisy. The task is to find "
            "value signals that account for intangible investment and "
            "intangible asset productivity."
        ),
        example_interactions=(
            "Intangible-adjusted value",
            "R&D-backed cheapness",
            "Value per intangible investment",
            "Mispriced knowledge capital",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="valuation",
        theme_b="distress",
        name="value_distress",
        interaction_label="Valuation x Distress",
        economic_story=(
            "Cheap stocks can be compensated value opportunities or distressed "
            "lottery-like traps. The task is to separate healthy cheap firms "
            "from cheap firms whose valuation reflects severe distress."
        ),
        example_interactions=(
            "Distress-adjusted value",
            "Healthy cheapness",
            "Low-leverage book-to-market",
            "Current-ratio value filter",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="investment",
        theme_b="accruals",
        name="invest_accrual",
        interaction_label="Investment x Accruals",
        economic_story=(
            "Asset growth driven by accruals can reverse differently from "
            "growth driven by real investment. The task is to distinguish "
            "accounting growth from productive investment."
        ),
        example_interactions=(
            "Real versus accounting growth",
            "Working capital growth",
            "Tangible investment share",
            "Accrual-adjusted investment",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="investment",
        theme_b="quality",
        name="invest_quality",
        interaction_label="Investment x Quality",
        economic_story=(
            "Investment is more informative when paired with quality: "
            "disciplined investment by high-quality firms may differ from asset "
            "growth by fragile firms. The task is to distinguish productive "
            "investment from low-quality expansion."
        ),
        example_interactions=(
            "Quality-adjusted asset growth",
            "Stable earnings with low investment",
            "Productive capex composite",
            "Low-quality growth penalty",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="investment",
        theme_b="financing",
        name="invest_finance",
        interaction_label="Investment x Financing",
        economic_story=(
            "Aggressive investment funded by external financing can reveal "
            "overvaluation, market timing, or empire building. The task is to "
            "separate self-funded investment from externally funded expansion."
        ),
        example_interactions=(
            "Externally funded growth",
            "Empire builder composite",
            "Self-funded versus externally funded investment",
            "Acquisition-driven growth versus organic capex",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="investment",
        theme_b="intangibles",
        name="invest_intangible",
        interaction_label="Investment x Intangibles",
        economic_story=(
            "Tangible and intangible investment can carry different return "
            "implications. The task is to compare physical expansion, R&D, "
            "goodwill, and intangible capital formation within investment "
            "signals."
        ),
        example_interactions=(
            "Intangible share of investment",
            "R&D growth versus asset growth",
            "Organic intangible investment",
            "Capex net of knowledge investment",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="investment",
        theme_b="distress",
        name="invest_distress",
        interaction_label="Investment x Distress",
        economic_story=(
            "Aggressive investment by distressed firms may indicate "
            "value-destroying expansion or urgent balance-sheet strain, while "
            "disciplined investment may signal resilience. The task is to "
            "condition investment signals on financial fragility."
        ),
        example_interactions=(
            "Distressed asset growth penalty",
            "Leverage-adjusted investment",
            "Working-capital-constrained growth",
            "Resilient low-investment composite",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="accruals",
        theme_b="quality",
        name="accrual_quality",
        interaction_label="Accruals x Quality",
        economic_story=(
            "Low accruals combined with persistent, stable earnings should "
            "identify genuinely cash-backed reported performance. The task is "
            "to distinguish durable cash earnings from accounting noise."
        ),
        example_interactions=(
            "Cash-quality composite",
            "Persistent cash earnings",
            "Smoothed accruals",
            "Cash earnings persistence",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="accruals",
        theme_b="financing",
        name="accrual_finance",
        interaction_label="Accruals x Financing",
        economic_story=(
            "External financing combined with aggressive accruals can indicate "
            "overvaluation, earnings management, or balance-sheet pressure. The "
            "task is to combine accounting quality with financing actions."
        ),
        example_interactions=(
            "Accrual-heavy issuer",
            "Financing-adjusted accruals",
            "Cash accruals with repurchase discipline",
            "Working-capital accruals funded externally",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="accruals",
        theme_b="intangibles",
        name="accrual_intangible",
        interaction_label="Accruals x Intangibles",
        economic_story=(
            "Intangible-intensive firms can have accounting noise from both "
            "expense treatment and accrual components. The task is to separate "
            "productive intangible investment from accounting-driven earnings."
        ),
        example_interactions=(
            "R&D with low accrual pressure",
            "Intangible-adjusted cash earnings",
            "Accrual-clean innovation signal",
            "Goodwill growth net of accruals",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="accruals",
        theme_b="distress",
        name="accrual_distress",
        interaction_label="Accruals x Distress",
        economic_story=(
            "Accrual problems can be more severe for financially fragile "
            "firms, where earnings management and liquidity pressure are more "
            "likely. The task is to combine accrual quality with distress risk."
        ),
        example_interactions=(
            "Distress-conditioned accruals",
            "Cash earnings under leverage pressure",
            "Low-accrual low-distress composite",
            "Working-capital quality for fragile firms",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="quality",
        theme_b="financing",
        name="quality_finance",
        interaction_label="Quality x Financing",
        economic_story=(
            "Financing actions are more informative when paired with firm "
            "quality. Repurchases by high-quality firms may confirm "
            "undervaluation, while issuance by low-quality firms may confirm "
            "overvaluation or funding stress."
        ),
        example_interactions=(
            "Quality repurchaser",
            "Low-quality issuer penalty",
            "Payout quality composite",
            "Stable cash-flow financing signal",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="quality",
        theme_b="intangibles",
        name="quality_intangible",
        interaction_label="Quality x Intangibles",
        economic_story=(
            "Intangible investment is more credible when paired with stable "
            "cash flows, earnings persistence, and balance-sheet quality. The "
            "task is to identify durable intangible productivity."
        ),
        example_interactions=(
            "Stable R&D productivity",
            "Quality-adjusted intangible capital",
            "Cash-flow-backed innovation",
            "Low-volatility knowledge investment",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="quality",
        theme_b="distress",
        name="quality_distress",
        interaction_label="Quality x Distress",
        economic_story=(
            "Quality and distress are opposing but not identical dimensions. "
            "The task is to find signals where stability, cash generation, and "
            "balance-sheet strength sharpen distress-related return predictions."
        ),
        example_interactions=(
            "High-quality low-distress composite",
            "Stable earnings under leverage",
            "Cash-flow solvency quality",
            "Low-volatility distress filter",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="financing",
        theme_b="intangibles",
        name="finance_intangible",
        interaction_label="Financing x Intangibles",
        economic_story=(
            "External financing for intangible investment can indicate valuable "
            "innovation funding or market-timed issuance. The task is to "
            "distinguish productive intangible financing from overvaluation-driven "
            "issuance."
        ),
        example_interactions=(
            "Externally funded R&D growth",
            "Repurchase-backed intangible value",
            "Issuance-adjusted knowledge capital",
            "Organic versus financed intangible investment",
        ),
    ),
    CrossThemeTaskSpec(
        theme_a="financing",
        theme_b="distress",
        name="finance_distress",
        interaction_label="Financing x Distress",
        economic_story=(
            "Financing behavior has different meaning for distressed firms. "
            "Debt issuance, repayment, payouts, and equity issuance may reveal "
            "liquidity pressure, market timing, or recovery prospects."
        ),
        example_interactions=(
            "Distressed issuer penalty",
            "Debt repayment under distress",
            "Payout resilience under leverage",
            "Equity issuance with weak solvency",
        ),
        exclude_financials=False,
    ),
    CrossThemeTaskSpec(
        theme_a="intangibles",
        theme_b="distress",
        name="intangible_distress",
        interaction_label="Intangibles x Distress",
        economic_story=(
            "Intangible-intensive firms can look distressed under traditional "
            "accounting even when they own productive knowledge capital. The "
            "task is to separate accounting distress from genuine financial "
            "fragility among intangible-heavy firms."
        ),
        example_interactions=(
            "Intangible-adjusted distress",
            "R&D resilience under leverage",
            "Knowledge capital solvency filter",
            "Goodwill-heavy distress penalty",
        ),
    ),
)

_TASK_BY_NAME = {task.name: task for task in CROSS_THEME_TASKS}


def get_cross_theme_task_spec(name: str) -> CrossThemeTaskSpec:
    try:
        return _TASK_BY_NAME[name]
    except KeyError as exc:
        known = ", ".join(sorted(_TASK_BY_NAME))
        raise KeyError(f"unknown cross-theme task {name!r}; known: {known}") from exc


def list_cross_theme_task_names() -> list[str]:
    return [task.name for task in CROSS_THEME_TASKS]


def build_cross_theme_task(
    spec: CrossThemeTaskSpec | str,
    mode: ModeSpec | None = None,
    sample_filters: tuple[str, ...] | list[str] | None = None,
    objective_override: str | None = None,
    alpha_factor_model: int = 5,
) -> DiscoveryTask:
    active_spec = get_cross_theme_task_spec(spec) if isinstance(spec, str) else spec
    active_mode = mode or ModeSpec()
    if sample_filters is None:
        active_filters = (
            ("exclude_financials",) if active_spec.exclude_financials else ()
        ) + ("exclude_microcap",)
    else:
        active_filters = tuple(sample_filters)
    exclude_financials = "exclude_financials" in active_filters
    exclude_microcap = "exclude_microcap" in active_filters
    description = (
        "Discover symbolic equity-return signals that use both themes "
        f"({active_spec.theme_a} and {active_spec.theme_b}) for "
        f"{active_spec.interaction_label}. The signal should capture an "
        "interaction beyond an isolated single-theme effect. Economic story: "
        f"{active_spec.economic_story}"
    )

    if objective_override and objective_override.strip():
        # RunSpec task.objective_override REPLACES the default objective
        # description (mode-level seed_prompt guidance still appends below).
        description = objective_override.strip()

    if active_mode.seed_prompt and active_mode.guidance_scope in {"task", "task_and_loop"}:
        description += f" Human guidance: {active_mode.seed_prompt}"

    return DiscoveryTask(
        task_id=f"cross_theme:{active_spec.name}",
        task_type="cross_theme",
        objective=TaskObjective(
            title=f"{active_spec.interaction_label} signal discovery",
            description=description,
        ),
        constraints=[
            TaskConstraint(kind="must_use_theme", value=active_spec.theme_a),
            TaskConstraint(kind="must_use_theme", value=active_spec.theme_b),
            TaskConstraint(
                kind="exclude_financials",
                value=str(exclude_financials).lower(),
            ),
            TaskConstraint(
                kind="exclude_microcap",
                value=str(exclude_microcap).lower(),
            ),
        ],
        output_schema=OutputSchema(
            required_fields=[
                "name",
                "expression",
                "hypothesis",
                "expected_sign",
                "reasoning",
                "interaction_type",
            ],
            allowed_expected_signs=["positive", "negative"],
            allowed_interaction_types=["composite", "conditional", "ratio", "novel"],
        ),
        evaluation_protocol=EvaluationProtocol(
            evaluator_type="signal_agent",
            primary_metric="fmb_tstat",
            success_threshold=1.96,
            require_expected_sign_match=True,
            alpha_factor_model=alpha_factor_model,
        ),
        success_criteria=SuccessCriteria(
            min_abs_tstat=1.96,
            require_expected_sign_match=True,
            min_coverage=0.0,
        ),
        novelty_policy=NoveltyPolicy(
            reject_exact_expression_duplicates=True,
            reject_normalized_expression_duplicates=True,
        ),
        metadata={
            "pair_name": active_spec.name,
            "theme_a": active_spec.theme_a,
            "theme_b": active_spec.theme_b,
            "reference_library": "canonical_single_theme_reference",
            "interaction_label": active_spec.interaction_label,
            "economic_story": active_spec.economic_story,
            "example_interactions": list(active_spec.example_interactions),
            "exclude_financials": exclude_financials,
            "exclude_microcap": exclude_microcap,
            "sample_filters": list(active_filters),
            "mode": active_mode.type,
            "seed_prompt": active_mode.seed_prompt,
            "guidance_scope": active_mode.guidance_scope,
        },
    )
