from __future__ import annotations

from dataclasses import dataclass

from siglab.agent.operators import OPERATOR_CATALOG
from siglab.agent.themes import get_theme
from siglab.lab.archive import EvaluatedSignal
from siglab.lab.prompting.context import PromptContext, PromptRenderResult
from siglab.lab.prompting.metric_guidance import (
    format_evaluation_search_objective,
    format_evaluation_stat_definitions,
)
from siglab.lab.prompting.operator_guidance import format_operator_arity_guard


def _guidance_block(guidance: str) -> str:
    return "\n".join([
        "",
        "=" * 60,
        "HUMAN RESEARCH GUIDANCE",
        "=" * 60,
        guidance,
        "",
        "Guardrail: Use this guidance to focus research direction only. Do not "
        "change the allowed variables, allowed operators, evaluation criteria, "
        "or historical results.",
    ])


def _intent_block(intent: str) -> str:
    instructions = {
        "repair": "Repair prior failures before exploring new variants.",
        "explore": "Explore distinct mechanisms and avoid early convergence.",
        "exploit": "Exploit promising mechanisms with focused refinements.",
        "reflect": "Reflect on prior evidence and revise the search direction.",
    }
    text = instructions.get(intent, instructions["explore"])
    return "\n".join([
        "",
        "=" * 60,
        "MODE INTENT",
        "=" * 60,
        text,
    ])


def _format_reference_section(context: PromptContext) -> str:
    metadata = context.task.metadata
    examples = metadata.get("example_interactions") or []
    lines = [
        "Reference single-theme signals",
        "Use these only as conceptual anchors. Proposals must combine both themes.",
    ]
    for example in examples:
        lines.append(f"- {example}")
    return "\n".join(lines)


def _format_output_schema(context: PromptContext) -> str:
    schema = context.task.output_schema
    placeholders = {
        "name": "<short unique name>",
        "expression": "<valid symbolic expression>",
        "hypothesis": "<economic hypothesis>",
        "expected_sign": "<positive or negative>",
        "interaction_type": "<composite | conditional | ratio | novel>",
        "reasoning": "<2-3 sentences>",
    }
    fields = "\n".join(
        f"{field.upper()}: {placeholders.get(field, '<value>')}"
        for field in schema.required_fields
    )
    return "\n".join([
        "Output format",
        "Return each proposal as:",
        "---SIGNAL 1---",
        fields,
        "",
        "Formatting constraints:",
        "- The schema field labels must start at the beginning of the line exactly as shown.",
        "- Do not prefix field labels with bullets, numbering, Markdown, or extra punctuation.",
        "- Never write '- NAME:', '- EXPRESSION:', or any other bulleted field label.",
        "",
        "Allowed EXPECTED_SIGN values: " + ", ".join(schema.allowed_expected_signs),
        "Allowed INTERACTION_TYPE values: " + ", ".join(schema.allowed_interaction_types),
    ])


def _format_archive_section(evaluated: list[EvaluatedSignal]) -> str:
    if not evaluated:
        return "Prior evaluated signals\n- None yet."

    lines = ["Prior evaluated signals"]
    for signal in evaluated:
        metric_bits: list[str] = []
        if signal.fmb_tstat is not None:
            metric_bits.append(f"fmb_tstat={signal.fmb_tstat:.3f}")
        if signal.ls_alpha is not None:
            metric_bits.append(f"ls_alpha={signal.ls_alpha:.3f}")
        if signal.ls_talpha is not None:
            metric_bits.append(f"ls_talpha={signal.ls_talpha:.3f}")
        if signal.ls_sharpe is not None:
            metric_bits.append(f"ls_sharpe={signal.ls_sharpe:.3f}")
        if signal.coverage is not None:
            metric_bits.append(f"coverage={signal.coverage:.3f}")
        if signal.error:
            metric_bits.append(f"error={signal.error}")
        metrics = "; ".join(metric_bits) if metric_bits else "metrics=missing"
        lines.append(
            f"- gen {signal.generation}: {signal.name}; {signal.expression}; "
            f"expected={signal.expected_sign}; {metrics}"
        )
    return "\n".join(lines)


def _format_theme_resource_section(context: PromptContext) -> str:
    theme_a = str(context.task.metadata["theme_a"])
    theme_b = str(context.task.metadata["theme_b"])
    a = get_theme(theme_a)
    b = get_theme(theme_b)
    return "\n".join([
        "THEME RESOURCE MENU",
        (
            f"{theme_a} variables: "
            + ", ".join(a.variables)
            + f"; preferred operators: {', '.join(a.preferred_ops)}"
        ),
        (
            f"{theme_b} variables: "
            + ", ".join(b.variables)
            + f"; preferred operators: {', '.join(b.preferred_ops)}"
        ),
        (
            "Allowed operators: "
            + ", ".join(sorted(OPERATOR_CATALOG))
            + ". Use numeric operator parameters, for example GROWTH(at, 1), "
            "MA(RATIO(ib, at), 3), and LAG(capx, 1)."
        ),
        format_operator_arity_guard(),
    ])


def _format_temporal_construction_section() -> str:
    return "\n".join([
        "TEMPORAL CONSTRUCTION OPTIONS",
        "You may use current levels, levels scaled by assets or market value, "
        "first changes such as DELTA(x, 1), growth rates such as GROWTH(x, 1), "
        "smoothed states such as MA(x, 3), trends such as TREND(x, 5), "
        "volatility or second-moment ideas such as VOL(x, 5), and change-of-"
        "change ideas such as ACCEL(x) or DELTA(DELTA(x, 1), 1). Also consider "
        "distance-from-extreme ideas such as TS_MAX(x, 5) or TS_MIN(x, 5), "
        "position-in-own-history ideas such as TS_RANK(x, 5), cumulative "
        "multi-year flows such as TS_SUM(xrd, 5), and reporting-persistence "
        "counts such as TS_COUNT(xrd, 5).",
        "Do not force temporal transformations. Use them only when the "
        "economic mechanism requires a state, change, acceleration, "
        "deceleration, persistence, instability, or reversal concept.",
        "Use COALESCE(a, b) only when the missing-data convention is "
        "economically justified (for example, unreported R&D treated as "
        "zero), never as a blanket coverage patch.",
    ])


def _format_search_discipline_section() -> str:
    return "\n".join([
        "SEARCH DISCIPLINE",
        "Use prior winners as evidence, not templates. Favor proposals that "
        "test distinguishable economic mechanisms rather than minor "
        "restatements of the same formula family.",
        "Near-duplicates are useful only when they isolate a specific design "
        "choice, such as timing, scaling, conditioning, or operator structure. "
        "Otherwise, prefer a proposal that changes the economic channel while "
        "remaining plausible and parsimonious.",
        "When proposing a batch, make the set informative as a whole: avoid "
        "several signals that would teach us essentially the same lesson if "
        "evaluated.",
        "Do not add novelty for its own sake. Each proposal should still have "
        "a clear hypothesis and a reason it could improve on the archive.",
    ])


def _format_diagnostic_line(diagnostic: object) -> str:
    code = getattr(diagnostic, "code", "unknown")
    severity = getattr(diagnostic, "severity", "info")
    message = getattr(diagnostic, "message", str(diagnostic))
    proposal_name = getattr(diagnostic, "proposal_name", None)
    if proposal_name:
        return (
            f"role=validator phase=validation proposal={proposal_name} "
            f"code={code} severity={severity} message={message}"
        )
    return f"role=parser phase=parse code={code} severity={severity} message={message}"


def _format_run_position_section(context: PromptContext) -> str:
    human_round = context.generation + 1
    remaining = context.max_generations - context.generation - 1
    return "\n".join([
        "RUN POSITION",
        f"- generation_index: {context.generation}",
        f"- human_round: {human_round} of {context.max_generations}",
        f"- remaining_after_this_generation: {remaining}",
    ])


def _format_public_feedback_section(context: PromptContext) -> str:
    lines = ["PUBLIC EVENT HISTORY"]
    notes = context.archive.critic_notes[-12:]
    parse_diagnostics = context.archive.parse_diagnostics[-12:]
    validation_diagnostics = context.archive.validation_diagnostics[-12:]

    if not notes and not parse_diagnostics and not validation_diagnostics:
        lines.append("- None yet.")
        return "\n".join(lines)

    if notes:
        lines.append("Critic notes:")
        for note in notes:
            lines.append(
                f"- gen={note.generation} role=critic phase=critique "
                f"target={note.target_name} verdict={note.verdict} "
                f"score={note.score} rationale={note.rationale}"
            )
    if parse_diagnostics:
        lines.append("Parser diagnostics:")
        for diagnostic in parse_diagnostics:
            lines.append(f"- {_format_diagnostic_line(diagnostic)}")
    if validation_diagnostics:
        lines.append("Validation diagnostics:")
        for diagnostic in validation_diagnostics:
            lines.append(f"- {_format_diagnostic_line(diagnostic)}")

    return "\n".join(lines)


def _initial_prompt(context: PromptContext, *, n_proposals: int) -> str:
    task = context.task
    metadata = task.metadata
    return "\n\n".join([
        task.objective.title,
        task.objective.description,
        f"Theme A: {metadata['theme_a']}",
        f"Theme B: {metadata['theme_b']}",
        f"Economic story: {metadata['economic_story']}",
        format_evaluation_stat_definitions(task.evaluation_protocol),
        format_evaluation_search_objective(task.evaluation_protocol),
        _format_run_position_section(context),
        _format_theme_resource_section(context),
        _format_temporal_construction_section(),
        _format_reference_section(context),
        (
            f"Propose {n_proposals} NEW signals. Each signal must use variables "
            "from both themes and express a real economic interaction, not a "
            "single-theme variant."
        ),
        _format_output_schema(context),
    ])


def _reflection_prompt(context: PromptContext, *, n_proposals: int) -> str:
    return "\n\n".join([
        f"GENERATION {context.generation} SIGNAL DISCOVERY",
        context.task.objective.description,
        format_evaluation_stat_definitions(context.task.evaluation_protocol),
        format_evaluation_search_objective(context.task.evaluation_protocol),
        _format_archive_section(context.archive.evaluated),
        _format_run_position_section(context),
        _format_public_feedback_section(context),
        _format_theme_resource_section(context),
        _format_temporal_construction_section(),
        _format_reference_section(context),
        _format_search_discipline_section(),
        (
            f"Propose {n_proposals} NEW signals. Repair at least one prior "
            "failure mode when useful, and explore at least one distinct "
            "mechanism."
        ),
        _format_output_schema(context),
    ])


@dataclass(frozen=True)
class CrossThemePromptRenderer:
    n_proposals: int

    def render(self, context: PromptContext) -> PromptRenderResult:
        if context.generation == 0:
            user_message = _initial_prompt(context, n_proposals=self.n_proposals)
        else:
            user_message = _reflection_prompt(context, n_proposals=self.n_proposals)
        user_message += _intent_block(context.intent)
        if context.guidance:
            user_message += _guidance_block(context.guidance)

        return PromptRenderResult(
            system_message=context.role.instructions,
            user_message=user_message,
            cache=True,
            metadata={
                "task_id": context.task.task_id,
                "generation": context.generation,
                "role": context.role.name,
                "intent": context.intent,
            },
        )
