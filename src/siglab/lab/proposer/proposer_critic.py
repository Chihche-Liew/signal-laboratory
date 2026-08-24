"""Proposer+Critic topology: propose → critique → revise.

Three LLM calls per generation:
  1. Propose — uses the standard cross_theme prompt (via Reflector)
  2. Critique — a fresh call with a critic system prompt; reviews all N
     proposals, flags rediscoveries and hypothesis-expression disagreements
  3. Revise — flagged proposals are sent back to the proposer for revision;
     unflagged pass through unchanged
"""
from __future__ import annotations

import re
from typing import Any, Literal

from siglab.lab.archive import Proposal, CriticNote, EvaluatedSignal
from siglab.lab.prompting.metric_guidance import (
    format_evaluation_revision_objective,
    format_evaluation_search_objective,
    format_evaluation_stat_definitions,
)
from siglab.lab.prompting.operator_guidance import format_operator_arity_guard
from siglab.lab.task import EvaluationProtocol


# ── System prompts ────────────────────────────────────────────────────────

CRITIC_SYSTEM = (
    "You are a skeptical quantitative finance reviewer. Your job is to read "
    "proposed cross-theme signals and flag any that are (a) direct rediscoveries "
    "of known published anomalies, (b) internally inconsistent (the stated "
    "hypothesis doesn't match what the expression computes, or the expected sign "
    "contradicts the expression's natural direction), or (c) too trivial to be "
    "worth testing. Be honest; do not flatter. If a proposal is genuinely novel "
    "and well-constructed, accept it."
)


# ── Prompt builders ───────────────────────────────────────────────────────

def _format_evaluated_signal(signal: EvaluatedSignal) -> list[str]:
    metric_bits = [f"generation={signal.generation}"]
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

    return [
        f"  - {signal.name}: {signal.expression}",
        f"    hypothesis: {signal.hypothesis}",
        (
            f"    expected_sign={signal.expected_sign}; "
            f"interaction_type={signal.interaction_type}; "
            + "; ".join(metric_bits)
        ),
    ]


def _build_critic_prompt(
    proposals: list[Proposal],
    evaluated_signals: list[EvaluatedSignal],
    evaluation_protocol: EvaluationProtocol,
) -> str:
    lines = [
        "You will review the following cross-theme signal proposals. For each, "
        "assign a VERDICT of 'accept' or 'flag', a SCORE from 1 (worst) to 5 (best), "
        "and a one-sentence RATIONALE.",
        "",
        "Signals already evaluated in this pair's archive (do NOT re-propose - "
        "flag any rediscovery):",
        "",
        format_evaluation_stat_definitions(evaluation_protocol),
        "",
        format_evaluation_search_objective(evaluation_protocol),
    ]
    if evaluated_signals:
        for signal in evaluated_signals:
            lines.extend(_format_evaluated_signal(signal))
    else:
        lines.append("  (none yet - this is generation 0's critic pass)")
    lines.append("")
    lines.append("Proposals under review:")
    lines.append("")
    for p in proposals:
        lines.extend([
            f"NAME: {p.name}",
            f"EXPRESSION: {p.expression}",
            f"HYPOTHESIS: {p.hypothesis}",
            f"EXPECTED_SIGN: {p.expected_sign}",
            f"INTERACTION_TYPE: {p.interaction_type}",
            f"REASONING: {p.reasoning}",
            "",
        ])
    lines.extend([
        "Output format - EXACTLY this, one block per proposal:",
        "",
        "---CRITIQUE <name>---",
        "VERDICT: <accept | flag>",
        "SCORE: <1-5>",
        "RATIONALE: <one sentence>",
    ])
    return "\n".join(lines)


def _parse_critic_response(text: str, proposal_names: list[str]) -> list[CriticNote]:
    """Parse the critic's response into a list of CriticNote.

    Only returns notes whose target_name is in `proposal_names` (defensive).
    Unparseable blocks are silently skipped.
    """
    notes: list[CriticNote] = []
    blocks = re.split(
        r"(?m)^---+\s*CRITIQUE\s+(.+?)\s*---+\s*$",
        text,
        flags=re.IGNORECASE,
    )
    for i in range(1, len(blocks) - 1, 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        if name not in proposal_names:
            continue
        verdict_m = re.search(r"VERDICT\s*:\s*(accept|flag)", body, re.IGNORECASE)
        score_m = re.search(r"SCORE\s*:\s*(\d+)", body, re.IGNORECASE)
        rationale_m = re.search(
            r"RATIONALE\s*:\s*(.+?)(?:\n\s*\n|$)", body, re.IGNORECASE | re.DOTALL
        )
        if not verdict_m:
            continue
        verdict: Literal["accept", "flag"] = verdict_m.group(1).lower()  # type: ignore
        score = int(score_m.group(1)) if score_m else 3
        rationale = rationale_m.group(1).strip() if rationale_m else ""
        notes.append(CriticNote(
            generation=0,  # filled in by caller
            target_name=name,
            verdict=verdict,
            score=score,
            rationale=rationale,
        ))
    return notes


def _record_progress(
    progress: Any | None,
    event: str,
    *,
    stage: str,
    generation: int,
    metadata: dict | None = None,
) -> None:
    if progress is not None:
        try:
            progress.record(
                event,
                stage=stage,
                generation=generation,
                metadata=metadata or {},
            )
        except Exception:
            pass


def _build_revise_prompt(
    accepted: list[Proposal],
    flagged: list[tuple[Proposal, str]],
    evaluation_protocol: EvaluationProtocol,
) -> str:
    """Build a prompt asking the proposer to revise flagged proposals.

    The proposer sees: accepted proposals (for context), flagged proposals
    with the critic's rationale, and an instruction to revise only the
    flagged ones.
    """
    lines = [
        "A critic has reviewed your proposals. Revise the FLAGGED ones below, "
        "addressing each critique. Keep the accepted proposals as-is — do not "
        "re-submit them here.",
        "",
        format_evaluation_revision_objective(evaluation_protocol),
        "",
    ]
    if accepted:
        lines.append("ACCEPTED (keep as-is, do not revise):")
        for p in accepted:
            lines.append(f"  - {p.name}: {p.expression}")
        lines.append("")
    lines.append("FLAGGED (revise each):")
    for p, critique in flagged:
        lines.extend([
            f"  Original name: {p.name}",
            f"  Original expression: {p.expression}",
            f"  Original hypothesis: {p.hypothesis}",
            f"  Critic's concern: {critique}",
            "",
        ])
    lines.extend([
        "For each flagged proposal, produce a revised version in the standard "
        "format:",
        "",
        "---SIGNAL 1---",
        "NAME: <name>",
        "EXPRESSION: <expression>",
        "HYPOTHESIS: <hypothesis>",
        "EXPECTED_SIGN: <positive or negative>",
        "INTERACTION_TYPE: <composite | conditional | ratio | novel>",
        "REASONING: <2-3 sentences>",
        "",
        "Formatting constraints:",
        "- The schema field labels must start at the beginning of the line exactly as shown.",
        "- Do not prefix field labels with bullets, numbering, Markdown, or extra punctuation.",
        "- Never write '- NAME:', '- EXPRESSION:', or any other bulleted field label.",
        "",
        format_operator_arity_guard(),
        "",
        "---SIGNAL 2---",
        "...",
    ])
    return "\n".join(lines)


# ── Orchestration class ───────────────────────────────────────────────────

from dataclasses import dataclass

from siglab.lab.archive import SignalArchive
from siglab.lab.llm.base import LLMClient, LLMSession
from siglab.lab.parsing import CrossThemeProposalParser
from siglab.lab.proposer._result import ProposerResult, LLMCallTrace
from siglab.lab.reflector.base import Reflector
from siglab.lab.task import DiscoveryTask


@dataclass(init=False)
class ProposerCriticProposer:
    """Two-stage topology: propose → critique → revise.

    - Proposer session is long-lived across generations — the revise turn
      is a continuation of the same session as propose, so critique context
      stays in the proposer conversation.
    - Critic is a one-shot session per generation (fresh system prompt,
      no history beyond the current proposal batch).
    - `critic_llm` may be a different provider/model from `proposer_llm`.
    """
    proposer_llm: LLMClient
    reflector: Reflector
    critic_llm: LLMClient
    _proposer_session: LLMSession | None

    def __init__(
        self,
        *,
        reflector: Reflector,
        llm: LLMClient | None = None,
        proposer_llm: LLMClient | None = None,
        critic_llm: LLMClient | None = None,
    ) -> None:
        if proposer_llm is None:
            if llm is None:
                raise TypeError("ProposerCriticProposer requires llm or proposer_llm")
            proposer_llm = llm
        self.proposer_llm = proposer_llm
        self.critic_llm = critic_llm or proposer_llm
        self.reflector = reflector
        self._proposer_session = None

    def propose(
        self, *,
        archive: SignalArchive,
        task: DiscoveryTask,
        generation: int,
        n: int,
        intent: str | None = None,
        progress: Any | None = None,
    ) -> ProposerResult:
        traces: list[LLMCallTrace] = []
        if self._proposer_session is None:
            self._proposer_session = self.proposer_llm.start_session(system="")

        # ── 1. Propose ─────────────────────────────────────────────────────
        summary = self.reflector.summarize(
            archive=archive,
            task=task,
            generation=generation,
            intent=intent,
        )
        _record_progress(
            progress,
            "llm_propose_started",
            stage="llm_propose",
            generation=generation,
            metadata={"n_requested": n},
        )
        propose_resp = self._proposer_session.send(summary.text, cache=True)
        traces.append(LLMCallTrace(
            role="proposer", prompt=summary.text, response=propose_resp.text,
            thinking=propose_resp.thinking,
            input_tokens=propose_resp.input_tokens,
            output_tokens=propose_resp.output_tokens,
            cached_input_tokens=propose_resp.cached_input_tokens, round=None,
        ))

        parsed = CrossThemeProposalParser().parse(
            propose_resp.text, task=task, generation=generation,
        )
        parse_diagnostics = list(parsed.diagnostics)

        initial_proposals = [Proposal(
            name=p.name, expression=p.expression, hypothesis=p.hypothesis,
            expected_sign=p.expected_sign, reasoning=p.reasoning,
            interaction_type=p.interaction_type,
            theme_a=p.theme_a, theme_b=p.theme_b, author="proposer",
        ) for p in parsed.proposals]
        _record_progress(
            progress,
            "llm_propose_completed",
            stage="llm_propose",
            generation=generation,
            metadata={
                "n_initial_proposals": len(initial_proposals),
                "input_tokens": propose_resp.input_tokens,
                "output_tokens": propose_resp.output_tokens,
            },
        )

        # ── 2. Critique ────────────────────────────────────────────────────
        critic_prompt = _build_critic_prompt(
            initial_proposals,
            archive.evaluated,
            task.evaluation_protocol,
        )
        critic_session = self.critic_llm.start_session(system=CRITIC_SYSTEM)
        _record_progress(
            progress,
            "llm_critic_started",
            stage="llm_critic",
            generation=generation,
            metadata={"n_initial_proposals": len(initial_proposals)},
        )
        critic_resp = critic_session.send(critic_prompt, cache=True)
        traces.append(LLMCallTrace(
            role="critic", prompt=critic_prompt, response=critic_resp.text,
            thinking=critic_resp.thinking,
            input_tokens=critic_resp.input_tokens,
            output_tokens=critic_resp.output_tokens,
            cached_input_tokens=critic_resp.cached_input_tokens, round=None,
        ))
        proposal_names = [p.name for p in initial_proposals]
        notes = _parse_critic_response(critic_resp.text, proposal_names)
        for nt in notes:
            nt.generation = generation
        n_flagged = sum(1 for note in notes if note.verdict == "flag")
        _record_progress(
            progress,
            "llm_critic_completed",
            stage="llm_critic",
            generation=generation,
            metadata={
                "n_critic_notes": len(notes),
                "n_flagged": n_flagged,
                "input_tokens": critic_resp.input_tokens,
                "output_tokens": critic_resp.output_tokens,
            },
        )

        # ── 3. Revise (only if something is flagged) ──────────────────────
        flagged_names = {n.target_name for n in notes if n.verdict == "flag"}
        accepted = [p for p in initial_proposals if p.name not in flagged_names]
        flagged_with_rationale = [
            (p, next(n.rationale for n in notes if n.target_name == p.name))
            for p in initial_proposals if p.name in flagged_names
        ]

        revise_tokens_in = 0
        revise_tokens_out = 0
        revise_cached_in = 0
        revise_cache_creation_in = 0
        revise_calls = 0
        revised: list[Proposal] = []
        if flagged_with_rationale:
            revise_prompt = _build_revise_prompt(
                accepted,
                flagged_with_rationale,
                task.evaluation_protocol,
            )
            _record_progress(
                progress,
                "llm_revise_started",
                stage="llm_revise",
                generation=generation,
                metadata={"n_flagged": len(flagged_with_rationale)},
            )
            revise_resp = self._proposer_session.send(revise_prompt, cache=True)
            traces.append(LLMCallTrace(
                role="proposer", prompt=revise_prompt, response=revise_resp.text,
                thinking=revise_resp.thinking,
                input_tokens=revise_resp.input_tokens,
                output_tokens=revise_resp.output_tokens,
                cached_input_tokens=revise_resp.cached_input_tokens, round=None,
            ))
            revised_parsed = CrossThemeProposalParser().parse(
                revise_resp.text, task=task, generation=generation,
            )
            parse_diagnostics.extend(revised_parsed.diagnostics)
            revised = [Proposal(
                name=p.name, expression=p.expression, hypothesis=p.hypothesis,
                expected_sign=p.expected_sign, reasoning=p.reasoning,
                interaction_type=p.interaction_type,
                theme_a=p.theme_a, theme_b=p.theme_b, author="proposer",
            ) for p in revised_parsed.proposals]
            revise_tokens_in = revise_resp.input_tokens
            revise_tokens_out = revise_resp.output_tokens
            revise_cached_in = revise_resp.cached_input_tokens
            revise_cache_creation_in = revise_resp.cache_creation_input_tokens
            revise_calls = 1
            _record_progress(
                progress,
                "llm_revise_completed",
                stage="llm_revise",
                generation=generation,
                metadata={
                    "n_revised_proposals": len(revised),
                    "input_tokens": revise_resp.input_tokens,
                    "output_tokens": revise_resp.output_tokens,
                },
            )

        final_proposals = accepted + revised

        return ProposerResult(
            proposals=final_proposals,
            considered=[],
            critic_notes=notes,
            moderator_note=None,
            llm_calls=traces,
            parse_diagnostics=parse_diagnostics,
            input_tokens=propose_resp.input_tokens + critic_resp.input_tokens + revise_tokens_in,
            output_tokens=propose_resp.output_tokens + critic_resp.output_tokens + revise_tokens_out,
            cached_input_tokens=(
                propose_resp.cached_input_tokens
                + critic_resp.cached_input_tokens
                + revise_cached_in
            ),
            cache_creation_input_tokens=(
                propose_resp.cache_creation_input_tokens
                + critic_resp.cache_creation_input_tokens
                + revise_cache_creation_in
            ),
            n_llm_calls=2 + revise_calls,
        )
