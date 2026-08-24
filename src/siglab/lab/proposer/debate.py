"""Debate topology — hierarchical multi-agent (3 personas + critic + moderator).

Per generation:
  1. Three persona proposers run in parallel. Each has a distinct system
     prompt focused on a different mechanism family (ratios / industry
     adjustment / temporal signals). Each carries its own session across
     generations.
  2. A stateless critic scores every proposal from 1-5 and writes a one-line
     rationale.
  3. A stateless moderator picks the top K proposals, balancing critic
     scores with mechanism diversity.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from siglab.lab.archive import Proposal


# -- Personas ------------------------------------------------------------

@dataclass(frozen=True)
class _Persona:
    name: str
    system_prompt: str


DEBATE_PERSONAS: tuple[_Persona, ...] = (
    _Persona(
        name="ratio_specialist",
        system_prompt=(
            "You are a quantitative finance researcher specializing in RATIO-type "
            "cross-theme signals. You favor expressions that directly combine "
            "variables from two themes into a single ratio (e.g. profit per dollar "
            "invested, cash per dollar financed). You are skeptical of pure "
            "composites (SUMRANK) and believe the cleanest cross-theme signals "
            "are those where the interaction appears in the numerator/denominator."
        ),
    ),
    _Persona(
        name="industry_adjuster",
        system_prompt=(
            "You are a quantitative finance researcher specializing in INDUSTRY-"
            "RELATIVE cross-theme signals. You believe that raw accounting ratios "
            "are noisy because of industry-specific baselines, and the strongest "
            "signals are those that apply IND_ADJ, IND_ZSCORE, or within-industry "
            "cross-sectional standardization to capture deviation from industry "
            "norms. You gravitate toward wrapping cross-theme ratios in industry "
            "adjustment."
        ),
    ),
    _Persona(
        name="temporal_specialist",
        system_prompt=(
            "You are a quantitative finance researcher specializing in TEMPORAL "
            "cross-theme signals. You believe that the best predictors are not "
            "static ratios but changes, trends, accelerations, or smoothed "
            "versions across time. You favor GROWTH, DELTA, TREND, MA, ACCEL "
            "applied to cross-theme combinations."
        ),
    ),
)


# -- Prompt builders -----------------------------------------------------

def _build_propose_prompt(reflector_summary_text: str, n: int) -> str:
    """The per-persona propose prompt is just the reflector summary - personas
    influence OUTPUT via system prompt, not prompt body."""
    return reflector_summary_text


DEBATE_CRITIC_SYSTEM = (
    "You are a skeptical cross-theme signal reviewer. You will be shown "
    "a batch of signals proposed by multiple specialists and asked to "
    "score each on a 1-5 scale. A 1 is trivial, redundant, or incoherent; "
    "a 5 is genuinely novel, economically well-motivated, and well-matched "
    "to the stated hypothesis. Give scores honestly; do not cluster them."
)


def _build_debate_critic_prompt(
    proposals: list[Proposal], evaluated_names: list[str],
) -> str:
    lines = [
        "Score each proposal on a 1-5 scale with a one-sentence rationale.",
        "",
        "Already-evaluated signals in the archive (do NOT reward rediscoveries):",
    ]
    if evaluated_names:
        for n in evaluated_names:
            lines.append(f"  - {n}")
    else:
        lines.append("  (none - generation 0)")
    lines.append("")
    lines.append("Proposals under review:")
    lines.append("")
    for p in proposals:
        lines.extend([
            f"NAME: {p.name}  [author: {p.author}]",
            f"EXPRESSION: {p.expression}",
            f"HYPOTHESIS: {p.hypothesis}",
            f"EXPECTED_SIGN: {p.expected_sign}",
            f"INTERACTION_TYPE: {p.interaction_type}",
            "",
        ])
    lines.extend([
        "Output format - one block per proposal, EXACTLY:",
        "",
        "---SCORE <name>---",
        "SCORE: <1-5>",
        "RATIONALE: <one sentence>",
    ])
    return "\n".join(lines)


def _parse_debate_critic_response(
    text: str, proposal_names: list[str],
) -> dict[str, tuple[int, str]]:
    """Return {name: (score, rationale)} for every name seen. Unparseable
    blocks are skipped."""
    out: dict[str, tuple[int, str]] = {}
    blocks = re.split(
        r"(?m)^---+\s*SCORE\s+(.+?)\s*---+\s*$",
        text,
        flags=re.IGNORECASE,
    )
    for i in range(1, len(blocks) - 1, 2):
        name = blocks[i].strip()
        body = blocks[i + 1]
        if name not in proposal_names:
            continue
        sm = re.search(r"SCORE\s*:\s*(\d+)", body, re.IGNORECASE)
        rm = re.search(r"RATIONALE\s*:\s*(.+?)(?:\n\s*\n|$)", body, re.IGNORECASE | re.DOTALL)
        if not sm:
            continue
        score = int(sm.group(1))
        rationale = rm.group(1).strip() if rm else ""
        out[name] = (score, rationale)
    return out


DEBATE_MODERATOR_SYSTEM = (
    "You are a research director overseeing a team of specialist researchers. "
    "Given a scored list of proposals, select the top K balancing score AND "
    "mechanism diversity. Do not pick all high-score proposals if they share "
    "the same underlying mechanism - prefer a diverse set."
)


def _build_moderator_prompt(
    proposals: list[Proposal],
    critic_scores: dict[str, tuple[int, str]],
    archive_top: list[str],
    k: int,
) -> str:
    lines = [
        f"Select the top {k} proposals below, balancing score with mechanism "
        "diversity (don't pick K near-duplicates).",
        "",
    ]
    if archive_top:
        lines.append("Current archive survivors (avoid duplicating their mechanisms):")
        for n in archive_top:
            lines.append(f"  - {n}")
        lines.append("")
    lines.append("Proposals:")
    lines.append("")
    for p in proposals:
        score, rat = critic_scores.get(p.name, (None, ""))
        lines.extend([
            f"NAME: {p.name}  [author: {p.author}]",
            f"  EXPRESSION: {p.expression}",
            f"  HYPOTHESIS: {p.hypothesis}",
            f"  CRITIC_SCORE: {score}",
            f"  CRITIC_RATIONALE: {rat}",
            "",
        ])
    lines.extend([
        "Output format - EXACTLY:",
        "",
        "SELECTED: <comma-separated list of K names>",
        "RATIONALE: <one short paragraph explaining the tradeoff>",
    ])
    return "\n".join(lines)


def _parse_moderator_response(text: str) -> tuple[list[str], str]:
    sel_m = re.search(r"SELECTED\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    rat_m = re.search(r"RATIONALE\s*:\s*(.+?)(?:$|\n\s*---)", text, re.IGNORECASE | re.DOTALL)
    if not sel_m:
        raise ValueError(
            "debate moderator response has no parseable 'SELECTED:' line; "
            "refusing to silently emit zero proposals for this generation. "
            f"Moderator text was:\n{text[:1000]}"
        )
    names = [n.strip() for n in sel_m.group(1).split(",") if n.strip()]
    rationale = rat_m.group(1).strip() if rat_m else ""
    return names, rationale


# -- Orchestration class -------------------------------------------------

from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from siglab.lab.archive import SignalArchive, ModeratorNote, CriticNote
from siglab.lab.llm.base import LLMClient, LLMSession
from siglab.lab.parsing import CrossThemeProposalParser
from siglab.lab.proposer._result import ProposerResult, LLMCallTrace
from siglab.lab.proposer._enforce import enforce_proposal_count
from siglab.lab.reflector.base import Reflector
from siglab.lab.task import DiscoveryTask


@dataclass
class DebateProposer:
    """Hierarchical topology: 3 personas + critic + moderator.

    Each persona keeps a long-lived session across generations; critic and
    moderator are one-shot sessions per gen.
    """
    llm: LLMClient
    reflector: Reflector
    n_proposals_per_persona: int = 2  # so 3 x 2 = 6 candidates per gen
    _persona_sessions: dict = field(default_factory=dict)

    def _ensure_session(self, persona: "_Persona") -> LLMSession:
        s = self._persona_sessions.get(persona.name)
        if s is None:
            s = self.llm.start_session(system=persona.system_prompt)
            self._persona_sessions[persona.name] = s
        return s

    def _call_persona(self, persona: "_Persona", user_prompt: str) -> tuple:
        """Single persona call — returns (response_obj, trace)."""
        session = self._ensure_session(persona)
        resp = session.send(user_prompt, cache=True)
        trace = LLMCallTrace(
            role=f"persona_{persona.name}",
            prompt=user_prompt, response=resp.text, thinking=resp.thinking,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            cached_input_tokens=resp.cached_input_tokens, round=None,
        )
        return resp, trace

    def propose(
        self, *,
        archive: SignalArchive,
        task: DiscoveryTask,
        generation: int,
        n: int,
        intent: str | None = None,
        progress: Any | None = None,
    ) -> ProposerResult:
        summary = self.reflector.summarize(
            archive=archive,
            task=task,
            generation=generation,
            intent=intent,
        )
        propose_prompt = _build_propose_prompt(summary.text, n=self.n_proposals_per_persona)

        traces: list[LLMCallTrace] = []
        total_in = 0
        total_out = 0
        total_cached = 0
        total_cache_creation = 0
        all_candidates: list[Proposal] = []
        parse_diagnostics = []

        # -- Round 1: parallel personas -------------------------------------
        with ThreadPoolExecutor(max_workers=len(DEBATE_PERSONAS)) as pool:
            future_to_persona = {
                pool.submit(self._call_persona, p, propose_prompt): p
                for p in DEBATE_PERSONAS
            }
            for future in future_to_persona:
                persona = future_to_persona[future]
                resp, trace = future.result()
                traces.append(trace)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                total_cached += resp.cached_input_tokens
                total_cache_creation += resp.cache_creation_input_tokens

                parsed = CrossThemeProposalParser().parse(
                    resp.text, task=task, generation=generation,
                )
                parse_diagnostics.extend(parsed.diagnostics)
                for p in parsed.proposals:
                    all_candidates.append(Proposal(
                        name=p.name, expression=p.expression, hypothesis=p.hypothesis,
                        expected_sign=p.expected_sign, reasoning=p.reasoning,
                        interaction_type=p.interaction_type,
                        theme_a=p.theme_a, theme_b=p.theme_b,
                        author=persona.name,
                    ))

        # -- Round 2: critic scores every candidate -------------------------
        evaluated_names = [e.name for e in archive.evaluated]
        critic_prompt = _build_debate_critic_prompt(all_candidates, evaluated_names)
        critic_session = self.llm.start_session(system=DEBATE_CRITIC_SYSTEM)
        critic_resp = critic_session.send(critic_prompt, cache=True)
        traces.append(LLMCallTrace(
            role="critic", prompt=critic_prompt, response=critic_resp.text,
            thinking=critic_resp.thinking,
            input_tokens=critic_resp.input_tokens,
            output_tokens=critic_resp.output_tokens,
            cached_input_tokens=critic_resp.cached_input_tokens, round=None,
        ))
        total_in += critic_resp.input_tokens
        total_out += critic_resp.output_tokens
        total_cached += critic_resp.cached_input_tokens
        total_cache_creation += critic_resp.cache_creation_input_tokens
        candidate_names = [p.name for p in all_candidates]
        scores = _parse_debate_critic_response(critic_resp.text, candidate_names)

        # -- Round 3: moderator selects top-K with diversity ----------------
        archive_top = [e.name for e in archive.successful()[:5]]
        mod_prompt = _build_moderator_prompt(
            proposals=all_candidates, critic_scores=scores,
            archive_top=archive_top, k=n,
        )
        mod_session = self.llm.start_session(system=DEBATE_MODERATOR_SYSTEM)
        mod_resp = mod_session.send(mod_prompt, cache=True)
        traces.append(LLMCallTrace(
            role="moderator", prompt=mod_prompt, response=mod_resp.text,
            thinking=mod_resp.thinking,
            input_tokens=mod_resp.input_tokens,
            output_tokens=mod_resp.output_tokens,
            cached_input_tokens=mod_resp.cached_input_tokens, round=None,
        ))
        total_in += mod_resp.input_tokens
        total_out += mod_resp.output_tokens
        total_cached += mod_resp.cached_input_tokens
        total_cache_creation += mod_resp.cache_creation_input_tokens

        selected_names, rationale = _parse_moderator_response(mod_resp.text)
        selected = [p for p in all_candidates if p.name in selected_names]
        if all_candidates and not selected:
            raise ValueError(
                "debate moderator SELECTED names matched none of this "
                f"generation's candidates: selected={selected_names!r}, "
                f"candidates={[p.name for p in all_candidates]!r}"
            )
        not_selected = [p for p in all_candidates if p.name not in selected_names]
        selected, overflow = enforce_proposal_count(selected, n, proposer="debate")
        not_selected = not_selected + overflow

        critic_notes: list[CriticNote] = []
        for p in all_candidates:
            score_tuple = scores.get(p.name)
            if score_tuple is None:
                continue
            score, rat = score_tuple
            verdict = "accept" if score >= 3 else "flag"
            critic_notes.append(CriticNote(
                generation=generation, target_name=p.name,
                verdict=verdict, score=score, rationale=rat,
            ))

        moderator_note = ModeratorNote(
            generation=generation,
            selected_names=[p.name for p in selected],
            rationale=rationale,
        )

        return ProposerResult(
            proposals=selected,
            considered=not_selected,
            critic_notes=critic_notes,
            moderator_note=moderator_note,
            llm_calls=traces,
            parse_diagnostics=parse_diagnostics,
            input_tokens=total_in,
            output_tokens=total_out,
            cached_input_tokens=total_cached,
            cache_creation_input_tokens=total_cache_creation,
            n_llm_calls=len(DEBATE_PERSONAS) + 2,
        )
