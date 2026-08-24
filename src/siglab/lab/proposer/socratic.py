"""Socratic topology — peer-flat multi-agent (3 peers × 2 rounds).

Per generation:
  Round 1: 3 peers propose independently in parallel; none sees the others.
  Round 2: each peer sees ALL round-1 proposals (with authorship), produces
           a brief critique of peers and its final K/N proposals — may be
           revisions of own Round 1, syntheses, or wholly new ideas.

Aggregation: union of Round-2 outputs, dedup by expression string. No
moderator role; no explicit critic role (everyone critiques).

Each peer keeps its own session across generations AND across rounds within
a generation (Round 2 is a continuation of the same session thread as
Round 1 for that peer).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from siglab.lab.archive import Proposal


# -- Peers ---------------------------------------------------------------

@dataclass(frozen=True)
class _Peer:
    name: str
    system_prompt: str


# Shared "zero-sum niche" clause appended to every peer's persona. Persona
# steering alone is too weak to override the archive's pull toward the
# obvious next step; an explicit panel-diversity constraint reframes each
# peer's task from "find the best signal" to "occupy a branch peers won't".
_PANEL_DIVERSITY_CLAUSE = (
    "\n\n"
    "CRITICAL PANEL RULE — panel diversity is a hard constraint. Your "
    "expression must occupy a mechanism niche that no other peer in this "
    "panel would independently reach from the same evidence. Treat the "
    "panel as zero-sum: if you reach for the obvious next step the archive "
    "suggests, your peers are reaching for the same thing, and the panel "
    "fails. If your draft looks like a trivial variable swap of what a "
    "plausible peer would propose (e.g. `ppent` ↔ `ppegt`, `dp` ↔ `dpc`, "
    "level ↔ growth of the same quantity), pick a structurally different "
    "mechanism family instead — even if you judge the 'obvious' branch as "
    "stronger. Your job is not to find the single best signal; it is to "
    "ensure the panel covers distinct mechanism branches."
)


SOCRATIC_PEERS: tuple[_Peer, ...] = (
    _Peer(
        name="empiricist",
        system_prompt=(
            "You are an empirical asset-pricing researcher. You weight prior "
            "empirical evidence heavily — you trust signals that have worked "
            "in adjacent contexts and are skeptical of purely theoretical "
            "constructs without empirical track record. When you revise, you "
            "tend toward variations of mechanisms the literature has "
            "validated."
        ) + _PANEL_DIVERSITY_CLAUSE,
    ),
    _Peer(
        name="theorist",
        system_prompt=(
            "You are a theoretical asset-pricing researcher. You prioritize "
            "mechanisms with clear economic intuition over mechanisms with "
            "strong empirical fit but murky theory. You are willing to "
            "propose signals that the literature has not yet tested if the "
            "economic story is compelling. When you revise, you sharpen the "
            "economic mechanism, sometimes at the cost of empirical support."
        ) + _PANEL_DIVERSITY_CLAUSE,
    ),
    _Peer(
        name="pragmatist",
        system_prompt=(
            "You are a practical quantitative researcher. You care most about "
            "whether a signal is implementable and robust: stable coverage, "
            "clear sign prediction, no weird numerical behavior. You are "
            "willing to sacrifice some theoretical or empirical purity for "
            "robustness and simplicity. When you revise, you aim for signals "
            "that would survive transaction costs and regime changes."
        ) + _PANEL_DIVERSITY_CLAUSE,
    ),
)


# -- Prompt builders -----------------------------------------------------

def _build_round2_prompt(
    round1_proposals: list[Proposal],
    my_name: str,
    k_per_peer: int,
) -> str:
    lines = [
        "META-FRAMING FOR ROUND 2 — read before anything else.",
        "",
        "If you see that ≥2 peers (possibly including you) produced the same "
        "or near-same expression in Round 1, that is prima facie evidence of "
        "PANEL FAILURE: identical evidence set plus identical model class "
        "pulls peers toward identical conclusions. It is NOT independent "
        "validation, and it is NOT 'remarkable convergence that strongly "
        "validates the idea' — the peers share an archive, a pair story, "
        "and a variable catalog, so any 'independence' between them is a "
        "prompt artifact, not a sampling guarantee. Re-submitting a "
        "converged Round-1 expression (or a trivial variable-swap of one, "
        "e.g. `ppent` ↔ `ppegt`, `dp` ↔ `dpc`, level ↔ growth of the same "
        "quantity) counts as a diversity failure, not as robustness "
        "confirmation.",
        "",
        "Your Round-2 job is to BRANCH into a mechanism family the panel "
        "has not covered, even if you judge a converged expression to be "
        "'the best one'. If no peer covered a genuinely different branch "
        "in Round 1, it is your job in Round 2 to supply one.",
        "",
        f"You are {my_name}. You have now seen every peer's Round 1 proposals. "
        f"Your task for Round 2 is to:",
        "",
        "  (a) Briefly critique each peer's proposals (1 sentence each, "
        "identifying their strongest or weakest point — be honest). If you "
        "observe convergence, name it explicitly as a panel failure.",
        f"  (b) Submit your final {k_per_peer} proposals. These should "
        "branch into mechanism families not yet covered by the panel. They "
        "may be syntheses that borrow structure from peers, or entirely "
        "new proposals informed by what you've seen. Do not re-submit a "
        "converged Round-1 expression or a trivial variable-swap of one.",
        "",
        "-- ROUND 1 PROPOSALS FROM ALL PEERS --",
        "",
    ]
    for p in round1_proposals:
        lines.extend([
            f"[{p.author}] {p.name}",
            f"  EXPRESSION: {p.expression}",
            f"  HYPOTHESIS: {p.hypothesis}",
            f"  EXPECTED_SIGN: {p.expected_sign}",
            f"  REASONING: {p.reasoning}",
            "",
        ])
    lines.extend([
        "Output format — EXACTLY:",
        "",
        "CRITIQUES:",
        "[peer_name/proposal_name]: <one-sentence critique>",
        "... (one line per proposal)",
        "",
        "FINAL PROPOSALS:",
        "",
        "---SIGNAL 1---",
        "NAME: <name>",
        "EXPRESSION: <expression>",
        "HYPOTHESIS: <hypothesis>",
        "EXPECTED_SIGN: <positive or negative>",
        "INTERACTION_TYPE: <composite | conditional | ratio | novel>",
        "REASONING: <2-3 sentences>",
        "",
        f"(repeat up to {k_per_peer} times)",
    ])
    return "\n".join(lines)


# -- Orchestration class -------------------------------------------------

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import field

from siglab.lab.archive import SignalArchive
from siglab.lab.llm.base import LLMClient, LLMSession
from siglab.lab.parsing import CrossThemeProposalParser
from siglab.lab.proposer._result import ProposerResult, LLMCallTrace
from siglab.lab.proposer._enforce import enforce_proposal_count
from siglab.lab.reflector.base import Reflector
from siglab.lab.task import DiscoveryTask


def _extract_final_proposals_block(text: str) -> str:
    """Return everything after 'FINAL PROPOSALS:' (case-insensitive)."""
    m = re.search(r"FINAL\s+PROPOSALS\s*:", text, re.IGNORECASE)
    if not m:
        return text
    return text[m.end():]


@dataclass
class SocraticProposer:
    """Peer-flat topology: 3 peers × 2 rounds.

    Each peer keeps a long-lived session across generations. Round 2 is a
    continuation of the same session thread as Round 1 within a generation
    (so the peer's thinking chain spans both rounds). Each call emits an
    LLMCallTrace with role=f"peer_{peer.name}" and round=1|2.
    """
    llm: LLMClient
    reflector: Reflector
    k_per_peer: int = 2           # per-peer Round-2 output size
    m_per_peer_r1: int = 2        # per-peer Round-1 proposals
    _peer_sessions: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self._peer_sessions:
            for p in SOCRATIC_PEERS:
                self._peer_sessions[p.name] = self.llm.start_session(
                    system=p.system_prompt,
                )

    def _call_peer(
        self, peer: "_Peer", user_prompt: str, round_num: int,
    ) -> tuple:
        """Single peer call — returns (response_obj, trace)."""
        session: LLMSession = self._peer_sessions[peer.name]
        resp = session.send(user_prompt, cache=True)
        trace = LLMCallTrace(
            role=f"peer_{peer.name}",
            prompt=user_prompt, response=resp.text, thinking=resp.thinking,
            input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            cached_input_tokens=resp.cached_input_tokens, round=round_num,
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
        r1_prompt = summary.text

        traces: list[LLMCallTrace] = []
        total_in = 0
        total_out = 0
        total_cached = 0
        total_cache_creation = 0
        r1_proposals_by_peer: dict[str, list[Proposal]] = {}
        parse_diagnostics = []

        # -- Round 1: parallel independent proposals ------------------------
        with ThreadPoolExecutor(max_workers=len(SOCRATIC_PEERS)) as pool:
            futures = {
                pool.submit(self._call_peer, peer, r1_prompt, 1): peer
                for peer in SOCRATIC_PEERS
            }
            for future in futures:
                peer = futures[future]
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
                r1_proposals_by_peer[peer.name] = [Proposal(
                    name=p.name, expression=p.expression, hypothesis=p.hypothesis,
                    expected_sign=p.expected_sign, reasoning=p.reasoning,
                    interaction_type=p.interaction_type,
                    theme_a=p.theme_a, theme_b=p.theme_b, author=peer.name,
                ) for p in parsed.proposals]

        all_r1 = [p for peer_list in r1_proposals_by_peer.values() for p in peer_list]

        # -- Round 2: cross-critique + revise (parallel) --------------------
        r2_proposals_by_peer: dict[str, list[Proposal]] = {}

        def _peer_r2(peer):
            r2_prompt = _build_round2_prompt(
                round1_proposals=all_r1, my_name=peer.name,
                k_per_peer=self.k_per_peer,
            )
            return peer, self._call_peer(peer, r2_prompt, 2)

        with ThreadPoolExecutor(max_workers=len(SOCRATIC_PEERS)) as pool:
            for peer, (resp, trace) in pool.map(_peer_r2, SOCRATIC_PEERS):
                traces.append(trace)
                total_in += resp.input_tokens
                total_out += resp.output_tokens
                total_cached += resp.cached_input_tokens
                total_cache_creation += resp.cache_creation_input_tokens

                final_block = _extract_final_proposals_block(resp.text)
                parsed = CrossThemeProposalParser().parse(
                    final_block, task=task, generation=generation,
                )
                parse_diagnostics.extend(parsed.diagnostics)
                r2_proposals_by_peer[peer.name] = [Proposal(
                    name=p.name, expression=p.expression, hypothesis=p.hypothesis,
                    expected_sign=p.expected_sign, reasoning=p.reasoning,
                    interaction_type=p.interaction_type,
                    theme_a=p.theme_a, theme_b=p.theme_b, author=peer.name,
                ) for p in parsed.proposals]

        # Union with dedup by expression string
        seen_exprs: set[str] = set()
        final: list[Proposal] = []
        for peer in SOCRATIC_PEERS:
            for p in r2_proposals_by_peer.get(peer.name, []):
                key = p.expression.strip()
                if key in seen_exprs:
                    continue
                seen_exprs.add(key)
                final.append(p)

        final, overflow = enforce_proposal_count(final, n, proposer="socratic")

        return ProposerResult(
            proposals=final,
            considered=all_r1 + overflow,
            critic_notes=[],
            moderator_note=None,
            llm_calls=traces,
            parse_diagnostics=parse_diagnostics,
            input_tokens=total_in,
            output_tokens=total_out,
            cached_input_tokens=total_cached,
            cache_creation_input_tokens=total_cache_creation,
            n_llm_calls=2 * len(SOCRATIC_PEERS),  # 3 peers × 2 rounds = 6
        )
