"""Return value for Proposer.propose() — proposals plus usage + per-call traces.

The encrypted thinking signature flows through the multi-turn API via the
Proposer's internal state (e.g. `_prior_turns`, `_persona_turns`,
`_peer_turns`); it is NOT stored in LLMCallTrace. The LLMCallTrace captures
the *summarized* thinking text (decrypted, human-readable) plus the prompt
and response for each underlying LLM call.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from siglab.lab.archive import Proposal, CriticNote, ModeratorNote


@dataclass
class LLMCallTrace:
    """Audit record for one LLM call.

    - `role`: "proposer", "critic", "moderator", "persona_ratio_specialist",
      "peer_empiricist", etc. Multi-agent topologies set this to identify
      which agent produced which text.
    - `prompt`: the user-message text sent to the LLM (final turn only;
      prior turns are implicit from the multi-turn history).
    - `response`: the LLM's visible text reply.
    - `thinking`: provider-returned summarized reasoning/thinking text (NOT
      Anthropic encrypted signatures — those stay in the multi-turn session).
    - `round`: for Socratic, 1 or 2; None otherwise.
    """
    role: str
    prompt: str
    response: str
    thinking: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    round: int | None = None


@dataclass
class ProposerResult:
    """What a Proposer returns each generation.

    - `proposals`: the K signals that should flow to the Evaluator.
    - `considered`: proposals seen but not selected. Every topology
      (single_agent/socratic/debate) routes `enforce_proposal_count`'s
      truncation overflow here when the LLM returns more than the
      requested n; Debate adds its N×M − K non-selected candidates on top,
      and Socratic adds its Round-1 outputs. proposer_critic has no count
      enforcement and never populates this (always empty).
    - `critic_notes`: per-proposal verdicts from a critic role (empty for
      single-agent; a small number for proposer_critic; one per candidate
      for Debate).
    - `moderator_note`: moderator's selection rationale (Debate only).
    - `llm_calls`: one `LLMCallTrace` per underlying LLM call this gen.
    - `parse_diagnostics`: parser status for model outputs before validation.
    - `input_tokens`/`output_tokens`/`cached_input_tokens`/
      `cache_creation_input_tokens`: summed across all LLM calls made in this
      generation, for Budget accounting.
    - `n_llm_calls`: number of LLM calls made (1 for single_agent, 2-3 for
      proposer_critic, ~5 for debate, ~6 for socratic).
    """
    proposals: list[Proposal]
    considered: list[Proposal] = field(default_factory=list)
    critic_notes: list[CriticNote] = field(default_factory=list)
    moderator_note: ModeratorNote | None = None
    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    parse_diagnostics: list = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    n_llm_calls: int = 0
