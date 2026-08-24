"""Single-agent Proposer: one LLM call per generation, stateful session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from siglab.lab.archive import SignalArchive, Proposal
from siglab.lab.llm.base import LLMClient, LLMSession
from siglab.lab.parsing import CrossThemeProposalParser
from siglab.lab.proposer._enforce import enforce_proposal_count
from siglab.lab.proposer._result import ProposerResult, LLMCallTrace
from siglab.lab.reflector.base import Reflector
from siglab.lab.task import DiscoveryTask


@dataclass
class SingleAgentProposer:
    """Baseline topology. One LLM call per generation; thinking chain survives
    across generations via the underlying `LLMSession` (Anthropic: local raw
    blocks; OpenAI/Gemini: server-side response/interaction ids). System prompt
    is empty.
    """
    llm: LLMClient
    reflector: Reflector
    _session: LLMSession | None = field(default=None, init=False)

    def propose(
        self, *,
        archive: SignalArchive,
        task: DiscoveryTask,
        generation: int,
        n: int,
        intent: str | None = None,
        progress: Any | None = None,
    ) -> ProposerResult:
        if self._session is None:
            self._session = self.llm.start_session(system="")

        summary = self.reflector.summarize(
            archive=archive,
            task=task,
            generation=generation,
            intent=intent,
        )
        resp = self._session.send(summary.text, cache=True)

        parsed = CrossThemeProposalParser().parse(
            resp.text, task=task, generation=generation,
        )
        proposals = [Proposal(
            name=p.name, expression=p.expression, hypothesis=p.hypothesis,
            expected_sign=p.expected_sign, reasoning=p.reasoning,
            interaction_type=p.interaction_type,
            theme_a=p.theme_a, theme_b=p.theme_b, author=p.author,
        ) for p in parsed.proposals]

        proposals, overflow = enforce_proposal_count(
            proposals, n, proposer="single_agent",
        )

        trace = LLMCallTrace(
            role="proposer",
            prompt=summary.text,
            response=resp.text,
            thinking=resp.thinking,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cached_input_tokens=resp.cached_input_tokens,
            round=None,
        )

        return ProposerResult(
            proposals=proposals,
            considered=overflow,
            llm_calls=[trace],
            parse_diagnostics=parsed.diagnostics,
            input_tokens=resp.input_tokens,
            output_tokens=resp.output_tokens,
            cached_input_tokens=resp.cached_input_tokens,
            cache_creation_input_tokens=resp.cache_creation_input_tokens,
            n_llm_calls=1,
        )
