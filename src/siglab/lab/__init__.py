"""siglab.lab — modular discovery-loop components.

Public API:
    DiscoveryLoop        - the driver
    SignalArchive        - shared mutable state
    Proposal, EvaluatedSignal, CriticNote, ModeratorNote
    SingleAgentProposer  - baseline topology
    ThinkingChainReflector
    FixedGens            - baseline stopping rule
    SignalAgentAdapter   - wraps siglab.agent.SignalEvaluator
    AnthropicClient, MockLLMClient, build_llm
    Budget, Recorder
    RunSpec, load_runspecs, build_components
"""
from siglab.lab.loop import DiscoveryLoop
from siglab.lab.archive import (
    SignalArchive, Proposal, EvaluatedSignal, CriticNote, ModeratorNote,
)
from siglab.lab.proposer.single_agent import SingleAgentProposer
from siglab.lab.proposer.proposer_critic import ProposerCriticProposer
from siglab.lab.proposer.debate import DebateProposer
from siglab.lab.proposer.socratic import SocraticProposer
from siglab.lab.reflector.thinking_chain import ThinkingChainReflector
from siglab.lab.stopping.fixed import FixedGens
from siglab.lab.evaluator.signal_agent_adapter import SignalAgentAdapter
from siglab.lab.llm.anthropic import AnthropicClient
from siglab.lab.llm.mock import MockLLMClient
from siglab.lab.llm.factory import build_llm
from siglab.lab.budget import Budget
from siglab.lab.recorder import Recorder
from siglab.lab.runspec import RunSpec, load_runspecs
from siglab.lab.runner import build_components

__all__ = [
    "DiscoveryLoop",
    "SignalArchive", "Proposal", "EvaluatedSignal", "CriticNote", "ModeratorNote",
    "SingleAgentProposer", "ProposerCriticProposer",
    "DebateProposer", "SocraticProposer",
    "ThinkingChainReflector",
    "FixedGens", "SignalAgentAdapter",
    "AnthropicClient", "MockLLMClient", "build_llm",
    "Budget", "Recorder",
    "RunSpec", "load_runspecs", "build_components",
]
