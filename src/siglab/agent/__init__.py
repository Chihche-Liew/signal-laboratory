"""Signal discovery and evaluation framework.

Modules:
- variables: 66-variable Compustat catalog with metadata
- operators: signal-construction operator library in 5 tiers
- themes: 8 economic themes for directed search
- executor: SignalEngine - recursive descent expression evaluator
- signal_evaluator: SignalEvaluator - evaluates proposed signal expressions
- signal_agent: SignalAgent - compatibility alias for SignalEvaluator
"""

from siglab.agent.executor import SignalEngine
from siglab.agent.signal_evaluator import SignalEvaluator, SignalResult
from siglab.agent.signal_agent import SignalAgent

__all__ = [
    "SignalEngine",
    "SignalEvaluator",
    "SignalAgent",
    "SignalResult",
]
