"""Backward-compatible imports for the signal evaluator."""

from siglab.agent.signal_evaluator import SignalEvaluator, SignalResult

SignalAgent = SignalEvaluator

__all__ = ["SignalEvaluator", "SignalAgent", "SignalResult"]
